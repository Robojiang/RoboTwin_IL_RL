import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from einops import rearrange

class SemanticGeometricFusion(nn.Module):
    """
    Fuses 3D Point Cloud features with 2D Semantic features from a frozen DINOv2 model
    using a Cross-Attention mechanism.
    """
    def __init__(self, d_point=128, dino_model_name='dinov2_vits14', num_heads=4, dropout=0.1, load_vision_encoder=True):
        """
        Args:
            d_point (int): Dimension of the input point features.
            dino_model_name (str): Name of the DINOv2 model to load from torch.hub.
            num_heads (int): Number of heads for MultiheadAttention.
            dropout (float): Dropout rate.
            load_vision_encoder (bool): Whether to load the DINOv2 model. Set to False if using precomputed features.
        """
        super().__init__()
        
        # 1. Visual Encoder: DINOv2
        self.load_vision_encoder = load_vision_encoder
        if self.load_vision_encoder:
            print(f"Loading {dino_model_name}...")
            
            # Try to load from local 'assets' folder first
            # Assuming assets is in the root of the workspace or relative to this file
            # Let's try to find the workspace root
            
            # Current file: policy/VGC/vgc/model/vision/semantic_geometric_fusion.py
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up 5 levels to reach workspace root: policy/VGC/vgc/model/vision -> policy/VGC/vgc/model -> policy/VGC/vgc -> policy/VGC -> policy -> root
            workspace_root = os.path.abspath(os.path.join(current_dir, "../../../../../"))
            local_model_path = os.path.join(workspace_root, "assets", f"{dino_model_name}.pth")
            
            if os.path.exists(local_model_path):
                model_path = local_model_path
            else:
                model_path = None

            if model_path:
                print(f"Found local model file at {model_path}. Loading directly...")
                self.visual_encoder = torch.hub.load('facebookresearch/dinov2', dino_model_name, source='github', pretrained=False)
                state_dict = torch.load(model_path, map_location="cpu")
                self.visual_encoder.load_state_dict(state_dict)
                print("Local weights loaded successfully.")
            else:
                # Fallback to torch.hub with potential local cache
                print(f"Local model file not found at {local_model_path}. Trying torch.hub...")
                try:
                    self.visual_encoder = torch.hub.load('facebookresearch/dinov2', dino_model_name)
                except Exception as e:
                    print(f"Failed to load from torch.hub: {e}")
                    print("Attempting to load from local cache...")
                    try:
                        # Fallback to local cache if available
                        local_repo_path = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
                        if os.path.exists(local_repo_path):
                            print(f"Found local repo at {local_repo_path}, loading with source='local'...")
                            self.visual_encoder = torch.hub.load(local_repo_path, dino_model_name, source='local')
                        else:
                            raise FileNotFoundError(f"Local repo not found at {local_repo_path}")
                    except Exception as local_e:
                        print(f"Failed to load from local cache: {local_e}")
                        raise e
                
            self.visual_encoder.eval()
            
            # Freeze DINO parameters
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
        else:
            print("Skipping DINOv2 model loading (expecting precomputed features).")
            self.visual_encoder = None
            
        # DINOv2 ViT-S/14 feature dimension is 384
        if 'vits' in dino_model_name:
            self.d_dino = 384
        elif 'vitb' in dino_model_name:
            self.d_dino = 768
        elif 'vitl' in dino_model_name:
            self.d_dino = 1024
        elif 'vitg' in dino_model_name:
            self.d_dino = 1536
        else:
            self.d_dino = 384 
        
        # 2. Projection Layer
        self.visual_proj = nn.Linear(self.d_dino, d_point)
        
        # 3. Fusion Layer: Cross Attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_point, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # 4. Gating/Residual & Norm
        self.layer_norm = nn.LayerNorm(d_point)
        self.gate = nn.Parameter(torch.zeros(1))
        
        # Standard ImageNet normalization for DINO
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, point_features, images=None, precomputed_features=None):
        """
        Args:
            point_features: (B, N, D_point) - Geometry features from PointNet.
            images: (B, K, C, H, W) - Multi-view images. K is number of cameras.
                    Images should be in range [0, 1].
            precomputed_features: (B, K, N_patches, D_dino) - Pre-computed DINO features.
        Returns:
            fused_features: (B, N, D_point) - Point features enriched with semantic info.
        """
        B, N, D_point = point_features.shape
        
        # --- Step 1: Vision (Frozen DINO) ---
        if precomputed_features is not None:
            # Use pre-computed features
            # Shape: (B, K, N_patches, D_dino)
            # We need to flatten K and N_patches
            B, K, N_patches, D_dino = precomputed_features.shape
            patch_features = rearrange(precomputed_features, 'b k n d -> (b k) n d')
        else:
            assert images is not None, "Either images or precomputed_features must be provided"
            if self.visual_encoder is None:
                raise ValueError("Visual encoder is not loaded, but images were provided. Initialize with load_vision_encoder=True or provide precomputed_features.")
            
            B, K, C, H, W = images.shape
            images_flat = rearrange(images, 'b k c h w -> (b k) c h w')
            images_flat = self.normalize(images_flat)

            with torch.no_grad():
                features_dict = self.visual_encoder.forward_features(images_flat)
                patch_features = features_dict['x_norm_patchtokens']
            
        visual_tokens = rearrange(patch_features, '(b k) n d -> b (k n) d', b=B, k=K)
        visual_tokens = self.visual_proj(visual_tokens)
        
        # --- Step 2: Fusion (Cross Attention) ---
        attn_output, _ = self.cross_attn(
            query=point_features,
            key=visual_tokens,
            value=visual_tokens
        )
        
        # --- Step 3: Residual + Norm ---
        fused_features = self.layer_norm(point_features + torch.tanh(self.gate) * attn_output)
        
        return fused_features
