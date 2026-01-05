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
    def __init__(self, d_point=128, dino_model_name='dinov2_vits14', num_heads=4, dropout=0.1, load_vision_encoder=True, max_cameras=10, max_patches=256):
        """
        Args:
            d_point (int): Dimension of the input point features.
            dino_model_name (str): Name of the DINOv2 model to load from torch.hub.
            num_heads (int): Number of heads for MultiheadAttention.
            dropout (float): Dropout rate.
            load_vision_encoder (bool): Whether to load the DINOv2 model. Set to False if using precomputed features.
            max_cameras (int): Maximum number of cameras for positional embedding.
            max_patches (int): Maximum number of patches for positional embedding.
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
        
        # Positional Embeddings
        self.cam_embed = nn.Parameter(torch.randn(1, max_cameras, 1, d_point) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 1, max_patches, d_point) * 0.02)
        self.point_pos_proj = nn.Linear(3, d_point)
        
        # Normalization before Attention
        self.vis_norm = nn.LayerNorm(d_point)
        self.point_norm = nn.LayerNorm(d_point)
        
        # 3. Fusion Layer: Cross Attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_point, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # 4. Gating/Residual & Norm
        self.layer_norm = nn.LayerNorm(d_point)
        self.gate = nn.Parameter(torch.zeros(1))
        
        # Standard ImageNet normalization for DINO
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, point_features, point_coords=None, images=None, precomputed_features=None):
        """
        Args:
            point_features: (B, N, D_point) - Geometry features from PointNet.
            point_coords: (B, N, 3) - XYZ coordinates of the points.
            images: (B, K, C, H, W) - Multi-view images. K is number of cameras.
                    Images should be in range [0, 1].
            precomputed_features: (B, K, N_patches, D_dino) - Pre-computed DINO features.
        Returns:
            fused_features: (B, N, D_point) - Point features enriched with semantic info.
        """
        B, N, D_point = point_features.shape
        
        # Add Point Positional Embedding
        if point_coords is not None:
            point_pos_emb = self.point_pos_proj(point_coords)
            point_features = point_features + point_pos_emb
        
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
            
            # Get N_patches from output
            _, N_patches, _ = patch_features.shape
            
        # Reshape to (B, K, N_patches, D_dino) to apply embeddings
        visual_tokens = rearrange(patch_features, '(b k) n d -> b k n d', b=B, k=K)
        
        # Project to d_point
        visual_tokens = self.visual_proj(visual_tokens) # (B, K, N_patches, D_point)
        
        # Add Positional Embeddings
        # Slice embeddings to match current K and N_patches
        curr_k = min(K, self.cam_embed.shape[1])
        curr_n = min(N_patches, self.pos_embed.shape[2])
        
        cam_emb = self.cam_embed[:, :curr_k, :, :]
        pos_emb = self.pos_embed[:, :, :curr_n, :]
        
        # If actual K or N is larger than max, we might have an issue, but usually max is set large enough.
        # For safety, we can repeat or interpolate if needed, but slicing is standard for fixed max.
        if K > self.cam_embed.shape[1]:
             print(f"Warning: Input cameras {K} > Max cameras {self.cam_embed.shape[1]}. Embeddings will be repeated.")
             # Simple fallback: repeat
             cam_emb = torch.cat([cam_emb] * (K // self.cam_embed.shape[1] + 1), dim=1)[:, :K, :, :]
             
        visual_tokens = visual_tokens + cam_emb + pos_emb
        
        # Flatten for Attention: (B, K*N_patches, D_point)
        visual_tokens = rearrange(visual_tokens, 'b k n d -> b (k n) d')
        
        # Apply LayerNorm before Attention
        visual_tokens = self.vis_norm(visual_tokens)
        point_features_norm = self.point_norm(point_features)
        
        # --- Step 2: Fusion (Cross Attention) ---
        attn_output, _ = self.cross_attn(
            query=point_features_norm,
            key=visual_tokens,
            value=visual_tokens
        )
        
        # --- Step 3: Residual + Norm ---
        # Note: We add residual to original point_features (not normed one), standard Pre-Norm/Post-Norm variation
        # Here we use Post-Norm style for the residual block as originally implemented
        fused_features = self.layer_norm(point_features + torch.tanh(self.gate) * attn_output)
        
        return fused_features
