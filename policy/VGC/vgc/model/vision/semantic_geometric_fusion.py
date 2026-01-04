import torch
import torch.nn as nn
import torchvision.transforms as T
from einops import rearrange

class SemanticGeometricFusion(nn.Module):
    """
    Fuses 3D Point Cloud features with 2D Semantic features from a frozen DINOv2 model
    using a Cross-Attention mechanism.
    """
    def __init__(self, d_point=128, dino_model_name='dinov2_vits14', num_heads=4, dropout=0.1):
        """
        Args:
            d_point (int): Dimension of the input point features.
            dino_model_name (str): Name of the DINOv2 model to load from torch.hub.
            num_heads (int): Number of heads for MultiheadAttention.
            dropout (float): Dropout rate.
        """
        super().__init__()
        
        # 1. Visual Encoder: DINOv2
        print(f"Loading {dino_model_name} from torch.hub...")
        self.visual_encoder = torch.hub.load('facebookresearch/dinov2', dino_model_name)
        self.visual_encoder.eval()
        
        # Freeze DINO parameters
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
            
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

    def forward(self, point_features, images):
        """
        Args:
            point_features: (B, N, D_point) - Geometry features from PointNet.
            images: (B, K, C, H, W) - Multi-view images. K is number of cameras.
                    Images should be in range [0, 1].
        Returns:
            fused_features: (B, N, D_point) - Point features enriched with semantic info.
        """
        B, N, D_point = point_features.shape
        B, K, C, H, W = images.shape
        
        # --- Step 1: Vision (Frozen DINO) ---
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
