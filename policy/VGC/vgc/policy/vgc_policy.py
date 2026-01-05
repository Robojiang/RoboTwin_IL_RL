import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Add DP3 path
current_file_path = os.path.abspath(__file__)
vgc_policy_dir = os.path.dirname(current_file_path) # policy/VGC/vgc/policy
vgc_pkg_dir = os.path.dirname(vgc_policy_dir) # policy/VGC/vgc
vgc_root_dir = os.path.dirname(vgc_pkg_dir) # policy/VGC
policy_dir = os.path.dirname(vgc_root_dir) # policy

dp3_pkg_path = os.path.join(policy_dir, 'DP3', '3D-Diffusion-Policy')
sys.path.append(dp3_pkg_path)

from diffusion_policy_3d.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer

sys.path.append(vgc_root_dir)
from vgc.model.vision.semantic_geometric_fusion import SemanticGeometricFusion

class PointNetEncoder(nn.Module):
    """
    Simplified PointNet that returns per-point features.
    """
    def __init__(self, in_channels=6, out_channels=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, out_channels)
        )

    def forward(self, x):
        # x: (B, N, C)
        return self.mlp(x)

class VGCPolicy(BasePolicy):
    def __init__(self, 
                 shape_meta, 
                 noise_scheduler,
                 horizon=16, 
                 n_action_steps=8, 
                 n_obs_steps=2,
                 num_inference_steps=100,
                 obs_as_global_cond=True,
                 diffusion_step_embed_dim=256,
                 down_dims=(256, 512, 1024),
                 kernel_size=5,
                 n_groups=8,
                 condition_type="film",
                 # Custom params
                 d_point=128,
                 dino_model_name='dinov2_vits14',
                 use_pc_features=False,
                 **kwargs):
        super().__init__()
        
        self.noise_scheduler = noise_scheduler
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        
        # Parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_dim = action_shape[0]
        
        # 1. Encoders
        # PointNet
        self.pointnet = PointNetEncoder(in_channels=6, out_channels=d_point)
        
        # Semantic Geometric Fusion
        self.fusion = SemanticGeometricFusion(
            d_point=d_point, 
            dino_model_name=dino_model_name,
            load_vision_encoder=not use_pc_features
        )
        
        # Agent Pos Encoder
        agent_pos_shape = shape_meta['obs']['agent_pos']['shape']
        self.agent_pos_dim = agent_pos_shape[0]
        self.agent_pos_mlp = nn.Sequential(
            nn.Linear(self.agent_pos_dim, 64),
            nn.ReLU(),
            nn.Linear(64, d_point)
        )
        
        # 2. Heads
        # Keypose Head
        self.keypose_dim = 18 # 9 (left) + 9 (right)
        self.keypose_head = nn.Sequential(
            nn.Linear(d_point * 2, 128),
            nn.ReLU(),
            nn.Linear(128, self.keypose_dim)
        )
        
        # 3. Diffusion Model
        global_cond_dim = (d_point * 2 + self.keypose_dim) * n_obs_steps
        
        self.diffusion_unet = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
        )
        
        # Normalizers (will be set during training)
        self.normalizer = LinearNormalizer()

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def forward(self, batch):
        # Batch is a dict with 'obs', 'action', 'target_keypose'
        
        nobs = {}
        for key, value in batch['obs'].items():
            if key in self.normalizer.params_dict:
                nobs[key] = self.normalizer[key].normalize(value)
            else:
                nobs[key] = value
        
        naction = self.normalizer['action'].normalize(batch['action'])
        
        # Normalize target keypose directly using the 'target_keypose' key in normalizer
        ntarget_keypose = self.normalizer['target_keypose'].normalize(batch['target_keypose'])
        
        point_cloud = nobs['point_cloud'] # (B, T, N, 6)
        agent_pos = nobs['agent_pos']     # (B, T, D_pos)
        
        # Flatten time dimension for encoding
        B, T, N, _ = point_cloud.shape
        point_cloud_flat = rearrange(point_cloud, 'b t n c -> (b t) n c')
        agent_pos_flat = rearrange(agent_pos, 'b t d -> (b t) d')
        
        # PointNet
        point_features = self.pointnet(point_cloud_flat) # ((B*T), N, D_point)
        point_coords = point_cloud_flat[..., :3]
        
        # Fusion
        if 'dino_features' in batch['obs']:
            dino_features = batch['obs']['dino_features'] # (B, T, K, N_patches, D_feat)
            dino_features_flat = rearrange(dino_features, 'b t k n d -> (b t) k n d')
            fused_features = self.fusion(point_features, point_coords=point_coords, precomputed_features=dino_features_flat)
        else:
            images = batch['obs']['images']   # (B, T, K, C, H, W)
            images_flat = rearrange(images, 'b t k c h w -> (b t) k c h w')
            fused_features = self.fusion(point_features, point_coords=point_coords, images=images_flat) # ((B*T), N, D_point)
        
        # Global Pooling (Max)
        global_fused = torch.max(fused_features, dim=1)[0] # ((B*T), D_point)
        
        # Agent Pos
        agent_pos_emb = self.agent_pos_mlp(agent_pos_flat) # ((B*T), D_point)
        
        # Combine for Keypose Head
        features = torch.cat([global_fused, agent_pos_emb], dim=-1) # ((B*T), 2*D_point)
        
        # Predict Keypose
        pred_keypose = self.keypose_head(features) # ((B*T), keypose_dim)
        
        # Reshape back to (B, T, ...)
        features = rearrange(features, '(b t) d -> b t d', b=B, t=T)
        pred_keypose = rearrange(pred_keypose, '(b t) d -> b t d', b=B, t=T)
        
        # Construct Global Condition for Diffusion
        # Take only the first n_obs_steps
        n_obs = self.n_obs_steps
        cond_features = features[:, :n_obs, :]
        cond_keypose = pred_keypose[:, :n_obs, :]
        
        global_cond = torch.cat([cond_features, cond_keypose], dim=-1) # (B, n_obs, D_cond)
        global_cond = rearrange(global_cond, 'b t d -> b (t d)')
        
        # Diffusion Loss
        # Sample noise
        noise = torch.randn(naction.shape, device=naction.device)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=naction.device).long()
        
        noisy_action = self.noise_scheduler.add_noise(naction, noise, timesteps)
        
        # Predict noise
        noise_pred = self.diffusion_unet(noisy_action, timesteps, global_cond=global_cond)
        
        loss_diffusion = F.mse_loss(noise_pred, noise)
        
        # Keypose Loss
        loss_keypose = F.mse_loss(pred_keypose, ntarget_keypose)
        
        total_loss = loss_diffusion + loss_keypose
        
        return total_loss, {"loss_diffusion": loss_diffusion, "loss_keypose": loss_keypose}

    def get_action(self, batch):
        # Batch is a dict with 'obs'
        # obs['point_cloud']: (B, T, N, 6)
        # obs['images']: (B, T, K, C, H, W)
        # obs['agent_pos']: (B, T, D_pos)
        
        nobs = {}
        for key, value in batch['obs'].items():
            if key in self.normalizer.params_dict:
                nobs[key] = self.normalizer[key].normalize(value)
            else:
                nobs[key] = value
        
        point_cloud = nobs['point_cloud']
        agent_pos = nobs['agent_pos']
        
        B, T, N, _ = point_cloud.shape
        
        # Flatten time dimension for encoding
        point_cloud_flat = rearrange(point_cloud, 'b t n c -> (b t) n c')
        agent_pos_flat = rearrange(agent_pos, 'b t d -> (b t) d')
        
        # PointNet
        point_features = self.pointnet(point_cloud_flat)
        point_coords = point_cloud_flat[..., :3]
        
        # Fusion
        if 'dino_features' in batch['obs']:
            dino_features = batch['obs']['dino_features']
            dino_features_flat = rearrange(dino_features, 'b t k n d -> (b t) k n d')
            fused_features = self.fusion(point_features, point_coords=point_coords, precomputed_features=dino_features_flat)
        else:
            images = batch['obs']['images']
            images_flat = rearrange(images, 'b t k c h w -> (b t) k c h w')
            fused_features = self.fusion(point_features, point_coords=point_coords, images=images_flat)
        
        # Global Pooling
        global_fused = torch.max(fused_features, dim=1)[0]
        
        # Agent Pos
        agent_pos_emb = self.agent_pos_mlp(agent_pos_flat)
        
        # Combine
        features = torch.cat([global_fused, agent_pos_emb], dim=-1)
        
        # Predict Keypose
        pred_keypose = self.keypose_head(features)
        
        # Reshape back
        features = rearrange(features, '(b t) d -> b t d', b=B, t=T)
        pred_keypose = rearrange(pred_keypose, '(b t) d -> b t d', b=B, t=T)
        
        # Construct Global Condition
        n_obs = self.n_obs_steps
        cond_features = features[:, :n_obs, :]
        cond_keypose = pred_keypose[:, :n_obs, :]
        
        global_cond = torch.cat([cond_features, cond_keypose], dim=-1)
        global_cond = rearrange(global_cond, 'b t d -> b (t d)')
        
        # Diffusion Sampling
        # Initialize noise
        naction = torch.randn((B, self.horizon, self.action_dim), device=global_cond.device)
        
        # DDPM/DDIM Scheduler Sampling
        self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
        
        for t in self.noise_scheduler.timesteps:
            # Predict noise
            model_output = self.diffusion_unet(naction, t, global_cond=global_cond)
            
            # Compute previous noisy sample x_t -> x_t-1
            naction = self.noise_scheduler.step(model_output, t, naction).prev_sample
            
        # Unnormalize action
        action = self.normalizer['action'].unnormalize(naction)
        
        # Return action sequence
        return action

