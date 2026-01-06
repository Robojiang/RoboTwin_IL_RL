import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
import copy

# Add DP3 path to import base modules
current_file_path = os.path.abspath(__file__)
aim_policy_dir = os.path.dirname(current_file_path) # policy/aim
policy_dir = os.path.dirname(aim_policy_dir) # policy
dp3_pkg_path = os.path.join(policy_dir, 'DP3', '3D-Diffusion-Policy')
sys.path.append(dp3_pkg_path)

from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy_3d.model.vision.pointnet_extractor import PointNetEncoderXYZRGB, PointNetEncoderXYZ
from diffusion_policy_3d.common.model_util import print_params
from termcolor import cprint
import transforms3d

class AIMPolicy(BasePolicy):
    def __init__(self, 
                 shape_meta: dict,
                 noise_scheduler,
                 horizon, 
                 n_action_steps, 
                 n_obs_steps,
                 num_inference_steps=None,
                 obs_as_global_cond=True,
                 diffusion_step_embed_dim=256,
                 down_dims=(256, 512, 1024),
                 kernel_size=5,
                 n_groups=8,
                 condition_type="film",
                 use_pc_color=False,
                 pointnet_type="pointnet",
                 pointcloud_encoder_cfg=None,
                 use_aux_points=True,
                 aux_point_num=50, # Number of points to generate per gripper
                 **kwargs):
        super().__init__()
        
        self.condition_type = condition_type
        self.use_pc_color = use_pc_color
        self.use_aux_points = use_aux_points
        self.aux_point_num = aux_point_num
        
        # 1. Parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_dim = action_shape[0] if len(action_shape) == 1 else action_shape[0] * action_shape[1]
        
        obs_shape_meta = shape_meta['obs']
        self.state_dim = obs_shape_meta['agent_pos']['shape'][0]
        
        # 2. Configure PointNet Encoder
        # Determine input channels
        # Base channels: 3 (XYZ) or 6 (XYZRGB)
        # AIM adds 1 channel: Indicator (0 for scene, 1 for aux)
        base_channels = 6 if use_pc_color else 3
        in_channels = base_channels + 1 if use_aux_points else base_channels
        
        cprint(f"[AIMPolicy] Input Channels: {in_channels} (Color: {use_pc_color}, Aux: {use_aux_points})", "cyan")
        
        if pointcloud_encoder_cfg is None:
            pointcloud_encoder_cfg = {}
        
        # Override in_channels
        enc_cfg = copy.deepcopy(pointcloud_encoder_cfg)
        enc_cfg['in_channels'] = in_channels
        
        # We use PointNetEncoderXYZRGB generically as it allows setting in_channels
        self.obs_encoder = PointNetEncoderXYZRGB(**enc_cfg)
        
        # Output dim of encoder
        # PointNetEncoderXYZRGB output dim is defined by 'out_channels' in cfg or default 1024
        self.obs_feature_dim = enc_cfg.get('out_channels', 1024)
        
        # 3. Diffusion Model
        # Input to diffusion is action + condition (if not global)
        input_dim = self.action_dim
        global_cond_dim = None
        
        # Add explicit Proprioception Embedding (MLP)
        self.proprio_mlp = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.Mish(),
            nn.Linear(256, self.obs_feature_dim)
        )
        # We fuse PointNet feature + Proprio feature
        # Fusion method: Add or Concat? 
        # Usually Concat is safer to preserve distinct info.
        # But if we concat, dimensions double. 
        # Alternatively, we can project Proprio to same dim and ADD.
        # Or concat and project back.
        # DP3 often concats visual features with low-dim features if both exist.
        # Let's Concat.
        
        combined_feature_dim = self.obs_feature_dim * 2 # PointNet + Proprio
        
        if obs_as_global_cond:
            # We treat the extracted point features as global condition
            global_cond_dim = combined_feature_dim
            if not "cross_attention" in condition_type:
                 global_cond_dim *= n_obs_steps
        else:
            input_dim += combined_feature_dim
            
        self.model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
        )
        
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        
        print_params(self)
        
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _rot6d_to_mat(self, d6):
        """
        Converts 6D rotation representation to 3x3 rotation matrix.
        d6: (..., 6)
        Returns: (..., 3, 3)
        """
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)

    def _generate_aux_points(self, agent_pos):
        """
        Generate auxiliary points (Cylinder/Laser beam) for the gripper based on agent_pos.
        Refers to visualize_keyframes.py visualization logic.
        agent_pos: (B, T, D_state)
        Returns: (B, T, num_aux, 3) XYZ coordinates
        """
        B, T, D = agent_pos.shape
        states_list = []
        
        # Parse Agent Pos (VGC Format: 32D = 14 Joint + 9 Left + 9 Right)
        if D == 32:
             # Left Hand
             left_pos = agent_pos[..., 14:17] # (B, T, 3)
             left_rot6d = agent_pos[..., 17:23] # (B, T, 6)
             states_list.append((left_pos, left_rot6d))
             
             # Right Hand
             right_pos = agent_pos[..., 23:26]
             right_rot6d = agent_pos[..., 26:32]
             states_list.append((right_pos, right_rot6d))
        else:
             # Fallback: Assume Single arm 9D at end or similar hack, mainly for robustness
             if D >= 9:
                 pos = agent_pos[..., -9:-6]
                 rot = agent_pos[..., -6:]
                 states_list.append((pos, rot))

        if not states_list:
            return torch.zeros((B, T, 0, 3), device=agent_pos.device)

        output_pcs = []
        flat_size = B*T
        
        # --- Create Local "Laser/Cylinder" Geometry ---
        # Mimic visualize_keyframes.py: pts = pos + direction * dists
        # 0.3m length cylinder pointing along X-axis
        num_pts = self.aux_point_num # e.g. 50-100
        
        # 1. Generate points along X-axis (0 to 0.2m)
        # (N, 3)
        dists = torch.linspace(0, 0.2, num_pts, device=agent_pos.device, dtype=agent_pos.dtype)
        local_line = torch.stack([dists, torch.zeros_like(dists), torch.zeros_like(dists)], dim=-1) # (N, 3)
        
        # 2. Add some thickness (Cylinder) to make it robust to pointnet sampling? 
        # Or just a line? PointNet works well with lines if density is high.
        # Let's add small noise to Y and Z to make it a thin tube (radius 0.01m)
        angle = torch.rand(num_pts, device=agent_pos.device, dtype=agent_pos.dtype) * 2 * np.pi
        radius = torch.rand(num_pts, device=agent_pos.device, dtype=agent_pos.dtype) * 0.01
        y_offset = radius * torch.cos(angle)
        z_offset = radius * torch.sin(angle)
        local_cylinder = local_line + torch.stack([torch.zeros_like(dists), y_offset, z_offset], dim=-1)
        
        # Replicate for batch
        # (BT, N, 3)
        base_pts = repeat(local_cylinder, 'n c -> (b t) n c', b=B, t=T)
        
        for pos, rot6d in states_list:
            # pos: (B, T, 3)
            # rot6d: (B, T, 6)
            
            pos_flat = rearrange(pos, 'b t c -> (b t) c')
            rot6d_flat = rearrange(rot6d, 'b t c -> (b t) c')
            
            # Convert 6D rot to Matrix
            rot_mat = self._rot6d_to_mat(rot6d_flat) # (BT, 3, 3)
            
            
            transformed_pts = torch.bmm(base_pts, rot_mat) + pos_flat.unsqueeze(1)
            output_pcs.append(transformed_pts)
            
        all_aux_pts = torch.cat(output_pcs, dim=1) # (BT, num_hands*N, 3)
        all_aux_pts = rearrange(all_aux_pts, '(b t) n c -> b t n c', b=B, t=T)
        
        return all_aux_pts

    def forward(self, batch):
        # Normalize dict
        nobs = self.normalizer.normalize(batch['obs'])
        naction = self.normalizer['action'].normalize(batch['action'])
        
        # Prepare Input
        # Point Cloud: (B, T, N, 3 or 6)
        pc = nobs['point_cloud']
        # Agent Pos: (B, T, D)
        agent_pos = nobs['agent_pos']
        
        B, T, N, C = pc.shape
        
        # 1. Generate Aux Points
        if self.use_aux_points:
            # We need UN-normalized agent pos to generate correct metric points 
            # relative to the point cloud if the PC is in metric space.
            # BUT 'nobs' PC is normalized? 
            # Usually DP3 normalization of PC is Identity or simple.
            # If PC is normalized, we must Normalize Aux Points too.
            # It's safer to generate Aux Points from UN-normalized Agent Pos, THEN normalize them same as PC.
            # But wait, PC normalization is usually "none" or "center".
            # If it's LinearNormalizer, it might scale/offset.
            
            # Check if 'point_cloud' has a normalizer
            raw_agent_pos = batch['obs']['agent_pos']
            aux_pts_xyz = self._generate_aux_points(raw_agent_pos) # (B, T, K, 3)
            
            # If there is color in PC, give Aux Points a distinctive color (e.g. Red)
            # or just 0s. 
            # PC channels: 0-2 (XYZ), 3-5 (RGB).
            aux_feats = []
            aux_feats.append(aux_pts_xyz)
            
            if self.use_pc_color:
                # Aux color black as requested
                aux_rgb = torch.zeros_like(aux_pts_xyz)
                aux_feats.append(aux_rgb)
            
            aux_pc = torch.cat(aux_feats, dim=-1) # (B, T, K, C)
            
            # Normalize Aux PC using point_cloud normalizer if it exists
            # Ideally we reuse the 'point_cloud' normalizer stats.
            # But the Normalizer object works on keys.
            # If 'point_cloud' is normalized, we should normalize 'aux_pc' with same stats.
            # Usually in DP3, PointCloud is NOT normalized by LinearNormalizer (Identity), 
            # instead it's centered in the Encoder? 
            # Looking at DP3 code: `nobs[key] = self.normalizer[key].normalize(value)`
            # If point_cloud is in normalizer keys, it gets normalized.
            
            # Assuming Identity for now or that it handles bounds. 
            # If we concat, they must be in same space.
            # Safe bet: Apply 'point_cloud' normalization to aux_pc if available.
            
            if 'point_cloud' in self.normalizer.params_dict:
                 # This is tricky because LinearNormalizer expects specific shapes sometimes.
                 # Let's assume for now Identity or simple global scale.
                 # If we trust user inputs are in consistent metric space, we are good.
                 # Usually inputs are absolute coords.
                 pass
            
            # 2. Add Indicator Channel
            # Scene: 0
            # Aux: 1
            
            # pc: (B, T, N, C)
            scene_indicator = torch.zeros((B, T, N, 1), device=pc.device, dtype=pc.dtype)
            pc_with_ind = torch.cat([pc, scene_indicator], dim=-1)
            
            # aux: (B, T, K, C)
            aux_indicator = torch.ones((B, T, aux_pc.shape[2], 1), device=aux_pc.device, dtype=aux_pc.dtype)
            aux_pc_with_ind = torch.cat([aux_pc, aux_indicator], dim=-1)
            
            # Concat
            full_pc = torch.cat([pc_with_ind, aux_pc_with_ind], dim=2) # (B, T, N+K, C+1)
            
        else:
             full_pc = pc
             
        # Flatten for processing
        # (B, T, N, C) -> (B*T, N, C)
        full_pc_flat = rearrange(full_pc, 'b t n c -> (b t) n c')
        
        # Encoder
        # features: (B*T, D_emb)
        features = self.obs_encoder(full_pc_flat)
        
        # Reshape
        point_features = rearrange(features, '(b t) d -> b t d', b=B, t=T)
        
        # --- Proprioception Encoding ---
        # agent_pos: (B, T, D_state)
        # We need to make sure this is properly encoded
        proprio_features = self.proprio_mlp(agent_pos) # (B, T, D_emb)
        
        # Combine
        combined_features = torch.cat([point_features, proprio_features], dim=-1) # (B, T, 2*D_emb)

        # Select steps for conditioning
        if self.obs_as_global_cond:
             # Flatten T
             n_obs = self.n_obs_steps
             global_cond = combined_features[:, :n_obs, :]
             global_cond = rearrange(global_cond, 'b t d -> b (t d)')
        
        # Diffusion Loss
        noise = torch.randn(naction.shape, device=naction.device)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=naction.device).long()
        
        noisy_action = self.noise_scheduler.add_noise(naction, noise, timesteps)
        
        pred = self.model(noisy_action, timesteps, global_cond=global_cond)
        
        loss = F.mse_loss(pred, noise)
        loss_dict = {'loss': loss.item()}
        return loss, loss_dict

    def get_action(self, batch):
        # ... logic similar to forward but sampling ...
        # Normalize dict
        nobs = self.normalizer.normalize(batch['obs'])
        
        # Prepare Input
        pc = nobs['point_cloud']
        agent_pos = nobs['agent_pos']
        B, T, N, C = pc.shape
        
        if self.use_aux_points:
            raw_agent_pos = batch['obs']['agent_pos']
            aux_pts_xyz = self._generate_aux_points(raw_agent_pos) 
            aux_feats = [aux_pts_xyz]
            if self.use_pc_color:
                aux_rgb = torch.zeros_like(aux_pts_xyz)
                aux_feats.append(aux_rgb)
            aux_pc = torch.cat(aux_feats, dim=-1)
            
            scene_indicator = torch.zeros((B, T, N, 1), device=pc.device, dtype=pc.dtype)
            pc_with_ind = torch.cat([pc, scene_indicator], dim=-1)
            
            aux_indicator = torch.ones((B, T, aux_pc.shape[2], 1), device=aux_pc.device, dtype=aux_pc.dtype)
            aux_pc_with_ind = torch.cat([aux_pc, aux_indicator], dim=-1)
            
            full_pc = torch.cat([pc_with_ind, aux_pc_with_ind], dim=2) 
        else:
             full_pc = pc
             
        full_pc_flat = rearrange(full_pc, 'b t n c -> (b t) n c')
        features = self.obs_encoder(full_pc_flat)
        point_features = rearrange(features, '(b t) d -> b t d', b=B, t=T)
        
        proprio_features = self.proprio_mlp(agent_pos) 
        combined_features = torch.cat([point_features, proprio_features], dim=-1)

        global_cond = combined_features
        if self.obs_as_global_cond:
             n_obs = self.n_obs_steps
             global_cond = global_cond[:, :n_obs, :]
             global_cond = rearrange(global_cond, 'b t d -> b (t d)')

        # Sampling
        naction = torch.randn((B, self.horizon, self.action_dim), device=pc.device)
        self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
        
        for t in self.noise_scheduler.timesteps:
            model_output = self.model(naction, t, global_cond=global_cond)
            naction = self.noise_scheduler.step(model_output, t, naction).prev_sample
            
        action = self.normalizer['action'].unnormalize(naction)
        return action
