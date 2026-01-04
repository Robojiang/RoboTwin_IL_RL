import sys
import os
import torch
import numpy as np
import copy
import zarr
import cv2
from typing import Dict

# Add DP3 path to sys.path to reuse its utilities
# We need to be careful with relative paths now that we moved the file
# Current file: policy/VGC/vgc/dataset/vgc_dataset.py
# DP3 path: policy/DP3/3D-Diffusion-Policy

current_file_path = os.path.abspath(__file__)
vgc_dataset_dir = os.path.dirname(current_file_path) # policy/VGC/vgc/dataset
vgc_pkg_dir = os.path.dirname(vgc_dataset_dir) # policy/VGC/vgc
vgc_root_dir = os.path.dirname(vgc_pkg_dir) # policy/VGC
policy_dir = os.path.dirname(vgc_root_dir) # policy

dp3_pkg_path = os.path.join(policy_dir, 'DP3', '3D-Diffusion-Policy')
sys.path.append(dp3_pkg_path)

from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset

def quat_to_rot6d(quat):
    # quat: (..., 4) [x, y, z, w]
    # Returns: (..., 6)
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    xw, yw, zw = x*w, y*w, z*w
    
    r11 = 1 - 2 * (yy + zz)
    r21 = 2 * (xy + zw)
    r31 = 2 * (xz - yw)
    
    r12 = 2 * (xy - zw)
    r22 = 1 - 2 * (xx + zz)
    r32 = 2 * (yz + xw)
    
    rot6d = np.stack([r11, r21, r31, r12, r22, r32], axis=-1)
    return rot6d

class VGCDataset(BaseDataset):
    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        task_name=None,
    ):
        super().__init__()
        self.task_name = task_name
        
        # Resolve zarr path
        # If relative, assume it's relative to policy/VGC/
        if not os.path.isabs(zarr_path):
            zarr_path = os.path.join(vgc_root_dir, zarr_path)
            
        print(f"Loading dataset from: {zarr_path}")
        
        # Load ReplayBuffer with all necessary keys
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, 
            keys=["state", "action", "point_cloud", "images", "left_endpose", "right_endpose"]
        )
        
        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)
        
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        # Construct data for normalization matching __getitem__ structure
        state = self.replay_buffer["state"]
        left_endpose = self.replay_buffer["left_endpose"]
        right_endpose = self.replay_buffer["right_endpose"]
        
        # Convert to 9D (3 pos + 6 rot)
        left_pos = left_endpose[..., :3]
        left_quat = left_endpose[..., 3:]
        left_rot6d = quat_to_rot6d(left_quat)
        left_endpose_9d = np.concatenate([left_pos, left_rot6d], axis=-1)
        
        right_pos = right_endpose[..., :3]
        right_quat = right_endpose[..., 3:]
        right_rot6d = quat_to_rot6d(right_quat)
        right_endpose_9d = np.concatenate([right_pos, right_rot6d], axis=-1)
        
        # agent_pos = state (14) + left (9) + right (9) = 32
        agent_pos = np.concatenate([state, left_endpose_9d, right_endpose_9d], axis=-1)
        
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": agent_pos,
            "point_cloud": self.replay_buffer["point_cloud"],
            "left_endpose": left_endpose_9d, # Used for keypose normalization in policy
            "right_endpose": right_endpose_9d,
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # sample contains arrays of shape (Horizon, ...)
        
        agent_pos = sample["state"].astype(np.float32)
        point_cloud = sample["point_cloud"].astype(np.float32)
        images = sample["images"] # (T, N_cams, H, W, C)
        
        # Resize images to 224x224 and normalize
        T, N_cams, H, W, C = images.shape
        resized_images = np.zeros((T, N_cams, 224, 224, C), dtype=np.float32)
        
        for t in range(T):
            for n in range(N_cams):
                resized_images[t, n] = cv2.resize(images[t, n], (224, 224)).astype(np.float32) / 255.0
        
        images = resized_images
            
        # Rearrange images to (T, N_cams, C, H, W) for PyTorch
        # Input is (T, N_cams, H, W, C) -> Output (T, N_cams, C, H, W)
        images = np.moveaxis(images, -1, 2) 
        
        left_endpose = sample["left_endpose"].astype(np.float32)
        right_endpose = sample["right_endpose"].astype(np.float32)
        
        # Convert to 9D (3 pos + 6 rot)
        left_pos = left_endpose[..., :3]
        left_quat = left_endpose[..., 3:]
        left_rot6d = quat_to_rot6d(left_quat)
        left_endpose_9d = np.concatenate([left_pos, left_rot6d], axis=-1)
        
        right_pos = right_endpose[..., :3]
        right_quat = right_endpose[..., 3:]
        right_rot6d = quat_to_rot6d(right_quat)
        right_endpose_9d = np.concatenate([right_pos, right_rot6d], axis=-1)
        
        # Update agent_pos to include end-effector poses
        # agent_pos = sample["state"] (14) + left (9) + right (9) = 32
        agent_pos = np.concatenate([agent_pos, left_endpose_9d, right_endpose_9d], axis=-1)
        
        # Combine endposes into a single keypose target (18 dims)
        keypose = np.concatenate([left_endpose_9d, right_endpose_9d], axis=-1)

        data = {
            "obs": {
                "point_cloud": point_cloud,  # T, N, 6
                "agent_pos": agent_pos,      # T, D_pos
                "images": images,            # T, K, C, H, W
            },
            "target_keypose": keypose,       # T, D_keypose
            "action": sample["action"].astype(np.float32),  # T, D_action
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
