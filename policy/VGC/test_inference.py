
import torch
import numpy as np
import os
import hydra
from omegaconf import OmegaConf
from deploy_policy import get_model, VGCWrapper, encode_obs

def test_inference():
    print("Testing VGC Policy Inference...")
    
    # Mock user args
    usr_args = {
        'task_name': 'beat_block_hammer',
        'checkpoint_path': 'data/outputs/2026.01.04/15.15.07_vgc_policy_beat_block_hammer/checkpoints/debug.ckpt' # Update this if needed
    }
    
    # Find the latest checkpoint automatically if specific one not found
    if not os.path.exists(usr_args['checkpoint_path']):
        # Search for latest
        base_dir = 'data/outputs'
        # ... simple search logic or just warn
        print(f"Warning: Checkpoint {usr_args['checkpoint_path']} not found. Please update path.")
        return

    # 1. Load Model
    print("Loading model...")
    wrapper = get_model(usr_args)
    
    # 2. Create Wrapper
    # wrapper = VGCWrapper(policy) # get_model already returns the wrapper

    
    # 3. Create Dummy Observation
    print("Creating dummy observation...")
    # Based on RoboTwin structure
    obs = {
        'joint_action': {'vector': np.random.randn(14)},
        'pointcloud': np.random.randn(1024, 6).astype(np.float32),
        'observation': {
            'cam_high': {'rgb': np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)},
            'cam_left': {'rgb': np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)},
            'cam_right': {'rgb': np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)},
            'cam_wrist': {'rgb': np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)},
        },
        'endpose': {
            'left_endpose': np.random.randn(7),
            'right_endpose': np.random.randn(7)
        }
    }
    
    # 4. Run Inference
    print("Running inference...")
    obs_encoded = encode_obs(obs)
    wrapper.update_obs(obs_encoded)
    action = wrapper.get_action()
    
    print(f"Inference successful!")
    print(f"Action shape: {action.shape}")
    print(f"Action values (first 5): {action[:5]}")

if __name__ == "__main__":
    test_inference()
