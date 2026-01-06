import zarr
import numpy as np
import argparse
from tqdm import tqdm
import os

def get_keyframe_mask(left_gripper, right_gripper, left_arm, right_arm, left_endpose, right_endpose, stopping_delta=0.01, gripper_delta=0.05, pos_threshold=0.015):
    """
    Generate keyframe mask based on gripper changes and robot stops.
    Filters out stops that are spatial duplicates of the previous keyframe.
    """
    length = left_gripper.shape[0]
    mask = np.zeros(length, dtype=bool)
    
    # 1. Calculate velocities
    left_vel = np.linalg.norm(left_arm[1:] - left_arm[:-1], axis=1)
    right_vel = np.linalg.norm(right_arm[1:] - right_arm[:-1], axis=1)
    
    left_vel = np.insert(left_vel, 0, 0)
    right_vel = np.insert(right_vel, 0, 0)
    
    # 2. Gripper Changes
    if left_gripper.ndim > 1: left_gripper = left_gripper.flatten()
    if right_gripper.ndim > 1: right_gripper = right_gripper.flatten()
    
    left_gripper_diff = np.abs(left_gripper[1:] - left_gripper[:-1])
    right_gripper_diff = np.abs(right_gripper[1:] - right_gripper[:-1])
    
    left_gripper_diff = np.insert(left_gripper_diff, 0, 0)
    right_gripper_diff = np.insert(right_gripper_diff, 0, 0)
    
    last_keyframe_idx = 0
    mask[0] = True # Always include start
    
    # Extract positions for distance check
    # endpose shape (T, 7), take first 3 for XYZ
    l_pos = left_endpose[:, :3]
    r_pos = right_endpose[:, :3]
    
    for i in range(1, length):
        is_gripper_change = False
        
        # Condition A: Gripper Changed significantly
        if left_gripper_diff[i] > gripper_delta or right_gripper_diff[i] > gripper_delta:
            is_gripper_change = True
            
        # Condition B: Robot Stopped
        is_stopped = (left_vel[i] < stopping_delta) and (right_vel[i] < stopping_delta)
        
        # Check spatial distance from last keyframe
        curr_l = l_pos[i]
        curr_r = r_pos[i]
        last_l = l_pos[last_keyframe_idx]
        last_r = r_pos[last_keyframe_idx]
        
        dist_l = np.linalg.norm(curr_l - last_l)
        dist_r = np.linalg.norm(curr_r - last_r)
        
        # Logic: 
        # 1. If Grip Change: Always Keyframe (state change is critical)
        # 2. If Stopped: Only Keyframe if moved far enough OR simply too much time passed
        
        is_far_enough = (dist_l > pos_threshold) or (dist_r > pos_threshold)
        
        # Enforce min time interval to filter jitter
        if (i - last_keyframe_idx) > 5: 
            if is_gripper_change:
                # Gripper change is always a keyframe
                mask[i] = True
                last_keyframe_idx = i
            elif is_stopped and is_far_enough:
                 # Only record stop if we actually moved somewhere new
                mask[i] = True
                last_keyframe_idx = i
                
    mask[-1] = True # Always include end
    return mask

def fix_keyframes(zarr_path, stopping_delta=0.01, gripper_delta=0.05, pos_threshold=0.015):
    print(f"Opening Zarr dataset: {zarr_path}")
    
    root = zarr.open(zarr_path, mode='r+')
    data_group = root['data']
    meta_group = root['meta']
    
    episode_ends = meta_group['episode_ends'][:]
    
    # Load required data fields
    # state: usually [joint_angles(14)] or [joint_angles(14) + other]
    # In process_data, state was vector[:-1]. vector comes from load_hdf5 -> joint_action/vector
    # joint_action/vector is [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)] = 14 dims
    
    state = data_group['state']
    left_endpose = data_group['left_endpose']
    right_endpose = data_group['right_endpose']
    keyframe_mask_dataset = data_group['keyframe_mask']
    
    total_keyframes_old = np.sum(keyframe_mask_dataset[:])
    total_frames = len(keyframe_mask_dataset)
    
    print(f"Total frames: {total_frames}")
    print(f"Old total keyframes: {total_keyframes_old}")
    
    new_mask_all = []
    
    start_idx = 0
    for i in tqdm(range(len(episode_ends)), desc="Fixing Keyframes"):
        end_idx = episode_ends[i]
        
        # Extract episode data
        ep_state = state[start_idx:end_idx]
        ep_l_end = left_endpose[start_idx:end_idx]
        ep_r_end = right_endpose[start_idx:end_idx]
        
        # Parse state to get arm and gripper
        # Assuming state structure: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
        # This matches RoboTwin joint_action/vector structure
        
        left_arm = ep_state[:, 0:6]
        left_gripper = ep_state[:, 6]
        right_arm = ep_state[:, 7:13]
        right_gripper = ep_state[:, 13]
        
        # Compute new mask
        new_mask = get_keyframe_mask(
            left_gripper, right_gripper, left_arm, right_arm, 
            ep_l_end, ep_r_end,
            stopping_delta=stopping_delta, 
            gripper_delta=gripper_delta, 
            pos_threshold=pos_threshold
        )
        
        new_mask_all.append(new_mask)
        
        start_idx = end_idx
        
    # Concatenate all masks
    new_mask_all = np.concatenate(new_mask_all)
    
    # Verify length matches
    assert len(new_mask_all) == total_frames, f"New mask length {len(new_mask_all)} != Total frames {total_frames}"
    
    # Write back to Zarr
    print("Writing new mask to Zarr...")
    keyframe_mask_dataset[:] = new_mask_all
    
    total_keyframes_new = np.sum(new_mask_all)
    print(f"New total keyframes: {total_keyframes_new}")
    print(f"Reduction: {total_keyframes_old - total_keyframes_new} frames removed.")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr_path", type=str, required=True, help="Path to Zarr dataset")
    parser.add_argument("--pos_threshold", type=float, default=0.015, help="Minimum spatial distance (m) between keyframes")
    
    args = parser.parse_args()
    
    fix_keyframes(args.zarr_path, pos_threshold=args.pos_threshold)
