import os
import h5py
import zarr
import numpy as np
import argparse
import shutil
import cv2
from tqdm import tqdm

def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        left_gripper = root["/joint_action/left_gripper"][()]
        left_arm = root["/joint_action/left_arm"][()]
        right_gripper = root["/joint_action/right_gripper"][()]
        right_arm = root["/joint_action/right_arm"][()]
        vector = root["/joint_action/vector"][()]
        pointcloud = root["/pointcloud"][()]
        
        # Load End Poses
        left_endpose = root["/endpose/left_endpose"][()]
        right_endpose = root["/endpose/right_endpose"][()]
        
        # Load RGB Images
        images = []
        # Sort keys to ensure consistent order: front, head, left, right
        cam_names = sorted([k for k in root["observation"].keys() if "rgb" in root["observation"][k]])
        for cam_name in cam_names:
            raw_data = root[f"observation/{cam_name}/rgb"][()]
            # Decode images
            cam_imgs = []
            for i in range(len(raw_data)):
                img = cv2.imdecode(np.frombuffer(raw_data[i], np.uint8), cv2.IMREAD_COLOR)
                cam_imgs.append(img)
            images.append(np.array(cam_imgs))
        
        # Stack images: (N_cams, T, H, W, C) -> (T, N_cams, H, W, C)
        if images:
            images = np.stack(images, axis=0)
            images = np.transpose(images, (1, 0, 2, 3, 4))
        else:
            images = None

    return left_gripper, left_arm, right_gripper, right_arm, vector, pointcloud, images, left_endpose, right_endpose

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
        # 2. If Stopped: Only Keyframe if moved far enough OR simply too much time passed (optional, but better to rely on dist)
        
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

def main():
    parser = argparse.ArgumentParser(description="Process episodes with PPI keyframe extraction.")
    parser.add_argument("task_name", type=str, help="The name of the task")
    parser.add_argument("task_config", type=str, help="Task configuration")
    parser.add_argument("expert_data_num", type=int, help="Number of episodes")
    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    num = args.expert_data_num

    # Paths
    
    load_dir = os.path.abspath(os.path.join(os.getcwd(), "../../data", task_name, task_config))
    save_dir = os.path.join(os.getcwd(), "data", f"{task_name}-{task_config}-{num}-ppi.zarr")

    print(f"Loading from: {load_dir}")
    print(f"Saving to: {save_dir}")

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    # Storage lists - REMOVED to save memory
    # We will write directly to Zarr
    
    total_count = 0
    
    # Zarr datasets (initialized on first iteration)
    zarr_datasets = {}
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    for current_ep in tqdm(range(num), desc="Processing Episodes"):
        load_path = os.path.join(load_dir, f"data/episode{current_ep}.hdf5")
        
        try:
            (
                left_gripper, left_arm, 
                right_gripper, right_arm, 
                vector, pointcloud, images,
                left_endpose, right_endpose
            ) = load_hdf5(load_path)
        except Exception as e:
            print(f"Skipping episode {current_ep}: {e}")
            continue

        # Calculate Keyframe Mask
        full_mask = get_keyframe_mask(left_gripper, right_gripper, left_arm, right_arm, left_endpose, right_endpose)
        
        # Check lengths
        T = vector.shape[0]
        if T < 2: continue
        
        # Align data
        ep_state = vector[:-1]
        ep_action = vector[1:]
        ep_pointcloud = pointcloud[:-1]
        ep_mask = full_mask[:-1]
        ep_left_endpose = left_endpose[:-1]
        ep_right_endpose = right_endpose[:-1]
        
        ep_images = None
        if images is not None:
            ep_images = images[:-1]
        
        # Prepare data dict for easier iteration
        current_data = {
            "state": ep_state,
            "action": ep_action,
            "point_cloud": ep_pointcloud,
            "keyframe_mask": ep_mask,
            "left_endpose": ep_left_endpose,
            "right_endpose": ep_right_endpose
        }
        if ep_images is not None:
            current_data["images"] = ep_images

        # Initialize Zarr Datasets on the first valid episode
        if not zarr_datasets:
            print("Initializing Zarr datasets...")
            # Define chunks
            chunks = {
                "state": (100, ep_state.shape[1]),
                "action": (100, ep_action.shape[1]),
                "point_cloud": (100, ep_pointcloud.shape[1], ep_pointcloud.shape[2]),
                "keyframe_mask": (100,),
                "episode_ends": (100,),
                "left_endpose": (100, ep_left_endpose.shape[1]),
                "right_endpose": (100, ep_right_endpose.shape[1])
            }
            if ep_images is not None:
                chunks["images"] = (100, ep_images.shape[1], ep_images.shape[2], ep_images.shape[3], ep_images.shape[4])

            for key, val in current_data.items():
                # Create dataset with shape=(0, ...) and maxshape=(None, ...) to allow appending
                shape = (0,) + val.shape[1:]
                maxshape = (None,) + val.shape[1:]
                dtype = val.dtype
                
                zarr_datasets[key] = zarr_data.create_dataset(
                    key, shape=shape, maxshape=maxshape, chunks=chunks.get(key), 
                    dtype=dtype, compressor=compressor, overwrite=True
                )
            
            # Initialize episode_ends separately in meta group
            zarr_datasets["episode_ends"] = zarr_meta.create_dataset(
                "episode_ends", shape=(0,), maxshape=(None,), chunks=(100,), 
                dtype="int64", compressor=compressor, overwrite=True
            )

        # Append data to Zarr
        for key, val in current_data.items():
            zarr_datasets[key].append(val)
            
        # Update and append episode ends
        total_count += len(ep_state)
        zarr_datasets["episode_ends"].append([total_count])

    print("Done.")
    
    print("Done.")

if __name__ == "__main__":
    main()