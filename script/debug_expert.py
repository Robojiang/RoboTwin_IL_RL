import sys
import os
import argparse
import yaml
import numpy as np
import importlib
from pathlib import Path
import time
import cv2

# Add paths
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_directory)
sys.path.append(os.path.join(parent_directory, "description/utils"))

from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance

def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args

def display_observation(obs):
    """Display observation data based on available keys."""
    print("\033[94mObservation:\033[0m")
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: Shape = {value.shape}")
        else:
            print(f"{key}: {value}")

def show_observation_images(env, obs=None):
    """Show observation images using cv2."""
    # RGB
    if env.data_type.get("rgb", False):
        rgb_dict = env.cameras.get_rgb()
        images = []
        for cam_name, cam_data in rgb_dict.items():
            if isinstance(cam_data, dict) and 'rgb' in cam_data:
                img = cam_data['rgb']
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # Add label
                cv2.putText(bgr, cam_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                images.append(bgr)
        
        if images:
            # Tile images
            n = len(images)
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
            h, w, c = images[0].shape
            
            # Create canvas
            canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
            for i, img in enumerate(images):
                r = i // cols
                c = i % cols
                canvas[r*h:(r+1)*h, c*w:(c+1)*w] = img
            
            # Resize if too big
            if canvas.shape[0] > 1000 or canvas.shape[1] > 1800:
                scale = min(1000/canvas.shape[0], 1800/canvas.shape[1])
                canvas = cv2.resize(canvas, (0, 0), fx=scale, fy=scale)
                
            cv2.imshow("RGB Observations", canvas)
            
    # Depth (Optional: normalize for visualization)
    if env.data_type.get("depth", False):
        depth_dict = env.cameras.get_depth()
        for cam_name, cam_data in depth_dict.items():
            if isinstance(cam_data, dict) and 'depth' in cam_data:
                img = cam_data['depth']
                # Normalize depth to 0-255 for visualization
                depth_vis = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                depth_vis = np.uint8(depth_vis)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                cv2.imshow(f"Depth: {cam_name}", depth_vis)
            
    # Point Cloud (Project to 2D for visualization)
    pcd = None
    if obs is not None and 'pointcloud' in obs:
        pcd = obs['pointcloud']
    elif env.data_type.get("pointcloud", False):
        try:
            pcd = env.cameras.get_pcd(if_combine=True)
        except Exception as e:
            print(f"Error getting point cloud: {e}")

    if pcd is not None and len(pcd) > 0:
        try:
            # Get point cloud (N, 6) -> XYZRGB
            # Create a blank image for BEV (Bird's Eye View) - XY plane
            img_size = 500
            bev_img = np.full((img_size, img_size, 3), 50, dtype=np.uint8) # Gray background
            
            # Normalize XY to image coordinates
            scale = img_size / 1.0 # 1 meter = 500 pixels
            offset = img_size / 2
            
            x = pcd[:, 0]
            y = pcd[:, 1]
            z = pcd[:, 2]
            colors = pcd[:, 3:6] # RGB 0-1
            
            # Filter points based on Z (height)
            mask = (z > -0.5) & (z < 2.0)
            x = x[mask]
            y = y[mask]
            colors = colors[mask]
            
            u = (x * scale + offset).astype(int)
            v = (y * scale + offset).astype(int) 
            v = img_size - v # Flip Y
            
            # Clip to image bounds
            valid = (u >= 0) & (u < img_size) & (v >= 0) & (v < img_size)
            u = u[valid]
            v = v[valid]
            colors = (colors[valid] * 255).astype(np.uint8)
            
            # Draw larger points
            for i in range(len(u)):
                cv2.circle(bev_img, (u[i], v[i]), 2, (int(colors[i][2]), int(colors[i][1]), int(colors[i][0])), -1)
            
            cv2.imshow("Point Cloud BEV (XY Plane)", bev_img)
        except Exception as e:
            print(f"Error visualizing point cloud: {e}")
    else:
        # Create a black image with text if no PCD
        img_size = 500
        bev_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        cv2.putText(bev_img, "No Point Cloud Data", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Point Cloud BEV (XY Plane)", bev_img)

    cv2.waitKey(1)

def main():
    parser = argparse.ArgumentParser(description="Debug Expert Script")
    parser.add_argument("--task_name", type=str, default="beat_block_hammer", help="Name of the task")
    parser.add_argument("--task_config", type=str, default="demo_clean", help="Task configuration file")
    parser.add_argument("--seed", type=int, default=2, help="Random seed")
    parser.add_argument("--render_freq", type=int, default=20, help="Render frequency")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--show_obs", type=int, default=1, help="Show observation (1: True, 0: False)")
    
    args_cli = parser.parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_cli.gpu_id)
    
    # Load Task Config
    task_config_path = os.path.join(parent_directory, f"task_config/{args_cli.task_config}.yml")
    with open(task_config_path, "r", encoding="utf-8") as f:
        task_args = yaml.load(f.read(), Loader=yaml.FullLoader)
        
    task_args['task_name'] = args_cli.task_name
    task_args["task_config"] = args_cli.task_config
    task_args["render_freq"] = args_cli.render_freq
    task_args["eval_mode"] = True
    
    # Embodiment Setup
    embodiment_type = task_args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = task_args["camera"]["head_camera_type"]
    task_args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    task_args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        task_args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        task_args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        task_args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        task_args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        task_args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        task_args["embodiment_dis"] = embodiment_type[2]
        task_args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")

    task_args["left_embodiment_config"] = get_embodiment_config(task_args["left_robot_file"])
    task_args["right_embodiment_config"] = get_embodiment_config(task_args["right_robot_file"])
    
    # Initialize Environment
    print(f"Initializing Task: {args_cli.task_name}")
    env = class_decorator(args_cli.task_name)
    
    # Setup Demo
    print(f"Setting up demo with seed {args_cli.seed}")
    env.setup_demo(now_ep_num=0, seed=args_cli.seed, is_test=True, **task_args)
    
    # Monkey Patch for Visualization and Delay
    # Must be done AFTER setup_demo because setup_demo re-initializes the viewer
    if hasattr(env, 'viewer') and env.viewer:
        original_render = env.viewer.render
        def patched_render():
            original_render()
            # Update cameras to get fresh data for visualization
            env.cameras.update_picture() # Ensure cameras capture the current frame
            if args_cli.show_obs:
                show_observation_images(env)
            time.sleep(0.1) # Slow down rendering
        env.viewer.render = patched_render
    
    # Run Expert
    print("Running Expert Policy...")
    try:
        env.play_once()
        if not env.plan_success:
            print("\033[91mExpert stopped early due to planning failure.\033[0m")
        else:
            print("Expert finished.")
    except Exception as e:
        print(f"Expert failed: {e}")
        import traceback
        traceback.print_exc()
        
    env.close_env()
    if args_cli.show_obs:
        print("Press Enter to close visualization windows...")
        input()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
