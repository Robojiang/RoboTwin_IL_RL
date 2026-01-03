
import sys
import os
import argparse
import yaml
import numpy as np
import importlib
from pathlib import Path
import time
import cv2
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Open3D not found. 3D visualization will be disabled.")

# Add paths
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_directory)
sys.path.append(os.path.join(parent_directory, "policy"))
sys.path.append(os.path.join(parent_directory, "description/utils"))

from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

# Global Visualizer
vis_o3d = None
pcd_o3d = None
first_frame = True

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

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "task_config/_camera_config.yml")
    assert os.path.isfile(camera_config_path), "task config file is missing"
    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]

def render_view(x, y, colors, img_size, label):
    img = np.full((img_size, img_size, 3), 50, dtype=np.uint8)
    scale = img_size / 1.5 # Zoom out a bit to fit more
    offset = img_size / 2
    
    u = (x * scale + offset).astype(int)
    v = (y * scale + offset).astype(int)
    v = img_size - v # Flip Y for image coords
    
    valid = (u >= 0) & (u < img_size) & (v >= 0) & (v < img_size)
    u = u[valid]
    v = v[valid]
    c = (colors[valid] * 255).astype(np.uint8)
    
    # Draw points
    for j in range(len(u)):
        # BGR color for OpenCV
        cv2.circle(img, (u[j], v[j]), 2, (int(c[j][2]), int(c[j][1]), int(c[j][0])), -1)
        
    cv2.putText(img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img

def show_observation_images(env, obs=None):
    """Show observation images using cv2."""
    global vis_o3d, pcd_o3d, first_frame
    
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
            x = pcd[:, 0]
            y = pcd[:, 1]
            z = pcd[:, 2]
            colors = pcd[:, 3:6] # RGB 0-1
            
            # Filter points based on Z (height)
            mask = (z > -0.5) & (z < 2.0)
            x = x[mask]
            y = y[mask]
            z = z[mask]
            colors = colors[mask]
            
            # 3D Visualization with Open3D
            if HAS_OPEN3D:
                if vis_o3d is None:
                    vis_o3d = o3d.visualization.Visualizer()
                    vis_o3d.create_window(window_name="3D Point Cloud", width=800, height=600)
                    pcd_o3d = o3d.geometry.PointCloud()
                    # Add dummy
                    pcd_o3d.points = o3d.utility.Vector3dVector(np.array([[0,0,0]]))
                    vis_o3d.add_geometry(pcd_o3d)
                    
                    ctr = vis_o3d.get_view_control()
                    ctr.set_front([0, -1, -1])
                    ctr.set_lookat([0, 0, 0])
                    ctr.set_up([0, 0, 1])
                    ctr.set_zoom(0.8)
                
                # Update Geometry
                points_o3d = pcd[mask, :3]
                colors_o3d = pcd[mask, 3:6]
                
                vis_o3d.clear_geometries()
                pcd_o3d = o3d.geometry.PointCloud()
                pcd_o3d.points = o3d.utility.Vector3dVector(points_o3d)
                pcd_o3d.colors = o3d.utility.Vector3dVector(colors_o3d)
                
                vis_o3d.add_geometry(pcd_o3d, reset_bounding_box=first_frame)
                
                opt = vis_o3d.get_render_option()
                opt.point_size = 3.0
                opt.background_color = np.asarray([0.1, 0.1, 0.1])
                
                vis_o3d.poll_events()
                vis_o3d.update_renderer()
                
                if first_frame:
                    first_frame = False

            # 2D 3-View Visualization
            img_size = 400
            
            # 1. Top View (XY)
            img_xy = render_view(x, y, colors, img_size, "Top View (XY)")
            
            # 2. Front View (XZ) - Z is up
            img_xz = render_view(x, z - 0.5, colors, img_size, "Front View (XZ)") 
            
            # 3. Side View (YZ)
            img_yz = render_view(y, z - 0.5, colors, img_size, "Side View (YZ)")
            
            # Combine
            combined_pc = np.hstack([img_xy, img_xz, img_yz])
            
            cv2.imshow("3-View Point Cloud", combined_pc)
        except Exception as e:
            print(f"Error visualizing point cloud: {e}")
    else:
        # Create a black image with text if no PCD
        img_size = 400
        bev_img = np.zeros((img_size, img_size*3, 3), dtype=np.uint8)
        cv2.putText(bev_img, "No Point Cloud Data", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("3-View Point Cloud", bev_img)

    cv2.waitKey(1)

def main():
    parser = argparse.ArgumentParser(description="Debug Episode Script")
    parser.add_argument("--task_name", type=str, default="beat_block_hammer", help="Name of the task")
    parser.add_argument("--task_config", type=str, default="demo_clean", help="Task configuration file")
    parser.add_argument("--policy_name", type=str, default="DP3", help="Policy name")
    parser.add_argument("--ckpt_setting", type=str, default="demo_3d_vision_easy", help="Checkpoint setting")
    parser.add_argument("--expert_data_num", type=int, default=100, help="Number of expert data")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--render_freq", type=int, default=20, help="Render frequency (set to >0 to visualize)")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--show_obs", type=int, default=1, help="Show observation (1: True, 0: False)")
    
    args_cli = parser.parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_cli.gpu_id)
    
    # Load Policy Config
    policy_config_path = os.path.join(parent_directory, f"policy/{args_cli.policy_name}/deploy_policy.yml")
    if os.path.exists(policy_config_path):
        with open(policy_config_path, "r", encoding="utf-8") as f:
            usr_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    else:
        usr_args = {}
        
    # Override with CLI args
    usr_args.update(vars(args_cli))
    
    # Load Task Config
    task_config_path = os.path.join(parent_directory, f"task_config/{args_cli.task_config}.yml")
    with open(task_config_path, "r", encoding="utf-8") as f:
        task_args = yaml.load(f.read(), Loader=yaml.FullLoader)
        
    task_args['task_name'] = args_cli.task_name
    task_args["task_config"] = args_cli.task_config
    task_args["ckpt_setting"] = args_cli.ckpt_setting
    task_args["render_freq"] = args_cli.render_freq # Force render freq
    task_args["eval_mode"] = True # Ensure step_lim is set
    
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
    
    # Update usr_args with arm dims
    usr_args["left_arm_dim"] = len(task_args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(task_args["right_embodiment_config"]["arm_joints_name"][1])
    
    # Initialize Environment
    print(f"Initializing Task: {args_cli.task_name}")
    env = class_decorator(args_cli.task_name)
    
    # Setup Demo / Reset Env
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
                show_observation_images(env, env.now_obs) # Pass current obs if available
            time.sleep(0.1) # Slow down rendering
        env.viewer.render = patched_render
    
    # Initialize Policy
    print(f"Initializing Policy: {args_cli.policy_name}")
    policy_module = importlib.import_module(f"{args_cli.policy_name}.deploy_policy")
    get_model = getattr(policy_module, "get_model")
    encode_obs = getattr(policy_module, "encode_obs")
    
    # Ensure expert_data_num is set correctly
    if usr_args.get("expert_data_num") is None:
        usr_args["expert_data_num"] = args_cli.expert_data_num
        
    model = get_model(usr_args)
    
    # Set Dummy Instruction
    env.set_instruction("do the task")
    
    # Reset Model
    model.env_runner.reset_obs()
    
    # Loop
    print("Starting Episode Loop...")
    step_count = 0
    
    # Initial Observation
    obs = env.get_obs()
    encoded_obs = encode_obs(obs)
    model.update_obs(encoded_obs)
    
    while env.take_action_cnt < env.step_lim:
        # Get Action Chunk
        actions = model.get_action()
        
        for i, action in enumerate(actions):
            step_count += 1
            print(f"\nStep: {step_count}")
            print(f"Action: {action}")
            
            # Print Observation Summary
            print("Observation Keys:", obs.keys())
            if 'qpos' in obs:
                print("QPos:", obs['qpos'])
            
            env.take_action(action)
            
            # Check success
            if env.check_success():
                print("\033[92mSuccess!\033[0m")
                break
                
            obs = env.get_obs()
            encoded_obs = encode_obs(obs)
            model.update_obs(encoded_obs)
            
            # Render is handled by env.take_action if render_freq > 0
            
        if env.check_success():
            break
            
    print("Episode Finished.")
    env.close_env()
    if args_cli.show_obs:
        print("Press Enter to close visualization windows...")
        input()
        if vis_o3d:
            vis_o3d.destroy_window()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
