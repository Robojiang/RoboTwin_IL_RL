import zarr
import numpy as np
import cv2
import os
import argparse

def inspect_zarr(zarr_path):
    print(f"Inspecting: {zarr_path}")
    
    if not os.path.exists(zarr_path):
        print("Path does not exist.")
        return

    try:
        root = zarr.open(zarr_path, mode='r')
    except Exception as e:
        print(f"Failed to open zarr: {e}")
        return

    print("\n=== Zarr Structure ===")
    try:
        print(root.tree())
    except:
        # Fallback if tree() is not available or fails
        def print_visitor(obj):
            print(obj.name)
        root.visitvalues(print_visitor)
    
    if 'data' not in root or 'images' not in root['data']:
        print("\nNo 'images' dataset found in /data.")
        return

    images = root['data']['images']
    print(f"\n=== Images Dataset ===")
    print(f"Shape: {images.shape}")
    print(f"Dtype: {images.dtype}")
    
    # Sample the first image
    # Shape is usually (N, K, H, W, C) or (N, K, C, H, W)
    # Based on previous context, it seems to be (N, K, H, W, C)
    
    first_sample = images[0]
    print(f"Sample shape (Time=0): {first_sample.shape}")
    
    # Assuming (K, H, W, C)
    num_cams = first_sample.shape[0]
    
    output_dir = "inspection_output"
    os.makedirs(output_dir, exist_ok=True)
    
    for k in range(num_cams):
        img = first_sample[k]
        
        # Check if channel first or last
        if img.shape[0] == 3: # (C, H, W)
            img = np.moveaxis(img, 0, -1)
            
        print(f"Camera {k} image shape: {img.shape}, Range: [{img.min()}, {img.max()}]")
        
        # Handle float vs int
        if img.dtype == np.float32 or img.dtype == np.float64:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        
        # Save as is (assuming RGB)
        # cv2.imwrite expects BGR, so if the data is RGB, we need to convert RGB -> BGR for saving
        # If the data is BGR, we save as is.
        
        # We will save two versions to help user decide
        
        # 1. Save assuming data is RGB (so we convert to BGR for opencv)
        save_path_rgb = os.path.join(output_dir, f"cam_{k}_assuming_data_is_rgb.png")
        cv2.imwrite(save_path_rgb, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # 2. Save assuming data is BGR (so we save directly)
        save_path_bgr = os.path.join(output_dir, f"cam_{k}_assuming_data_is_bgr.png")
        cv2.imwrite(save_path_bgr, img)
        
    print(f"\nSaved sample images to directory: {os.path.abspath(output_dir)}")
    print("Please check the images. If 'assuming_data_is_rgb.png' looks correct (colors are natural), then your Zarr contains RGB data.")
    print("If 'assuming_data_is_bgr.png' looks correct, then your Zarr contains BGR data.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("zarr_path", type=str, help="Path to Zarr dataset")
    args = parser.parse_args()
    
    inspect_zarr(args.zarr_path)
