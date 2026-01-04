
import sys
import os
import torch
import numpy as np
import zarr
import cv2
import argparse
from tqdm import tqdm
import torchvision.transforms as T
from einops import rearrange

# Add paths
current_file_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_file_path)
vgc_dir = os.path.dirname(scripts_dir)
sys.path.append(vgc_dir)

# Import DINO loading logic (or re-implement to be self-contained)
# We'll re-implement the loading part to be robust
def load_dinov2(model_name='dinov2_vits14', device='cuda'):
    print(f"Loading {model_name}...")
    try:
        model = torch.hub.load('facebookresearch/dinov2', model_name)
    except Exception as e:
        print(f"Failed to load from torch.hub: {e}")
        print("Attempting to load from local cache...")
        try:
            local_repo_path = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
            if os.path.exists(local_repo_path):
                model = torch.hub.load(local_repo_path, model_name, source='local')
            else:
                raise FileNotFoundError(f"Local repo not found at {local_repo_path}")
        except Exception as local_e:
            raise local_e
            
    model.to(device)
    model.eval()
    return model

def precompute_features(zarr_path, model_name='dinov2_vits14', batch_size=32, device='cuda'):
    print(f"Processing {zarr_path}")
    
    # Open Zarr
    root = zarr.open(zarr_path, mode='r+')
    
    if 'images' not in root['data']:
        print("No images found in dataset.")
        return

    images = root['data']['images'] # (N, K, H, W, C) or (N, K, C, H, W)?
    # Based on vgc_dataset.py, images are stored as (N, K, H, W, C) usually uint8
    
    N, K, H, W, C = images.shape
    print(f"Dataset shape: {images.shape}")
    
    # Check if already exists
    if 'dino_features' in root['data']:
        print("dino_features already exists. Skipping or Overwriting? (Overwriting)")
        # return
        
    # Load Model
    model = load_dinov2(model_name, device)
    
    # Normalization
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Determine feature shape
    # Run one dummy forward pass
    with torch.no_grad():
        dummy_img = torch.zeros(1, 3, 224, 224).to(device)
        features_dict = model.forward_features(dummy_img)
        patch_features = features_dict['x_norm_patchtokens']
        # (1, N_patches, D_feat)
        _, n_patches, d_feat = patch_features.shape
        print(f"Feature shape per image: ({n_patches}, {d_feat})")
        
    # Create Zarr array for features
    # Shape: (N, K, n_patches, d_feat)
    feature_shape = (N, K, n_patches, d_feat)
    chunks = (1, K, n_patches, d_feat)
    
    if 'dino_features' in root['data']:
        dino_features = root['data']['dino_features']
    else:
        dino_features = root['data'].create_dataset('dino_features', shape=feature_shape, chunks=chunks, dtype='float32', overwrite=True)
        
    # Process in batches
    num_batches = (N + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="Extracting Features"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, N)
        
        # Load batch images
        batch_imgs_np = images[start_idx:end_idx] # (B, K, H, W, C)
        B = batch_imgs_np.shape[0]
        
        # Preprocess
        # Resize to 224x224, Normalize to 0-1, Transpose to CHW
        # We can do this efficiently
        batch_imgs_processed = []
        for b in range(B):
            imgs_k = []
            for k in range(K):
                img = batch_imgs_np[b, k]
                # Resize
                img = cv2.resize(img, (224, 224))
                # Normalize 0-1
                img = img.astype(np.float32) / 255.0
                imgs_k.append(img)
            batch_imgs_processed.append(np.stack(imgs_k))
            
        batch_imgs_processed = np.stack(batch_imgs_processed) # (B, K, 224, 224, 3)
        batch_imgs_processed = np.moveaxis(batch_imgs_processed, -1, 2) # (B, K, 3, 224, 224)
        
        # Flatten B and K
        batch_tensor = torch.from_numpy(batch_imgs_processed).float().to(device)
        batch_tensor = rearrange(batch_tensor, 'b k c h w -> (b k) c h w')
        
        # Normalize (ImageNet)
        batch_tensor = normalize(batch_tensor)
        
        # Forward
        with torch.no_grad():
            features_dict = model.forward_features(batch_tensor)
            batch_features = features_dict['x_norm_patchtokens'] # (B*K, N_patches, D_feat)
            
        # Reshape back
        batch_features = rearrange(batch_features, '(b k) n d -> b k n d', b=B, k=K)
        
        # Save to Zarr
        dino_features[start_idx:end_idx] = batch_features.cpu().numpy()
        
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr_path", type=str, help="Path to Zarr dataset")
    parser.add_argument("--model", type=str, default="dinov2_vits14", help="DINOv2 model name")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    precompute_features(args.zarr_path, args.model, args.batch_size, args.device)
