# VGC Policy (Visual Geometric Control)

This repository contains the implementation of the VGC policy for the RoboTwin environment. VGC leverages 3D point cloud geometry and 2D semantic features (from DINOv2) to perform bimanual manipulation tasks.

## Prerequisites

- **RoboTwin Environment**: Ensure the RoboTwin simulation environment is set up.
- **DINOv2**: The policy uses DINOv2 for visual feature extraction.
- **Python Dependencies**: `torch`, `numpy`, `zarr`, `opencv-python`, `hydra-core`, `einops`, `open3d`, etc.

## Workflow

The complete workflow consists of three main steps: Data Preprocessing, Training, and Evaluation.

### 1. Data Preprocessing

Before training, raw demonstration data must be converted into a Zarr dataset, and visual features should be precomputed to accelerate training.

#### Step 1.1: Convert Raw Data to Zarr

Use `process_data_ppi.py` to convert raw HDF5/Pickle data into the Zarr format required by the VGC policy. This script also handles keyframe extraction.

```bash
# Example usage
python policy/VGC/scripts/process_data_ppi.py \
    --source_dir data/beat_block_hammer \
    --dest_path policy/VGC/data/beat_block_hammer-demo_3d_vision_easy-100-ppi.zarr \
    --num_episodes 100
```

*Note: Adjust `source_dir` and `dest_path` according to your data location.*

#### Step 1.2: Precompute DINOv2 Features

To speed up training, we precompute the DINOv2 features for all images in the dataset. This avoids running the heavy vision encoder during every training iteration.

```bash
# Example usage
python policy/VGC/scripts/precompute_dino_features.py \
    --zarr_path policy/VGC/data/beat_block_hammer-demo_3d_vision_easy-100-ppi.zarr \
    --model dinov2_vits14 \
    --batch_size 32 \
    --device cuda
```

*   **Important**: This script applies ImageNet normalization (`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`) to the images before feature extraction.
*   The features are stored directly in the Zarr file under the `dino_features` key.

### 2. Training

Train the VGC policy using the preprocessed data. The training script uses Hydra for configuration.

```bash
# Example usage
python policy/VGC/train.py task=beat_block_hammer
```

*   **Configuration**:
    *   Main config: `policy/VGC/config/vgc_policy.yaml`
    *   Task configs: `policy/VGC/config/task/*.yaml`
*   **Note on Vision Encoder**: During training, the policy is configured to **skip** loading the DINOv2 model and instead load the precomputed features from the Zarr dataset. This is controlled by `policy.use_pc_features=True` (default for training).

### 3. Evaluation / Inference

Evaluate the trained policy in the simulation environment.

```bash
# Usage: bash policy/VGC/eval.sh <task_name> <task_config> <ckpt_setting> <seed> <gpu_id>

# Example
bash eval.sh beat_block_hammer demo_3d_vision_easy debug 0 0
```

*   **Inference Logic**:
    *   During inference, we cannot use precomputed features because the images are coming live from the simulation.
    *   The `deploy_policy.py` script automatically overrides the configuration to set `policy.use_pc_features=False`.
    *   It loads the DINOv2 model (even if it wasn't used/loaded during training) to extract features in real-time.
    *   **Image Processing**: The inference script handles the conversion from the simulation's image format (BGR/RGB) to the format expected by DINOv2 (RGB, normalized).

## Directory Structure

```
policy/VGC/
├── config/                 # Hydra configuration files
├── data/                   # Zarr datasets (generated)
├── scripts/                # Preprocessing and utility scripts
│   ├── process_data_ppi.py
│   ├── precompute_dino_features.py
│   ├── inspect_zarr_images.py
│   └── ...
├── vgc/                    # Source code
│   ├── model/              # Network architecture (Fusion, PointNet, etc.)
│   ├── policy/             # Policy logic
│   └── dataset/            # Dataset loader
├── train.py                # Training entry point
├── deploy_policy.py        # Inference wrapper
└── eval.sh                 # Evaluation script
```
