#!/bin/bash
# beat_block_hammer stack_blocks_two
task_name=beat_block_hammer
task_config=demo_3d_vision_easy
expert_data_num=100

python scripts/process_data_ppi.py $task_name $task_config $expert_data_num

# Pre-compute DINOv2 features
zarr_path="data/${task_name}-${task_config}-${expert_data_num}-ppi.zarr"
python scripts/precompute_dino_features.py $zarr_path