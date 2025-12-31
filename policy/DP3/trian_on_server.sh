
cd /mnt/afs/250010074/robot_manipulation/RoboTwin_IL_RL/policy/DP3

# source conda.sh 让 conda activate 能用
source /opt/miniconda3/etc/profile.d/conda.sh

conda activate robotwin


bash train_rgb.sh  stack_blocks_two demo_3d_vision_easy 100 0 0
bash train_rgb.sh  beat_block_hammer demo_3d_vision_easy 100 0 0
bash train.sh stack_blocks_two demo_3d_vision_easy 100 0 0
bash train.sh beat_block_hammer demo_3d_vision_easy 100 0 0
