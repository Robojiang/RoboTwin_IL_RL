cd /mnt/afs/250010074/robot_manipulation/RoboTwin_IL_RL/policy/AIM

# source conda.sh 让 conda activate 能用
source /opt/miniconda3/etc/profile.d/conda.sh

conda activate robotwin

python train.py  task=beat_block_hammer 