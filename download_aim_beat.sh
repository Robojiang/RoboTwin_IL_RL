cd policy/AIM/checkpoints
rsync -avz --no-o --no-g -e "ssh -p 10074" \
root@118.145.32.133:/mnt/afs/250010074/robot_manipulation/RoboTwin_IL_RL/policy/AIM/data/outputs/21.27.19_aim_policy_beat_block_hammer/checkpoints/latest.ckpt \
./beat_block_hammer/