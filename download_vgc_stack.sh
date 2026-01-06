cd policy/VGC/checkpoints
rsync -avz --no-o --no-g -e "ssh -p 10074" \
root@118.145.32.133:/mnt/afs/250010074/robot_manipulation/RoboTwin_IL_RL/policy/VGC/data/outputs/22.15.09_vgc_policy_stack_blocks/checkpoints/latest.ckpt \
./stack_blocks_two/