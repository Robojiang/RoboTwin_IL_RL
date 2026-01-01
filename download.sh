cd /media/tao/E8F6F2ECF6F2BA40/bimanial_manipulation/RoboTwin/policy/DP3/checkpoints
rsync -avz --no-o --no-g -e "ssh -p 11074" \
root@118.145.32.133:/mnt/afs/250010074/robot_manipulation/RoboTwin_IL_RL/policy/DP3/checkpoints/ \
./