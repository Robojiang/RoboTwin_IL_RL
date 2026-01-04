cd policy/VGC/data/outputs
rsync -avz --no-o --no-g -e "ssh -p 10074" \
root@118.145.32.133:/mnt/afs/250010074/robot_manipulation/RoboTwin_IL_RL/policy/VGC/data/outputs/ \
./