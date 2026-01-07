# bash eval.sh ${task_name} ${task_config} ${ckpt_setting} ${expert_data_num} ${seed} ${gpu_id}
bash eval.sh beat_block_hammer demo_3d_vision_easy demo_3d_vision_easy 100 0 0
# This command trains the policy using the `demo_clean` setting ($ckpt_setting)
# and evaluates it using the same `demo_clean` setting ($task_config).
#
# To evaluate a policy trained on the `demo_clean` setting and tested on the `demo_randomized` setting, run:
# bash eval.sh beat_block_hammer demo_randomized demo_clean 50 0 0