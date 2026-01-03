#!/bin/bash

task_name=stack_blocks_two
task_config=demo_3d_vision_easy
expert_data_num=100

python scripts/process_data_ppi.py $task_name $task_config $expert_data_num