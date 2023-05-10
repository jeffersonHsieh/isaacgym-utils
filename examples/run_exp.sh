#!/bin/bash

conda activate rlgpu
mkdir -p exp

LOG="exp/pick_block_log.txt"

for kp in 5 7 10
do 
for kd in 0.7 0.8
do
    python franka_tree_control_exp.py --kp $kp --kd $kd --log_file $LOG

done
done
