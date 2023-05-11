#!/bin/bash

conda activate rlgpu
mkdir -p exp

LOG="../exp/res/tree1_log.txt"

for kp in 100 500 700 1000 2000
do 
for kd in 0.1 10 30 50 100
do
    python franka_tree_control_exp.py --kp $kp --kd $kd --log_file $LOG

done
done
