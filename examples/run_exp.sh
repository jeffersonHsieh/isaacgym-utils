#!/bin/bash

conda activate rlgpu
mkdir -p exp

LOG="exp/pick_block_log.txt"

for kp in 0.1 1 3 5 10
do 
for kd in 0.1 1 3 5 10
do
    python franka_tree_control_exp.py --kp $kp --kd $kd --logfile $LOG

done
done
