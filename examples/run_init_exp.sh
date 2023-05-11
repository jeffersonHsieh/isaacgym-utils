#!/bin/bash

conda activate rlgpu
EXPDIR="../exp"
LOG="$EXPDIR/res/all_tree_logs.txt"
echo $LOG
mkdir -p $EXPDIR

for method in `ls -d ../exp/traj/*/`
do 
for fname in `ls $method*`
do
    echo $fname
    python franka_tree_control_exp.py --kp 2000 --kd 0.1 --log_file $LOG --traj $fname

done
done
