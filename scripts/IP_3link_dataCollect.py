import sys
import math
import random
import multiprocessing
import SCA_tree_gen as sca
import franka_import_tree_multi_env as fit

import isaacgym_loader as ig_loader


import argparse
import combine_dataset_files as combine
import make_relative_rotational_dataset as mkori
import yaml
import os
import sys
import numpy as np

'''
This script is for Interactive Perception 2023 paper.
This script loads a small 3 link tree URDF and varies K stiffness to observe displacement upon fixed push force.
input: PATH to 3 link URDF
input: K stiffness yaml files
output: X_final, Y_final, X_edge_def, X_force_applied
'''


print(f" ----------- script starting  ----------- ")

#import URDF
urdf_path = '/home/mark/github/isaacgym-utils/scripts/dataset_mark/[3]tree0.urdf'
yaml_path = '/home/mark/github/isaacgym-utils/scripts/dataset_mark/[3]tree0.yaml'

name_dict = {'joints': ['link1_jointx', 'link1_jointy', 'link1_jointz', 'link2_jointx', 'link2_jointy', 'link2_jointz', 'link3_jointx', 'link3_jointy', 'link3_jointz'],  
'links': ['base_link', 'link1x', 'link1z', 'link1', 'link2x', 'link2z', 'link2', 'link3x', 'link3z', 'link3']}

edge_def = [(0, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 6), (2, 7), (3, 8), (6, 9), (4, 10), (9, 11), (7, 12), (8, 13), (5, 14), (11, 15)]
damping_list = np.array([25]*len(name_dict['joints'])) 
stiffness_list = np.array([50]*len(name_dict['joints'])) 

tree = 0
tree_pts = 3

save_path = '/home/mark/github/isaacgym-utils/scripts/dataset_mark/IP_data/'
num_iter = 1

#import into IG
NUM_JOINTS = 9
stiffness_value_list = [10,20,30]


# #load IG trails with various stiffness values
# for stiffness_value in stiffness_value_list:
#     stiffness_list = np.array([stiffness_value]*NUM_JOINTS) 
#     ig = ig_loader.IG_loader(stiffness_list)
#     ig.run_policy()

#     ig.destory_sim()

# ============ just single loop =============

ig = ig_loader.IG_loader(stiffness_list)
ig.run_policy()



#get data from IG

#save data




print(f" ----------- script ending ----------- ")
sys.exit()