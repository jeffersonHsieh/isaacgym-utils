import sys
import math
import random
import multiprocessing

import franka_import_tree_multi_env as fit

import isaacgym_loader as ig_loader


import argparse
import make_relative_rotational_dataset as mkori
import yaml
import os
import sys
import numpy as np

'''
This script is for Interactive Perception 2023 paper.
This script loads a tree for visualiziation check
input: PATH to URDF, yaml files
input: Both inputs obtained from running SCA_tree_gen.py main()
output: IG visualizer
'''


print(f" ----------- script starting  ----------- ")

#import URDF
path = '/home/mark/data/IsaacGym/dataset_mark/real_URDF/'
urdf_path = '/home/mark/data/IsaacGym/dataset_mark/real_URDF/[10]tree0.urdf'
yaml_path = '/home/mark/data/IsaacGym/dataset_mark/real_URDF/[10]tree0.yaml'

tree = 0
tree_pts = 10

name_dict =   {'joints': ['joint0_x_to_1', 'joint0_y_to_1', 'joint0_z_to_1', 'joint1_x_to_3', 'joint1_y_to_3', 'joint1_z_to_3', 'joint1_x_to_4', 'joint1_y_to_4', 'joint1_z_to_4', 'joint3_x_to_6', 'joint3_y_to_6', 'joint3_z_to_6', 'joint3_x_to_7', 'joint3_y_to_7', 'joint3_z_to_7', 'joint4_x_to_5', 'joint4_y_to_5', 'joint4_z_to_5', 'joint6_x_to_9', 'joint6_y_to_9', 'joint6_z_to_9', 'joint7_x_to_8', 'joint7_y_to_8', 'joint7_z_to_8', 'joint7_x_to_2', 'joint7_y_to_2', 'joint7_z_to_2'], 'links': ['base_link', 'link_0_to_1', 'link_1_to_3', 'link_1_to_4', 'link2_tip', 'link_3_to_6', 'link_3_to_7', 'link_4_to_5', 'link5_tip', 'link_6_to_9', 'link_7_to_8', 'link_7_to_2', 'link8_tip', 'link9_tip']} 
edge_def = [(0, 1), (1, 2), (1, 3), (2, 5), (2, 6), (3, 7), (7, 8), (5, 9), (6, 10), (6, 11), (10, 12), (9, 13), (11, 4)] 

damping_list = np.array([25]*len(name_dict['joints'])) 
stiffness_list = np.array([50]*len(name_dict['joints'])) 



#import into IG
NUM_JOINTS = len(name_dict['joints'])

K_min = 5
K_max = 500
NUM_K_VAL = 100
NUM_ENVS = 100
stiffness_value_list = np.linspace(K_min, K_max, NUM_K_VAL)
stiffness_value_list = stiffness_value_list.astype(int)

F_push_min = 1
F_push_max = 100
F_push_array = (np.linspace(F_push_min, F_push_max, NUM_ENVS)).astype(int)

#load IG trails with various stiffness values
# def __init__(self, path=DEFAULT_PATH, save_path = SAVE_PATH, stiffness_list=None, stiffness_increment = 10, F_push_array=F_push_array_default):
ig = ig_loader.IG_loader(path = path, save_path = None, stiffness_list = None, stiffness_increment = None)
ig.run_policy_do_nothing()





print(f" ----------- script ending ----------- ")
sys.exit()