import math
import random

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
import argparse

from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits import mplot3d

import os
import sys
import yaml

#custom util lib for tree IG
# import combine_dataset_files as combine
import SCA_tree_gen as sca
import isaacgym_loader as ig_loader
from pathlib import Path

''' 
##############################################################
This script is for creating a tree structure for the Isaac Gym simu

input: SCA parameters
output: URDF, YAML of tree (files)
output: name[dict], edge def (print out)
output: box list for collision
##############################################################
''' 

# output file path
PATH = str(Path(__file__).parent.parent/"assets/franka_description/robots/")
SAVE_PATH = str(Path(__file__).parent.parent/"assets/franka_description/robots/")
import pdb;pdb.set_trace()
## Parameters for the SCA tree generation algo. Lists specify possible parameters, of which one is chosen at random for every tree
SCALING = 1
PIPE_MODEL_EXPONENT = 3 #suggested values: 2 or 3
STEP_WIDTH = 0.5 #determines the lenght of individual tree segments
HEIGHT_STRECH_VALS = [0.05, 0.10] #[0.5, 0.33, 1] #factors to strech the crown shape of the tree to be non circular
WIDTH_STRECH_VALS = [1.0, 9.0] #[0.5, 0.33, 1] #factors to strech the crown shape of the tree to be non circular
ATT_PTS_NUM = [80, 160, 320, 640] #800, 1600, 3200, 6400 number of attraction points

TREE_PTS = 10 #number of tree links (how big is the tree)
env_num = 1
gui_on = True



def main():
    print(f" ================ starting sample script ================  ")

    # ================ create tree =================
    tree = 0 #tree id

    trunk_height = STEP_WIDTH * 0.75 / SCALING #TRUNK_HEIGHT_FACTORS[random.randrange(0, len(TRUNK_HEIGHT_FACTORS))] / SCALING
    d_termination = SCALING/10
    d_attraction_values = [math.ceil(trunk_height)+1, math.ceil(trunk_height) + 2, math.ceil(trunk_height) + 4, math.ceil(trunk_height) + 8, math.ceil(trunk_height) + 16, math.ceil(trunk_height) + 32, math.ceil(trunk_height) + 64]
    d_attraction = d_attraction_values[random.randrange(0, len(d_attraction_values) - 1)]
    height_strech = HEIGHT_STRECH_VALS[random.randrange(0,len(HEIGHT_STRECH_VALS)-1)]
    width_strech = WIDTH_STRECH_VALS[random.randrange(0,len(WIDTH_STRECH_VALS)-1)]
    att_pts_max = ATT_PTS_NUM[random.randrange(0, len(ATT_PTS_NUM)-1)]
    print("tree%s: \n\t d_termination: %s \n\t d_attraction: %s \n\t height_strech: %s \n\t width_strech: %s \n\t att_pts_max: %s"%(tree, d_termination, d_attraction, height_strech, width_strech, att_pts_max))
    tg = sca.TreeGenerator(path=PATH,max_steps=10000, att_pts_max=att_pts_max, da=d_attraction, dt=d_termination, step_width=STEP_WIDTH, offset=[-0.5, -0.5, trunk_height], scaling=SCALING, max_tree_points=TREE_PTS, tip_radius=0.008, tree_id=tree, pipe_model_exponent=PIPE_MODEL_EXPONENT, z_strech=height_strech, y_strech=width_strech, x_strech=width_strech, step_width_scaling=0.65, env_num=env_num, gui_on=gui_on)
    tg.generate_tree()
    tg.calculate_branch_thickness()
    name_dict, edge_def, urdf_path = tg.generate_urdf()
    # import pdb;pdb.set_trace()
    yaml_path, stiffness_list, damping_list = tg.generate_yaml()
    edge_def2 = tg.calc_edge_tuples()

    print(f" urdf_path: {urdf_path}")
    print(f" name_dict: {name_dict}")
    print(f" edge_def: {edge_def}")

    # sys.exit()

 

    # load the tree in IG for visualization debugging
    # if gui_on:
    #     num_iter= 0 #num of pushes (not desired for OCRL project)

    #     ig = ig_loader.IG_loader(PATH, SAVE_PATH, name_dict = name_dict, edge_def = edge_def, tree_num = tree)
    #     ig.run_policy_do_nothing()

    # print(f" ================ ending sample script ================  ")


if __name__ == '__main__':
    main()