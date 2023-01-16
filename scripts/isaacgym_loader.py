import argparse

import numpy as np
from numpy import save 
#from autolab_core import YamlConfig, RigidTransform
import yaml
import os

from isaacgym import gymapi
from isaacgym_utils.scene import GymScene
from isaacgym_utils.assets import GymFranka, GymBoxAsset
from isaacgym_utils.assets import variable_tree as vt
from isaacgym_utils.camera import GymCamera
from isaacgym_utils.math_utils import RigidTransform_to_transform, np_to_vec3, vec3_to_np, quat_to_np
from isaacgym_utils.policy import GraspBlockPolicy, MoveBlockPolicy
from isaacgym_utils.draw import draw_transforms, draw_contacts, draw_camera, draw_spheres

import pdb
import sys
import datetime

DEFAULT_PATH = "/home/mark/github/isaacgym-utils/scripts/dataset" #"/mnt/hdd/jan-malte/10Nodes_new_test/" #"/home/jan-malte/Dataset/8Nodes/" #"/home/jan-malte/Dataset/" #"/media/jan-malte/INTENSO/"
urdf_path = '/home/mark/github/isaacgym-utils/scripts/dataset_mark/[3]tree0.urdf'
yaml_path = '/home/mark/github/isaacgym-utils/scripts/dataset_mark/[3]tree0.yaml'
name_dict = {'joints': ['link1_jointx', 'link1_jointy', 'link1_jointz', 'link2_jointx', 'link2_jointy', 'link2_jointz', 'link3_jointx', 'link3_jointy', 'link3_jointz'],  
'links': ['base_link', 'link1x', 'link1z', 'link1', 'link2x', 'link2z', 'link2', 'link3x', 'link3z', 'link3']}

NUM_JOINTS = len(name_dict['joints'])
edge_def = [(0, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 6), (2, 7), (3, 8), (6, 9), (4, 10), (9, 11), (7, 12), (8, 13), (5, 14), (11, 15)]
damping_list = np.array([25]*NUM_JOINTS) 
stiffness_list = np.array([50]*NUM_JOINTS) 



class IG_loader(object):
    def __init__(self, stiffness_list=stiffness_list):
  
        #load yaml file with config data about IG param and tree param
        with open(yaml_path, "r") as f:
            self.cfg = yaml.load(f, Loader=yaml.Loader)

        #create a GymScene object
        self.scene = GymScene(self.cfg['scene'])
        #create GymVarTree object. This may not be needed if we create a new GymVarTree object in the setup function
        self.tree = vt.GymVarTree(self.cfg['tree'], urdf_path, name_dict, self.scene, actuation_mode='joints')
        self.tree_name = 'tree'

        # self.scene.setup_all_envs(self.setup) # to create the same tree in all envs
        self.scene.setup_all_envs_inc_K(self.setup_inc_K) #to create varying K tree in all envs
        
        self.policy_loop_counter = 0   

        
    def setup(self, scene, _):
        tree_transform = gymapi.Transform(p=gymapi.Vec3(0, 0, 0))
        scene.add_asset(self.tree_name, self.tree, tree_transform, collision_filter=1) # avoid self-collisions

    def setup_inc_K(self, scene, _, K_incremeter):
        print(f" === inside setup a tree asset ===  ")

        self.cfg['tree']['dof_props']['stiffness'] = stiffness_list + K_incremeter
        print(f" cfg: {self.cfg['tree']['dof_props']['stiffness']}  ")

        self.tree = vt.GymVarTree(self.cfg['tree'], urdf_path, name_dict, self.scene, actuation_mode='joints')
        self.tree_name = 'tree'

        tree_transform = gymapi.Transform(p=gymapi.Vec3(0, 0, 0))
        scene.add_asset(self.tree_name, self.tree, tree_transform, collision_filter=1) # avoid self-collisions



    def destory_sim(self):
        self.scene._gym.destroy_sim(self.scene._sim)

    def run_policy(self):
        self.scene.run(policy=self.policy)


    
    #================================================================================================
    def policy(self, scene, env_idx, t_step, t_sim):
        self.policy_loop_counter += 1
        print(f" self.policy_loop_counter {self.policy_loop_counter}  ")
        if self.policy_loop_counter >= 10:
           
           #save data

           return True

    #================================================================================================

def main():
    print(f" ================ starting sample script ================  ")
    ig = IG_loader()
    ig.run_policy()
    ig.destory_sim()
    print(f" ================ ending sample script ================  ")


if __name__ == '__main__':
    main()

