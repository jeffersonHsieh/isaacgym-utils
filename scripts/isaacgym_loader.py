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

import copy
import pdb
import sys
import datetime

import isaacgym_tree_policies as tree_policy
from pathlib import Path

DEFAULT_PATH = "/home/mark/github/isaacgym-utils/scripts/dataset/"
SAVE_PATH = "/home/mark/github/isaacgym-utils/scripts/IP_dataset/"


num_iteration = 0 # hard coded to be Equal to NUM ENV * num push

F_push_min = 1
F_push_max = 100
F_push_array_default = (np.linspace(F_push_min, F_push_max, 100)).astype(int)

class IG_loader(object):


    def __init__(self, path=DEFAULT_PATH, save_path = SAVE_PATH, name_dict = None, edge_def = None,  tree_num = 0):

        path = Path(path)
        self.NUM_JOINTS = len(name_dict['joints'])
        self.NUM_LINKS = len(name_dict['links'])
        self.name_dict = name_dict
        self.edge_def = edge_def

        self.urdf_path = str(path/'[10]tree0.urdf')
        self.yaml_path = str(path/'[10]tree0.yaml')
        self.save_path = save_path
        self.tree_num = tree_num


        import pdb;pdb.set_trace()
        #load yaml file with config data about IG param and tree param
        with open(self.yaml_path, "r") as f:
            self.cfg = yaml.load(f, Loader=yaml.Loader)

        
        #create a GymScene object
        self.scene = GymScene(self.cfg['scene'])
        
     
        #create GymVarTree object for reference even though this is created with varying K assets in the setup_inc_K function
        self.tree = vt.GymVarTree(self.cfg['tree'], self.urdf_path, name_dict, self.scene, actuation_mode='joints')
        self.tree_name = 'tree'


        #create a F_push list to apply on each env
        self.NUM_ENVS = self.cfg['scene']['n_envs']

        self.tree.num_links = self.NUM_LINKS

        self.push_num = 0

        self.scene.setup_all_envs(self.setup) 
        

    def setup(self, scene, _):
        tree_transform = gymapi.Transform(p=gymapi.Vec3(0, 0, 0))
        scene.add_asset(self.tree_name, self.tree, tree_transform, collision_filter=1) # avoid self-collisions

        

    def destory_sim(self):
        self.scene._gym.destroy_sim(self.scene._sim)



    def get_link_names_without_duplicates(self):
        '''
        return tree_link names without duplicates for saving data without duplicates
        i.e. tree link names without duplicate (no base, and no shared children)
        '''

        tree_link_without_duplicates = []
        tree_location_no_duplicate_list = []
        
        for link_name in self.name_dict["links"]:
            
            #find parent name from link name (link_1_to_3, link_1_to_4 -> link_1)
            parent_name = link_name.split("to_")[0]

            # print(f" parent_name {parent_name}, self.tree_link_without_duplicates {self.tree_link_without_duplicates}")
            if any (parent_name in _names_ for _names_ in tree_link_without_duplicates) :
                # print(f"this is a duplicate")
                pass

            else:
                if not "base" in link_name:
                    # print(f"adding new")
                    #get link name in string
                    tree_link_without_duplicates.append(link_name)
                    #get link transform
                    tree_location_no_duplicate_list.append(self.tree.get_link_transform(0, self.tree_name, link_name))


        print(f"tree_link_without_duplicates : {tree_link_without_duplicates}")
        return tree_link_without_duplicates, tree_location_no_duplicate_list


        
    def get_link_poses_without_duplicates(self, env_idx):
        '''
        return pose of links without duplicate nodes
        modified to return pose of all links (Excluding duplicate nodes 
        i.e. link1_2, link1_4 should only return link1_2 b/c same pose) 
        '''
        vertex_pos = np.zeros((7, self.NUM_LINKS_WITHOUT_DUPLICATES) ) #x,y,z,qx,qy,qz,qw

        for i in range( (self.NUM_LINKS_WITHOUT_DUPLICATES) ):
            link_tf = self.tree.get_link_transform(env_idx, self.tree_name, self.tree_link_without_duplicates[i])
            pos = vec3_to_np(link_tf.p)
            quat = quat_to_np(link_tf.r)
            #print(link_tf.r)
            #print(quat)

            vertex_pos[0,i] = pos[0]
            vertex_pos[1,i] = pos[1]
            vertex_pos[2,i] = pos[2]
            vertex_pos[3,i] = quat[0]
            vertex_pos[4,i] = quat[1]
            vertex_pos[5,i] = quat[2]
            vertex_pos[6,i] = quat[3]

  
        return vertex_pos

    def get_link_poses(self, env_idx):
        '''
        return pose of all links
        '''
        vertex_pos = np.zeros((7, self.tree.num_links)) #x,y,z,qx,qy,qz,qw

        for i in range(self.tree.num_links):
            link_tf = self.tree.get_link_transform(env_idx, self.tree_name, self.tree.link_names[i])
            pos = vec3_to_np(link_tf.p)
            quat = quat_to_np(link_tf.r)
            #print(link_tf.r)
            #print(quat)

            vertex_pos[0,i] = pos[0]
            vertex_pos[1,i] = pos[1]
            vertex_pos[2,i] = pos[2]
            vertex_pos[3,i] = quat[0]
            vertex_pos[4,i] = quat[1]
            vertex_pos[5,i] = quat[2]
            vertex_pos[6,i] = quat[3]

  
        return vertex_pos


    def custom_draws(self,scene):
        global contact_transform

        for env_idx in scene.env_idxs:
            transforms = []
            for link_name in self.name_dict["links"]:
                transforms.append(self.tree.get_ee_transform_MARK(env_idx, self.tree_name, link_name))
            

            draw_transforms(scene, [env_idx], transforms)


    def run_policy(self):
        self.scene.run(policy=self.policy)

    def run_policy_do_nothing(self):
        self.scene.run(policy=tree_policy.Policy_Do_Nothing(self.tree, self.tree_name) )

    def run_policy_random_pushes(self):
        self.scene.run(policy=self.policy_random_pushes)

     #================================================================================================
    def policy_random_pushes(self, scene, env_idx, t_step, t_sim):
        pass


    #================================================================================================
    def policy(self, scene, env_idx, t_step, t_sim):
        pass
        

    #================================================================================================

def main():
    print(f" ================ starting sample script ================  ")

    DEFAULT_PATH = "/home/mark/course/16745_orcl/OCRL_project_treemanipulate/isaacgym-utils-ocrl/scripts/"
    SAVE_PATH = "/home/mark/course/16745_orcl/OCRL_project_treemanipulate/isaacgym-utils-ocrl/scripts/"

    name_dict = {'joints': ['joint0_x_to_1', 'joint0_y_to_1', 'joint0_z_to_1', 'joint1_x_to_2', 'joint1_y_to_2', 'joint1_z_to_2', 'joint1_x_to_4', 'joint1_y_to_4', 'joint1_z_to_4', 'joint1_x_to_8', 'joint1_y_to_8', 'joint1_z_to_8', 'joint2_x_to_3', 'joint2_y_to_3', 'joint2_z_to_3', 'joint2_x_to_7', 'joint2_y_to_7', 'joint2_z_to_7', 'joint3_x_to_5', 'joint3_y_to_5', 'joint3_z_to_5', 'joint4_x_to_6', 'joint4_y_to_6', 'joint4_z_to_6', 'joint5_x_to_9', 'joint5_y_to_9', 'joint5_z_to_9', 'joint6_x_to_10', 'joint6_y_to_10', 'joint6_z_to_10'], 'links': ['base_link', 'link_0_to_1', 'link_1_to_2', 'link_1_to_4', 'link_1_to_8', 'link_2_to_3', 'link_2_to_7', 'link_3_to_5', 'link_4_to_6', 'link_5_to_9', 'link_6_to_10', 'link7_tip', 'link8_tip', 'link9_tip', 'link10_tip']}
    edge_def = [(0, 1), (1, 2), (1, 3), (1, 4), (2, 5), (2, 6), (5, 7), (3, 8), (7, 9), (8, 10), (6, 11), (4, 12), (9, 13), (10, 14)]

    ig = IG_loader(DEFAULT_PATH, SAVE_PATH, name_dict, edge_def, 0)
    ig.run_policy_do_nothing()
    # ig.destory_sim()
    print(f" ================ ending sample script ================  ")


if __name__ == '__main__':
    main()

