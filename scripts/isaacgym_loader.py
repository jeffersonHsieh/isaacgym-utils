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

DEFAULT_PATH = "/home/mark/github/isaacgym-utils/scripts/dataset/"
SAVE_PATH = "/home/mark/github/isaacgym-utils/scripts/IP_dataset/"


name_dict =   {'joints': ['joint0_x_to_1', 'joint0_y_to_1', 'joint0_z_to_1', 'joint1_x_to_3', 'joint1_y_to_3', 'joint1_z_to_3', 'joint1_x_to_4', 'joint1_y_to_4', 'joint1_z_to_4', 'joint3_x_to_6', 'joint3_y_to_6', 'joint3_z_to_6', 'joint3_x_to_7', 'joint3_y_to_7', 'joint3_z_to_7', 'joint4_x_to_5', 'joint4_y_to_5', 'joint4_z_to_5', 'joint6_x_to_9', 'joint6_y_to_9', 'joint6_z_to_9', 'joint7_x_to_8', 'joint7_y_to_8', 'joint7_z_to_8', 'joint7_x_to_2', 'joint7_y_to_2', 'joint7_z_to_2'], 'links': ['base_link', 'link_0_to_1', 'link_1_to_3', 'link_1_to_4', 'link2_tip', 'link_3_to_6', 'link_3_to_7', 'link_4_to_5', 'link5_tip', 'link_6_to_9', 'link_7_to_8', 'link_7_to_2', 'link8_tip', 'link9_tip']} 
edge_def = [(0, 1), (1, 2), (1, 3), (2, 5), (2, 6), (3, 7), (7, 8), (5, 9), (6, 10), (6, 11), (10, 12), (9, 13), (11, 4)] 



NUM_JOINTS = len(name_dict['joints'])
NUM_LINKS = len(name_dict['links'])

damping_list = np.array([25]*NUM_JOINTS) 
stiffness_list = np.array([50]*NUM_JOINTS) 
stiffness_increment = 10

num_iteration = 0 # hard coded to be Equal to NUM ENV * num push

F_push_min = 1
F_push_max = 100
F_push_array_default = (np.linspace(F_push_min, F_push_max, 100)).astype(int)

class IG_loader(object):
    def __init__(self, path=DEFAULT_PATH, save_path = SAVE_PATH, stiffness_list=None, stiffness_increment = 10, F_push_array=F_push_array_default):

        global num_iteration

        self.urdf_path = path + '[10]tree0.urdf'
        self.yaml_path = path + '[10]tree0.yaml'
        self.save_path = save_path

  
        #load yaml file with config data about IG param and tree param
        with open(self.yaml_path, "r") as f:
            self.cfg = yaml.load(f, Loader=yaml.Loader)

        
        #create a GymScene object
        self.scene = GymScene(self.cfg['scene'])
        
        #either take input param constant K values or use the ones given in the yaml file
        if stiffness_list is not None:
            self.stiffness_list = stiffness_list
            self.cfg['tree']['dof_props']['stiffness'] = stiffness_list

        #HARD CODE DAMPING VALUE FOR FASTER DATA COLLECTION
        self.cfg['tree']['dof_props']['damping'] = 5 * np.ones(NUM_JOINTS)

        
        #create GymVarTree object for reference even though this is created with varying K assets in the setup_inc_K function
        self.tree = vt.GymVarTree(self.cfg['tree'], self.urdf_path, name_dict, self.scene, actuation_mode='joints')
        self.tree_name = 'tree'

        #load multiple envs with varying K 
        if stiffness_increment is not None:

            if len(stiffness_increment) > 1:
                print(f"Creating {self.cfg['scene']['n_envs']} IG ENVs with varying K across ENV, varying K in trees")
                self.scene.setup_all_envs_varying_K(self.setup_vary_K, stiffness_increment) 

            elif len(stiffness_increment) == 1:
                print(f"Creating {self.cfg['scene']['n_envs']} IG ENVs with varying K across ENV, but consant K used for all tree nodes in each ENV")
                self.scene.setup_all_envs_inc_K(self.setup_inc_K, stiffness_list[0], stiffness_increment) #to create varying K tree in all envs

        #load multiple envs with same K
        else:
            self.scene.setup_all_envs(self.setup) # to create the same tree in all envs
        

        #create a F_push list to apply on each env
        self.NUM_ENVS = self.cfg['scene']['n_envs']
        self.NUM_PUSHES = len(F_push_array)
        num_iteration = self.NUM_ENVS * self.NUM_PUSHES


        self.F_push_array = F_push_array
        self.F_push_max_index = self.F_push_array.shape[0]




        self.policy_loop_counter = 0   
        self.tree.num_links = NUM_LINKS

        self.push_num = 0
        self.stiffness_value = self.cfg['tree']['dof_props']['stiffness'][0] 

        


        

        


        

        self.vertex_init_pos_dict = {}#[[]] * scene._n_envs
        self.vertex_final_pos_dict = {}#[[]] * scene._n_envs
        self.force_applied_dict = {}#[[]] * scene._n_envs
        self.tree_location_list = []
        self.legal_push_indices = []


        idx = 0
        for link_name in name_dict["links"]:
            self.tree_location_list.append(self.tree.get_link_transform(0, self.tree_name, link_name))
            if not "base" in link_name and not "tip" in link_name: # Exclude base from being a push option
                self.legal_push_indices.append(idx)
            idx += 1


        self.tree_link_without_duplicates, self.tree_location_no_duplicate_list = self.get_link_names_without_duplicates()
        self.NUM_LINKS_WITHOUT_DUPLICATES = len(self.tree_link_without_duplicates)

        print(f" NUM_LINKS : {NUM_LINKS}, NUM_LINKS_WITHOUT_DUPLICATES: {self.NUM_LINKS_WITHOUT_DUPLICATES}, NUM_JOINTS : {NUM_JOINTS}")


        self.tree.num_links = self.NUM_LINKS_WITHOUT_DUPLICATES

        self.vertex_init_pos = [np.zeros((7,self.tree.num_links))] * self.scene._n_envs #x,y,z,qx,qy,qz,qw
        self.vertex_final_pos = [np.zeros((7,self.tree.num_links))] * self.scene._n_envs #x,y,z,qx,qy,qz,qw
        self.last_pos = [np.zeros((7,self.tree.num_links))] * self.scene._n_envs #x,y,z,qx,qy,qz,qw
        self.current_pos = [np.zeros((7,self.tree.num_links))] * self.scene._n_envs
        self.force_applied = [np.zeros((3,self.tree.num_links))] * self.scene._n_envs #fx,fy,fz
        self.force_vecs = [np_to_vec3([0, 0, 0])]*self.scene._n_envs
        self.rand_idxs = [0]*self.scene._n_envs
        self.done = [False] * self.scene._n_envs
        self.push_switch = [False] * self.scene._n_envs
        self.last_timestamp = [0] * self.scene._n_envs

        self.no_contact = [True] * self.scene._n_envs
        self.not_saved = [True] * self.scene._n_envs
        self.F_push_counter_env = [0] * self.scene._n_envs
        self.force = np_to_vec3([0, 0, 0])

        self.contact_transform = self.tree_location_list[0]


    def setup(self, scene, _):
        tree_transform = gymapi.Transform(p=gymapi.Vec3(0, 0, 0))
        scene.add_asset(self.tree_name, self.tree, tree_transform, collision_filter=1) # avoid self-collisions

    def setup_inc_K(self, scene, _, K_incremeter):
        # print(f" === inside setup a tree asset ===  ")
        self.cfg['tree']['dof_props']['stiffness'] = self.stiffness_list + K_incremeter
        # print(f" cfg: {self.cfg['tree']['dof_props']['stiffness']}  ")

        self.tree = vt.GymVarTree(self.cfg['tree'], self.urdf_path, name_dict, self.scene, actuation_mode='joints')
        self.tree_name = 'tree'

        tree_transform = gymapi.Transform(p=gymapi.Vec3(0, 0, 0))
        scene.add_asset(self.tree_name, self.tree, tree_transform, collision_filter=1) # avoid self-collisions

    def setup_vary_K(self, scene, _, K_multiplier):
        # print(f" === inside setup a tree asset with K_multiplier:  {K_multiplier} ===  ")
        temp_cfg = copy.deepcopy(self.cfg)
        
        temp_cfg['tree']['dof_props']['stiffness'] = np.asarray(self.cfg['tree']['dof_props']['stiffness'])  * K_multiplier
        # print(f" === temp_cfg['tree']['dof_props']['stiffness'] = {temp_cfg['tree']['dof_props']['stiffness']} ===")

        self.tree = vt.GymVarTree(temp_cfg['tree'], self.urdf_path, name_dict, self.scene, actuation_mode='joints')
        self.tree_name = 'tree'

        tree_transform = gymapi.Transform(p=gymapi.Vec3(0, 0, 0))
        scene.add_asset(self.tree_name, self.tree, tree_transform, collision_filter=1) # avoid self-collisions



    def destory_sim(self):
        self.scene._gym.destroy_sim(self.scene._sim)

    def get_K_stiffness(self, env_idx):
        K_stiffness = self.tree.get_stiffness(env_idx, self.tree_name)
        # print(f" K_stiffness: {K_stiffness} ")
        return K_stiffness


    def get_link_names_without_duplicates(self):
        '''
        return tree_link names without duplicates for saving data without duplicates
        i.e. tree link names without duplicate (no base, and no shared children)
        '''

        tree_link_without_duplicates = []
        tree_location_no_duplicate_list = []
        
        for link_name in name_dict["links"]:
            
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

    def set_force(self,force, index):

        force_applied_ = np.zeros((3,self.tree.num_links))
        force_applied_[0,index] = force[0]
        force_applied_[1,index] = force[1]
        force_applied_[2,index] = force[2]

        # print(f"force_applied_[0,{index}]  {force_applied_[:,index] } ")

        return force_applied_ 

    def custom_draws(self,scene):
        global contact_transform

        for env_idx in scene.env_idxs:
            transforms = []
            for link_name in name_dict["links"]:
                transforms.append(self.tree.get_ee_transform_MARK(env_idx, self.tree_name, link_name))
            

            draw_transforms(scene, [env_idx], transforms)


    def save_data(self,env_idx, vertex_init_pos_list_arg, vertex_final_pos_list_arg, force_applied_list_arg):
        print(f" ======================== saving data  ========================  ")
        # K = self.get_K_stiffness(env_idx)[0] 
        # print(f" force_applied_list_arg {force_applied_list_arg} ")
        # print(f" vertex_final_pos_list_arg {vertex_final_pos_list_arg} ")

        NUM_LINKS = 14
        save(self.save_path + '[%s]X_force_applied_treeK_env%s'%(NUM_LINKS,   env_idx), force_applied_list_arg )
        save(self.save_path + '[%s]Y_vertex_final_pos_treeK_env%s'%(NUM_LINKS, env_idx), vertex_final_pos_list_arg )
        save(self.save_path + '[%s]X_vertex_init_pos_treeK_env%s'%(NUM_LINKS,  env_idx), vertex_init_pos_list_arg )
        

    def run_policy(self):
        self.scene.run(policy=self.policy)

    def run_policy_do_nothing(self):
        self.scene.run(policy=tree_policy.policy_do_nothing)

    
    #================================================================================================
    def policy(self, scene, env_idx, t_step, t_sim):

        # #get pose 
        # tree_tf3 = tree.get_link_transform(0, tree_name, name_dict["links"][2])


        #counter
        sec_interval = t_sim%1
        sec_counter = int(t_sim)

        ### DETECT STABILIZATION ###
        if sec_interval == 0 or sec_interval == 0.5:
            self.current_pos[env_idx] = self.get_link_poses(env_idx)
            if np.sum(np.linalg.norm(np.round(self.last_pos[env_idx][:3] - self.current_pos[env_idx][:3], 5))) == 0 or sec_counter - self.last_timestamp[env_idx] > 30: #tree has stabilized at original position
                self.push_switch[env_idx] = not self.push_switch[env_idx]
                self.last_timestamp[env_idx] = sec_counter
            self.last_pos[env_idx] = self.current_pos[env_idx]


        if self.push_switch[env_idx]:#ten_sec_interval > 5:

            # print(f" *** For env {env_idx}, env push: {self.F_push_counter_env[env_idx]}, global push: {self.push_num} /  {num_iteration} , self.not_saved: {self.not_saved[env_idx] }  ")

            #let each env only push through F_push_max_index times
            if self.F_push_counter_env[env_idx] <= self.F_push_max_index:

                ### BREAK CONTACT PROTOCOL (execute when push_switch[env_idx] turns false) ###
                if self.no_contact[env_idx] == False:
                    self.vertex_final_pos[env_idx] = self.get_link_poses_without_duplicates(env_idx)
                    #print("vertex_final: %s"%datetime.datetime.now())
                    print(f" i: {self.push_num}/{num_iteration} ")
                    print(f"===== breaking contact ========")
                    #print(vertex_init_pos[env_idx][:3]-vertex_final_pos[env_idx][:3])
                    #print("env%s saves"%env_idx)
                    if env_idx in self.vertex_init_pos_dict.keys():
                        self.vertex_init_pos_dict[env_idx].append(self.vertex_init_pos[env_idx])
                    else:   
                        self.vertex_init_pos_dict[env_idx] = [self.vertex_init_pos[env_idx]]
                    
                    if env_idx in self.vertex_final_pos_dict.keys():
                        self.vertex_final_pos_dict[env_idx].append(self.vertex_final_pos[env_idx])
                    else:
                        self.vertex_final_pos_dict[env_idx] = [self.vertex_final_pos[env_idx]]

                    if env_idx in self.force_applied_dict.keys():
                        self.force_applied_dict[env_idx].append(self.force_applied[env_idx])
                    else:
                        self.force_applied_dict[env_idx] = [self.force_applied[env_idx]]
                    self.push_num += 1 #globally counted
 
                    
                    self.no_contact[env_idx] = True
                    self.force = np_to_vec3([0, 0, 0])

                    ### APPLY ZERO-FORCE ###
                self.tree.apply_force(env_idx, self.tree_name, name_dict["links"][2], self.force, self.tree_location_list[2].p)


            #save after all push has been done for all env
            if self.push_num >= num_iteration and self.not_saved[env_idx]:
                #print(np.shape(vertex_init_pos_list))

                self.save_data(env_idx, self.vertex_init_pos_dict[env_idx], self.vertex_final_pos_dict[env_idx], self.force_applied_dict[env_idx])
                self.not_saved[env_idx] = False
                self.done[env_idx] = True
            if all(self.done):
                print(f" ------ policy all done --------- ")
                return True

        else:

            ### INITIALIZE CONTACT PROTOCOL ###
            if self.no_contact[env_idx] == True:

                #let each env only push through F_push_max_index times
                if self.F_push_counter_env[env_idx] < self.F_push_max_index:

                    self.vertex_init_pos[env_idx] = self.get_link_poses_without_duplicates(env_idx)
                    #print("vertex_init: %s"%datetime.datetime.now())
                    self.no_contact[env_idx] = False

                    #force from push array [NUM_PUSH x 4] idx,Fx,Fy,Fz
                    #iterate through F push array but with environment specific index
                    #location from push array idx [NUM_PUSH x 4] idx,Fx,Fy,Fz THIS IS FOR REAL2SIM DATA ONLY


                    push_index_env = self.F_push_counter_env[env_idx] 
                    force_input_loaded = self.F_push_array[push_index_env, 1:]
                    self.force = np_to_vec3(list(force_input_loaded))
                    # self.force = np_to_vec3([-10,-10,0])
                    # self.force = np_to_vec3([fx, fy, fz])
                    self.force_vecs[env_idx] = self.force

                    
                    push_node_idx = (self.F_push_array[push_index_env, 0]).astype(int)
                    self.rand_idxs[env_idx] = push_node_idx


                    # random push index from leagal push indices (!base nor tip) THIS IS FOR SIM DATA ONLY
                    # random_index = np.random.randint(0, len(self.legal_push_indices)) #roll on the list of legal push indices
                    # random_index = self.legal_push_indices[random_index] # extract the real random push index

                    self.force_applied[env_idx] = self.set_force( vec3_to_np(self.force), self.rand_idxs[env_idx])
                    

                    

                    self.contact_transform = self.tree_location_no_duplicate_list[self.rand_idxs[env_idx]]
                    contact_name = self.tree_link_without_duplicates[self.rand_idxs[env_idx]]
                    #print(tree.link_names[random_index])

                    print(f"===== env_idx {env_idx}, push count: {self.F_push_counter_env[env_idx]}, making contact {contact_name} with F: {self.force} at node: { self.rand_idxs[env_idx] } ========")

                    
                    self.F_push_counter_env[env_idx] += 1 #increment the push counter for the specific env

                    #print(self.rand_idxs)
                    #contact_draw(scene, env_idx, contact_transform)
            
            ### APPLY FORCE  outside constantly ###
            self.tree.apply_force(env_idx, self.tree_name, self.tree_link_without_duplicates[self.rand_idxs[env_idx]], self.force_vecs[env_idx], self.tree_location_no_duplicate_list[self.rand_idxs[env_idx]].p)
        return False

    #================================================================================================

def main():
    print(f" ================ starting sample script ================  ")
    ig = IG_loader()
    ig.run_policy()
    ig.destory_sim()
    print(f" ================ ending sample script ================  ")


if __name__ == '__main__':
    main()

