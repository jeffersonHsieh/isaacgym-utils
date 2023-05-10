from abc import ABC, abstractmethod

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


class Policy(ABC):

    def __init__(self):
        self._time_horizon = -1

    @abstractmethod
    def __call__(self, scene, env_idx, t_step, t_sim):
        pass

    def reset(self):
        pass

    @property
    def time_horizon(self):
        return self._time_horizon


class Policy_Do_Nothing(Policy):
    def __init__(self, tree, tree_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tree = tree
        self.tree_name = tree_name

        self._time_horizon = 5


    def __call__(self, scene, env_idx, t_step, t_sim):
        # print(f"Policy_Do_Nothing: {t_sim}")
        pass
        

class Policy_Random_Pushes(Policy):
    def __init__(self, tree, tree_name, num_iteration, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tree = tree
        self.tree_name = tree_name
        self.num_iteration = num_iteration
        print(f" *********** self.num_iteration: {self.num_iteration} *********** ")


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

    def __call__(self, scene, env_idx, t_step, t_sim):
        #counter
        sec_interval = t_sim%1
        sec_counter = int(t_sim)

        ### DETECT STABILIZATION ###
        # if sec_interval == 0 or sec_interval == 0.5:
        #     self.current_pos[env_idx] = self.get_link_poses(env_idx)
        #     if np.sum(np.linalg.norm(np.round(self.last_pos[env_idx][:3] - self.current_pos[env_idx][:3], 5))) == 0 or sec_counter - self.last_timestamp[env_idx] > 30: #tree has stabilized at original position
        #         self.push_switch[env_idx] = not self.push_switch[env_idx]
        #         self.last_timestamp[env_idx] = sec_counter
        #     self.last_pos[env_idx] = self.current_pos[env_idx]


        # if self.push_switch[env_idx]:#ten_sec_interval > 5:

        #     # print(f" *** For env {env_idx}, env push: {self.F_push_counter_env[env_idx]}, global push: {self.push_num} /  {num_iteration} , self.not_saved: {self.not_saved[env_idx] }  ")

        #     #let each env only push through F_push_max_index times
        #     if self.F_push_counter_env[env_idx] <= self.F_push_max_index:

        #         ### BREAK CONTACT PROTOCOL (execute when push_switch[env_idx] turns false) ###
        #         if self.no_contact[env_idx] == False:
        #             self.vertex_final_pos[env_idx] = self.get_link_poses_without_duplicates(env_idx)
        #             #print("vertex_final: %s"%datetime.datetime.now())
        #             print(f" i: {self.push_num}/{self.num_iteration} ")
        #             print(f"===== breaking contact ========")
        #             #print(vertex_init_pos[env_idx][:3]-vertex_final_pos[env_idx][:3])
        #             #print("env%s saves"%env_idx)
        #             if env_idx in self.vertex_init_pos_dict.keys():
        #                 self.vertex_init_pos_dict[env_idx].append(self.vertex_init_pos[env_idx])
        #             else:   
        #                 self.vertex_init_pos_dict[env_idx] = [self.vertex_init_pos[env_idx]]
                    
        #             if env_idx in self.vertex_final_pos_dict.keys():
        #                 self.vertex_final_pos_dict[env_idx].append(self.vertex_final_pos[env_idx])
        #             else:
        #                 self.vertex_final_pos_dict[env_idx] = [self.vertex_final_pos[env_idx]]

        #             if env_idx in self.force_applied_dict.keys():
        #                 self.force_applied_dict[env_idx].append(self.force_applied[env_idx])
        #             else:
        #                 self.force_applied_dict[env_idx] = [self.force_applied[env_idx]]
        #             self.push_num += 1 #globally counted
 
                    
        #             self.no_contact[env_idx] = True
        #             self.force = np_to_vec3([0, 0, 0])

        #             ### APPLY ZERO-FORCE ###
        #         self.tree.apply_force(env_idx, self.tree_name, self.name_dict["links"][2], self.force, self.tree_location_list[2].p)


        #     #save after all push has been done for all env
        #     if self.push_num >= self.num_iteration and self.not_saved[env_idx]:
        #         #print(np.shape(vertex_init_pos_list))

        #         self.save_data(env_idx, self.vertex_init_pos_dict[env_idx], self.vertex_final_pos_dict[env_idx], self.force_applied_dict[env_idx])
        #         self.not_saved[env_idx] = False
        #         self.done[env_idx] = True
        #     if all(self.done):
        #         print(f" ------ policy all done --------- ")
        #         return True

        # else:

        #     ### INITIALIZE CONTACT PROTOCOL ###
        #     if self.no_contact[env_idx] == True:

        #         #let each env only push through F_push_max_index times
        #         if self.F_push_counter_env[env_idx] < self.F_push_max_index:

        #             self.vertex_init_pos[env_idx] = self.get_link_poses_without_duplicates(env_idx)
        #             #print("vertex_init: %s"%datetime.datetime.now())
        #             self.no_contact[env_idx] = False

        #             #force from push array [NUM_PUSH x 4] idx,Fx,Fy,Fz
        #             #iterate through F push array but with environment specific index
        #             #location from push array idx [NUM_PUSH x 4] idx,Fx,Fy,Fz THIS IS FOR REAL2SIM DATA ONLY


        #             push_index_env = self.F_push_counter_env[env_idx] 
        #             force_input_loaded = self.F_push_array[push_index_env, 1:]
        #             self.force = np_to_vec3(list(force_input_loaded))
        #             # self.force = np_to_vec3([-10,-10,0])
        #             # self.force = np_to_vec3([fx, fy, fz])
        #             self.force_vecs[env_idx] = self.force

                    
        #             push_node_idx = (self.F_push_array[push_index_env, 0]).astype(int)
        #             self.rand_idxs[env_idx] = push_node_idx


        #             # random push index from leagal push indices (!base nor tip) THIS IS FOR SIM DATA ONLY
        #             # random_index = np.random.randint(0, len(self.legal_push_indices)) #roll on the list of legal push indices
        #             # random_index = self.legal_push_indices[random_index] # extract the real random push index

        #             self.force_applied[env_idx] = self.set_force( vec3_to_np(self.force), self.rand_idxs[env_idx])
                    

                    

        #             self.contact_transform = self.tree_location_no_duplicate_list[self.rand_idxs[env_idx]]
        #             contact_name = self.tree_link_without_duplicates[self.rand_idxs[env_idx]]
        #             #print(tree.link_names[random_index])

        #             print(f"===== env_idx {env_idx}, push count: {self.F_push_counter_env[env_idx]}, making contact {contact_name} with F: {self.force} at node: { self.rand_idxs[env_idx] } ========")

                    
        #             self.F_push_counter_env[env_idx] += 1 #increment the push counter for the specific env

        #             #print(self.rand_idxs)
        #             #contact_draw(scene, env_idx, contact_transform)
            
        #     ### APPLY FORCE  outside constantly ###
        #     self.tree.apply_force(env_idx, self.tree_name, self.tree_link_without_duplicates[self.rand_idxs[env_idx]], self.force_vecs[env_idx], self.tree_location_no_duplicate_list[self.rand_idxs[env_idx]].p)
        # return False
           


# #================================================================================================
def policy_random_pushes(self, scene, env_idx, t_step, t_sim):

    # #get pose 
    # tree_tf3 = tree.get_link_transform(0, tree_name, name_dict["links"][2])

    # #create random force

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

        ### BREAK CONTACT PROTOCOL (execute when push_switch[env_idx] turns false) ###
        if self.no_contact[env_idx] == False:
            self.vertex_final_pos[env_idx] = self.get_link_poses(env_idx)
            #print("vertex_final: %s"%datetime.datetime.now())
            print(self.push_num)
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
            #for x in range(0, scene._n_envs):
            #    if x in self.vertex_init_pos_dict.keys():
            #        print(len(self.vertex_init_pos_dict[x]))
            #print(cmpr.all())
            
            self.no_contact[env_idx] = True
            self.force = np_to_vec3([0, 0, 0])
                # # force = np_to_vec3([np.random.rand()*force_magnitude, np.random.rand()*force_magnitude, np.random.rand()*force_magnitude])

            ### APPLY ZERO-FORCE ###
        self.tree.apply_force(env_idx, self.tree_name, name_dict["links"][2], self.force, self.tree_location_list[2].p)

        if self.push_num >= num_iteration and self.not_saved[env_idx]:
            #print(np.shape(vertex_init_pos_list))

            self.save_data(env_idx, self.vertex_init_pos_dict[env_idx], self.vertex_final_pos_dict[env_idx], self.force_applied_dict[env_idx])
            self.not_saved[env_idx] = False
            self.done[env_idx] = True
        if all(self.done):
            print(f" ------ policy all done --------- ")
            return True

            #sys.exit()
    else:

        ### INITIALIZE CONTACT PROTOCOL ###
        if self.no_contact[env_idx] == True:

            self.vertex_init_pos[env_idx] = self.get_link_poses(env_idx)
            #print("vertex_init: %s"%datetime.datetime.now())
            self.no_contact[env_idx] = False

            #for idx in range(0, scene._n_envs):
            #force random
            while True:
                sx = np.random.randint(0,2)
                fx = np.random.randint(10,30)
                if sx == 0:
                    fx = -fx

                sy = np.random.randint(0,2)
                fy = np.random.randint(10,30)
                if sy == 0:
                    fy = -fy

                sz = np.random.randint(0,2)
                fz = np.random.randint(10,30)
                if sz == 0:
                    fz = -fz
                if abs(fx) + abs(fy) + abs(fz) != 0:
                    break
            

            self.force = np_to_vec3([self.F_push_array[env_idx],0,0])
            # self.force = np_to_vec3([-10,-10,0])
            # self.force = np_to_vec3([fx, fy, fz])
            self.force_vecs[env_idx] = self.force
            

            #location random
            random_index = np.random.randint(0, len(self.legal_push_indices)) #roll on the list of legal push indices
            random_index = self.legal_push_indices[random_index] # extract the real random push index
            random_index = 8
            
            self.rand_idxs[env_idx] = random_index

            
            # print(f"[fx,fy,fz] {[fx,fy,fz]} ")


            self.force_applied[env_idx] = self.set_force( vec3_to_np(self.force), self.rand_idxs[env_idx])
            

            self.contact_transform = self.tree_location_list[self.rand_idxs[env_idx]]
            contact_name = self.tree.link_names[self.rand_idxs[env_idx]]
            #print(tree.link_names[random_index])

            print(f"===== env_idx {env_idx}, making contact {contact_name} with F {self.force} ========")

        #print(self.rand_idxs)
        #contact_draw(scene, env_idx, contact_transform)
        ### APPLY RANDOM-FORCE ###
        self.tree.apply_force(env_idx, self.tree_name, self.tree.link_names[self.rand_idxs[env_idx]], self.force_vecs[env_idx], self.tree_location_list[self.rand_idxs[env_idx]].p)
    return False
