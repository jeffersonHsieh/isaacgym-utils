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

PATH = "/mnt/hdd/jan-malte/10Nodes_new_test/" #"/home/jan-malte/Dataset/8Nodes/" #"/home/jan-malte/Dataset/" #"/media/jan-malte/INTENSO/"

def import_tree(name_dict, urdf_path, yaml_path, edge_def, stiffness_list, damping_list, tree_num, tree_pts, path=PATH, num_iteration=10000):
    global no_contact, force, loc_tree, random_index, contact_transform, not_saved
    with open(yaml_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    scene = GymScene(cfg['scene'])

    tree = vt.GymVarTree(cfg['tree'], urdf_path, name_dict, scene, actuation_mode='joints')

    tree_name = 'tree'

    current_iteration = 0
    force_magnitude = 10
    push_toggle = True

    def setup(scene, mutable_idx):
        #x = mutable_idx #- mutable_idx%10
        #y = mutable_idx
        tree_transform = gymapi.Transform(p=gymapi.Vec3(0, 0, 0))
        scene.add_asset(tree_name, tree, tree_transform, collision_filter=1) # avoid self-collisions

    scene.setup_all_envs(setup)   

    def contact_draw(scene, env_idx, loc_tree ):
        
        for env_idx in scene.env_idxs:
            contact_transform = (loc_tree)
            #draw_transforms_contact(scene, [env_idx], [contact_transform])

    def custom_draws(scene):
        global contact_transform

        for env_idx in scene.env_idxs:
            transforms = []
            for link_name in name_dict["links"]:
                transforms.append(tree.get_ee_transform_MARK(env_idx, tree_name, link_name))
            

            draw_transforms(scene, [env_idx], transforms)
        draw_contacts(scene, scene.env_idxs)

    
    no_contact = [True] * scene._n_envs
    not_saved = [True] * scene._n_envs
    force = np_to_vec3([0, 0, 0])

    def get_link_poses(env_idx):
        vertex_pos = np.zeros((7,tree.num_links)) #x,y,z,qx,qy,qz,qw

        for i in range(tree.num_links):
            link_tf = tree.get_link_transform(env_idx, tree_name, tree.link_names[i])
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

    # TODO: ask mark what this function is supposed to do, and why it just assumes
    def get_stiffness():
        coeffecients = np.zeros((2, tree.num_joints)) #K stiffness, d damping
        #stiff_k = 600
        #damping = 400
        coeffecients[0,:] = np.array(stiffness_list)
        coeffecients[1,:] = np.array(damping_list)  

        return coeffecients


    def set_force(force, index):
        force_applied_ = np.zeros((3,tree.num_links))
        force_applied_[0,index] = force[0]
        force_applied_[1,index] = force[1]
        force_applied_[2,index] = force[2]

        return force_applied_ 


    def save_data(env_idx, vertex_init_pos_list_arg, vertex_final_pos_list_arg, force_applied_list_arg):
        #coeff_stiff_damp = get_stiffness()

        #if five_sec_counter == (num_iteration+1):
        #vertex_init_pos_list.append(vertex_init_pos)
        #vertex_final_pos_list.append(vertex_final_pos)
        #force_applied_list.append(force_applied)

        #print(len(vertex_final_pos_list))

        print(f" ********* saving data ********* ")
        #print(np.shape(vertex_init_pos_list_arg))
        save(path + '[%s]X_vertex_init_pos_tree%s_env%s'%(tree_pts, tree_num, env_idx), vertex_init_pos_list_arg )
        #save('X_coeff_stiff_damp_tree%s_env%s'%(tree_num, env_idx), coeff_stiff_damp )
        #save('X_edge_def_tree%s_env%s'%(tree_num, env_idx), edge_def )
        save(path + '[%s]X_force_applied_tree%s_env%s'%(tree_pts, tree_num, env_idx), force_applied_list_arg )
        save(path + '[%s]Y_vertex_final_pos_tree%s_env%s'%(tree_pts, tree_num, env_idx), vertex_final_pos_list_arg )

        #print(f"Vinit, Vfinal, Fapplied lengths: {vertex_init_pos_list}")
        #sys.exit() 

        #print(f" ---- appending data {ten_sec_counter}th time ---- ")
        #print(f"Vinit, Vfinal, Fapplied lengths: {len(vertex_init_pos_list)},  {len(vertex_final_pos_list)},  {len(force_applied_list)}")
        #print(f"vertex_init_pos {vertex_init_pos}")
        #print(f"vertex_final_pos {vertex_final_pos}")
        #print(f"force_applied {force_applied}")

            
    tree_location_list = []
    legal_push_indices = []
    idx = 0
    for link_name in name_dict["links"]:
        tree_location_list.append(tree.get_link_transform(0, tree_name, link_name))
        if not "base" in link_name and not "tip" in link_name: # Exclude base from being a push option
            legal_push_indices.append(idx)
        idx += 1

    #print(legal_push_indices)
    #print(len(tree_location_list))

    global contact_transform
    contact_transform = tree_location_list[0]
    
    loc_tree = tree_location_list[2].p
    random_index = 1

    global rand_idxs, force_vecs, current_pos, last_timestamp, push_switch, done, vertex_init_pos_list, vertex_final_pos_list, force_applied_list, vertex_init_pos, vertex_final_pos, force_applied, last_pos, push_num
    vertex_init_pos_dict = {}#[[]] * scene._n_envs
    vertex_final_pos_dict = {}#[[]] * scene._n_envs
    force_applied_dict = {}#[[]] * scene._n_envs
    push_num = 0

    vertex_init_pos = [np.zeros((7,tree.num_links))] * scene._n_envs #x,y,z,qx,qy,qz,qw
    vertex_final_pos = [np.zeros((7,tree.num_links))] * scene._n_envs #x,y,z,qx,qy,qz,qw
    last_pos = [np.zeros((7,tree.num_links))] * scene._n_envs #x,y,z,qx,qy,qz,qw
    current_pos = [np.zeros((7,tree.num_links))] * scene._n_envs
    force_applied = [np.zeros((3,tree.num_links))] * scene._n_envs #fx,fy,fz
    force_vecs = [np_to_vec3([0, 0, 0])]*scene._n_envs
    rand_idxs = [0]*scene._n_envs
    done = [False] * scene._n_envs
    push_switch = [False] * scene._n_envs
    last_timestamp = [0] * scene._n_envs

    coeff_stiff_damp = get_stiffness()
    save(PATH + '[%s]X_coeff_stiff_damp_tree%s'%(tree_pts,tree_num), coeff_stiff_damp)
    save(PATH + '[%s]X_edge_def_tree%s'%(tree_pts,tree_num), edge_def)

    def policy(scene, env_idx, t_step, t_sim): #TODO: Fix issue where this saves init and final vetor identically
        global rand_idxs, force_vecs, current_pos, last_timestamp, push_switch, done, push_num, last_pos, no_contact, force, loc_tree, random_index, contact_transform, force_vecs, rand_idxs, vertex_init_pos_list, vertex_final_pos_list, force_applied_list, vertex_init_pos, vertex_final_pos, force_applied, not_saved
        # #get pose 
        # tree_tf3 = tree.get_link_transform(0, tree_name, name_dict["links"][2])

        # #create random force

        #counter
        sec_interval = t_sim%1
        sec_counter = int(t_sim)

        ### DETECT STABILIZATION ###
        if sec_interval == 0 or sec_interval == 0.5:
            current_pos[env_idx] = get_link_poses(env_idx)
            if np.sum(np.linalg.norm(np.round(last_pos[env_idx][:3] - current_pos[env_idx][:3], 5))) == 0 or sec_counter - last_timestamp[env_idx] > 60: #tree has stabilized at original position
                push_switch[env_idx] = not push_switch[env_idx]
                last_timestamp[env_idx] = sec_counter
            last_pos[env_idx] = current_pos[env_idx]


        if push_switch[env_idx]:#ten_sec_interval > 5:

            ### BREAK CONTACT PROTOCOL (execute when push_switch[env_idx] turns false) ###
            if no_contact[env_idx] == False:
                vertex_final_pos[env_idx] = get_link_poses(env_idx)
                #print("vertex_final: %s"%datetime.datetime.now())
                print(push_num)
                print(f"===== breaking contact ========")
                #print(vertex_init_pos[env_idx][:3]-vertex_final_pos[env_idx][:3])
                #print("env%s saves"%env_idx)
                if env_idx in vertex_init_pos_dict.keys():
                    vertex_init_pos_dict[env_idx].append(vertex_init_pos[env_idx])
                else:   
                    vertex_init_pos_dict[env_idx] = [vertex_init_pos[env_idx]]
                
                if env_idx in vertex_final_pos_dict.keys():
                    vertex_final_pos_dict[env_idx].append(vertex_final_pos[env_idx])
                else:
                    vertex_final_pos_dict[env_idx] = [vertex_final_pos[env_idx]]

                if env_idx in force_applied_dict.keys():
                    force_applied_dict[env_idx].append(force_applied[env_idx])
                else:
                    force_applied_dict[env_idx] = [force_applied[env_idx]]

                #for x in range(0, scene._n_envs):
                #    if x in vertex_init_pos_dict.keys():
                #        print(len(vertex_init_pos_dict[x]))
                #print(cmpr.all())
                
                no_contact[env_idx] = True
                force = np_to_vec3([0, 0, 0])
                 # # force = np_to_vec3([np.random.rand()*force_magnitude, np.random.rand()*force_magnitude, np.random.rand()*force_magnitude])
                #loc_tree = tree_location_list[2].p
                push_num += 1 #globally counted
            
            ### APPLY ZERO-FORCE ###
            tree.apply_force(env_idx, tree_name, name_dict["links"][2], force, tree_location_list[2].p)

            if push_num >= num_iteration and not_saved[env_idx]:
                #print(np.shape(vertex_init_pos_list))
                save_data(env_idx, vertex_init_pos_dict[env_idx], vertex_final_pos_dict[env_idx], force_applied_dict[env_idx])
                not_saved[env_idx] = False
                done[env_idx] = True
            if all(done):
                return True
                #sys.exit()
        else:

            ### INITIALIZE CONTACT PROTOCOL ###
            if no_contact[env_idx] == True:

                vertex_init_pos[env_idx] = get_link_poses(env_idx)
                #print("vertex_init: %s"%datetime.datetime.now())
                no_contact[env_idx] = False

                #for idx in range(0, scene._n_envs):
                #force random
                while True:
                    sx = np.random.randint(0,1)
                    fx = np.random.randint(10,30)
                    if sx == 0:
                        fx = -fx

                    sy = np.random.randint(0,1)
                    fy = np.random.randint(10,30)
                    if sy == 0:
                        fy = -fy

                    sz = np.random.randint(0,1)
                    fz = np.random.randint(10,30)
                    if sz == 0:
                        fz = -fz
                    if abs(fx) + abs(fy) + abs(fz) != 0:
                        break
                
                force = np_to_vec3([fx, fy, fz])
                force_vecs[env_idx] = force
                #force = np_to_vec3([-10,-10,0])

                #location random
                random_index = np.random.randint(0, len(legal_push_indices)) #roll on the list of legal push indices
                random_index = legal_push_indices[random_index] # extract the real random push index
                rand_idxs[env_idx] = random_index

                force_applied[env_idx] = set_force([fx,fy,fz], rand_idxs[env_idx])
                
                #loc_tree = tree_location_list[rand_idxs[env_idx]].p
                contact_transform = tree_location_list[rand_idxs[env_idx]]
                contact_name = tree.link_names[rand_idxs[env_idx]]
                #print(tree.link_names[random_index])

                print(f"===== making contact {contact_name} with F {force} ========")

            #print(rand_idxs)
            #contact_draw(scene, env_idx, contact_transform)
            ### APPLY RANDOM-FORCE ###
            tree.apply_force(env_idx, tree_name, tree.link_names[rand_idxs[env_idx]], force_vecs[env_idx], tree_location_list[rand_idxs[env_idx]].p)
        return False

    scene.run(policy=policy)

    # clean up to allow multiple runs
    #if scene._viewer is not None:
    #   scene._gym.destroy_viewer(scene._viewer)
    scene._gym.destroy_sim(scene._sim)