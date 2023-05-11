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

# TODO: rewrite to allow for multi environment simulation
def import_tree(name_dict, urdf_path, yaml_path, edge_def, stiffness_list, damping_list, tree_num):
    global vertex_init_pos, no_contact, force, loc_tree, vertex_final_pos, force_applied, random_index, contact_transform, init_fri
    with open(yaml_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    scene = GymScene(cfg['scene'])

    tree = vt.GymVarTree(cfg['tree'], urdf_path, name_dict, scene, actuation_mode='joints')

    tree_transform = gymapi.Transform(p=gymapi.Vec3(0, 0, 0))

    tree_name = 'tree'

    current_iteration = 0
    num_iteration = 100
    force_magnitude = 10
    push_toggle = True
    
    global vertex_init_pos, vertex_final_pos, force_applied 

    vertex_init_pos = np.zeros((7,tree.num_links)) #x,y,z,qx,qy,qz,qw
    vertex_final_pos = np.zeros((7,tree.num_links)) #x,y,z,qx,qy,qz,qw
    force_applied = np.zeros((3,tree.num_links)) #fx,fy,fz     

    def setup(scene, _):
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

    
    no_contact = True
    init_fri = True
    force = np_to_vec3([0, 0, 0])

    def get_link_poses():
        vertex_pos = np.zeros((7,tree.num_links)) #x,y,z,qx,qy,qz,qw

        for i in range(tree.num_links):
            link_tf = tree.get_link_transform(0, tree_name, tree.link_names[i])
            pos = vec3_to_np(link_tf.p)
            quat = quat_to_np(link_tf.r)

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


    
    vertex_init_pos_list = []
    vertex_final_pos_list = []
    force_applied_list = []

    def save_data(save_bool=False):
        coeff_stiff_damp = get_stiffness()

        #if five_sec_counter == (num_iteration+1):
        if save_bool:
            vertex_init_pos_list.append(vertex_init_pos)
            vertex_final_pos_list.append(vertex_final_pos)
            force_applied_list.append(force_applied)

            #print(len(vertex_final_pos_list))

            print(f" ********* saving data ********* ")
            save('X_vertex_init_pose_tree%s_pid%s'%(tree_num, os.getpid()), vertex_init_pos_list )
            save('X_coeff_stiff_damp_tree%s_pid%s'%(tree_num, os.getpid()), coeff_stiff_damp )
            save('X_edge_def_tree%s_pid%s'%(tree_num, os.getpid()), edge_def )
            save('X_force_applied_tree%s_pid%s'%(tree_num, os.getpid()), force_applied_list )
            save('Y_vertex_final_pos_tree%s_pid%s'%(tree_num, os.getpid()), vertex_final_pos_list )

            #print(f"Vinit, Vfinal, Fapplied lengths: {vertex_init_pos_list}")
            #sys.exit() 

        else:
            vertex_init_pos_list.append(vertex_init_pos)
            vertex_final_pos_list.append(vertex_final_pos)
            force_applied_list.append(force_applied)

            #print(f" ---- appending data {ten_sec_counter}th time ---- ")
            #print(f"Vinit, Vfinal, Fapplied lengths: {len(vertex_init_pos_list)},  {len(vertex_final_pos_list)},  {len(force_applied_list)}")
            #print(f"vertex_init_pos {vertex_init_pos}")
            #print(f"vertex_final_pos {vertex_final_pos}")
            #print(f"force_applied {force_applied}")

            
    tree_location_list = []

    for link_name in name_dict["links"]:
        tree_location_list.append(tree.get_link_transform(0, tree_name, link_name))

    global contact_transform
    contact_transform = tree_location_list[0]
    
    loc_tree = tree_location_list[2].p
    random_index = 1

    # TODO: make sure all of the performed moves (every env) are recorded properly.
    def policy(scene, env_idx, t_step, t_sim):
        global vertex_init_pos, no_contact, force, loc_tree, vertex_final_pos, force_applied, random_index, contact_transform, init_fri, force_vecs, rand_idxs
        # #get pose 
        # tree_tf3 = tree.get_link_transform(0, tree_name, name_dict["links"][2])

        # #create random force

        #counter
        five_sec_interval = t_sim%5
        five_sec_counter = int(t_sim//5)

        if five_sec_interval < 3:
            if no_contact == False:
                print(f"===== breaking contact ========")
                vertex_final_pos = get_link_poses()
                save_data()
                no_contact = True
                force = np_to_vec3([0, 0, 0])
                 # # force = np_to_vec3([np.random.rand()*force_magnitude, np.random.rand()*force_magnitude, np.random.rand()*force_magnitude])
                loc_tree = tree_location_list[2].p
            
            tree.apply_force(env_idx, tree_name, name_dict["links"][2], force, loc_tree)
        
        if init_fri == True:
           force_vecs = [np_to_vec3([0, 0, 0])]*scene._n_envs
           rand_idxs = [0]*scene._n_envs
           init_fri = False

        if five_sec_interval > 3:
            if no_contact == True:

                vertex_init_pos = get_link_poses()
                no_contact = False

                for idx in range(0, scene._n_envs):
                    #force random
                    fx = np.random.randint(-force_magnitude,force_magnitude)
                    fy = np.random.randint(-force_magnitude,force_magnitude)
                    fz = np.random.randint(-force_magnitude,force_magnitude)
                    force = np_to_vec3([fx, fy, fz])
                    force_vecs[idx] = force
                    #force = np_to_vec3([-10,-10,-10])

                    #location random
                    random_index = np.random.randint(1, len(tree_location_list))
                    rand_idxs[idx] = random_index

                force_applied = set_force([fx,fy,fz], rand_idxs[env_idx])
                
                loc_tree = tree_location_list[rand_idxs[env_idx]].p
                contact_transform = tree_location_list[rand_idxs[env_idx]]
                contact_name = tree.link_names[rand_idxs[env_idx]]
                #print(tree.link_names[random_index])

                print(f"===== making contact {contact_name} with F {force} ========")

            #print(rand_idxs)
            contact_draw(scene, env_idx, contact_transform)
            tree.apply_force(env_idx, tree_name, tree.link_names[rand_idxs[env_idx]], force_vecs[env_idx], loc_tree)
               

        # get delta pose

        # release tree
    time_horizon = int(num_iteration * 5 / scene.dt)
    scene.run(policy=policy, time_horizon=time_horizon)
    save_data(save_bool=True)

    # clean up to allow multiple runs
    if scene._viewer is not None:
        scene._gym.destroy_viewer(scene._viewer)
    scene._gym.destroy_sim(scene._sim)