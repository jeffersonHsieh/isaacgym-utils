import argparse

import numpy as np
from numpy import save 
from autolab_core import YamlConfig, RigidTransform

from isaacgym import gymapi
from isaacgym_utils.scene import GymScene
from isaacgym_utils.assets import GymFranka, GymBoxAsset
from isaacgym_utils.assets import GymTree
from isaacgym_utils.camera import GymCamera
from isaacgym_utils.math_utils import RigidTransform_to_transform, np_to_vec3, vec3_to_np, quat_to_np
from isaacgym_utils.policy import GraspBlockPolicy, MoveBlockPolicy
from isaacgym_utils.draw import draw_transforms, draw_contacts, draw_camera, draw_spheres
import pdb
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='/home/jan-malte/OCRL_project/isaacgym-utils/cfg/franka_tree_force.yaml')
    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)

 

    scene = GymScene(cfg['scene'])


    franka = GymFranka(cfg['franka'], scene, actuation_mode='torques')
    tree = GymTree(cfg['tree'], scene, actuation_mode='joints')

    # block = GymBoxAsset(scene, **cfg['block']['dims'],  shape_props=cfg['block']['shape_props'])


    franka_transform = gymapi.Transform(p=gymapi.Vec3(1, 1, 0))
    tree_transform = gymapi.Transform(p=gymapi.Vec3(0, 0, 0))

    franka_name, tree_name, block_name = 'franka', 'tree', 'block'

    current_iteration = 0
    num_iteration = 100
    force_magnitude = 50
    push_toggle = True
    
    global vertex_init_pos, vertex_final_pos, force_applied 

    vertex_init_pos = np.zeros((7,tree.num_links)) #x,y,z,qx,qy,qz,qw
    vertex_final_pos = np.zeros((7,tree.num_links)) #x,y,z,qx,qy,qz,qw
    force_applied = np.zeros((3,tree.num_links)) #fx,fy,fz     

    



    def setup(scene, _):

        scene.add_asset(franka_name, franka, franka_transform, collision_filter=1) # avoid self-collisions

        scene.add_asset(tree_name, tree, tree_transform, collision_filter=1) # avoid self-collisions
        # scene.add_asset('block', block, gymapi.Transform(p=gymapi.Vec3(-1, -1, cfg['block']['dims']['sz']/2)) )

    scene.setup_all_envs(setup)    


    def contact_draw(scene, env_idx, loc_tree ):
        
        for env_idx in scene.env_idxs:
            # print(f"random index {random_index}")

            contact_transform = (loc_tree)


    def custom_draws(scene):
        global contact_transform

        for env_idx in scene.env_idxs:

            ee_transform_0 = tree.get_ee_transform_MARK(env_idx, tree_name, 'link1')
            ee_transform_1 = tree.get_ee_transform_MARK(env_idx, tree_name, 'link2')
            ee_transform_2 = tree.get_ee_transform_MARK(env_idx, tree_name, 'link3')
            ee_transform_3 = tree.get_ee_transform_MARK(env_idx, tree_name, 'link4')
            ee_transform_4 = tree.get_ee_transform_MARK(env_idx, tree_name, 'link5')
            ee_transform_5 = tree.get_ee_transform_MARK(env_idx, tree_name, 'link6')
            ee_transform_6 = tree.get_ee_transform_MARK(env_idx, tree_name, 'link7')
            ee_transform_7 = tree.get_ee_transform_MARK(env_idx, tree_name, 'link8')
            ee_transform_8 = tree.get_ee_transform_MARK(env_idx, tree_name, 'link9')
            ee_transform_9 = tree.get_ee_transform_MARK(env_idx, tree_name, 'link10')
            ee_transform_10 = tree.get_ee_transform_MARK(env_idx, tree_name, 'link11')
            ee_transform_11 = tree.get_ee_transform_MARK(env_idx, tree_name, 'link12')

            ee_transform_l7 = tree.get_ee_transform_MARK(env_idx, tree_name, 'link7_leaf')
            ee_transform_l8 = tree.get_ee_transform_MARK(env_idx, tree_name, 'link8_leaf')
            ee_transform_l9 = tree.get_ee_transform_MARK(env_idx, tree_name, 'link9_leaf')
            ee_transform_l10 = tree.get_ee_transform_MARK(env_idx, tree_name, 'link10_leaf')
            ee_transform_l11 = tree.get_ee_transform_MARK(env_idx, tree_name, 'link11_leaf')
            ee_transform_l12 = tree.get_ee_transform_MARK(env_idx, tree_name, 'link12_leaf')

            transforms = [ee_transform_0, ee_transform_1, ee_transform_2, ee_transform_3, ee_transform_6, ee_transform_8,  ee_transform_10, ee_transform_l7,ee_transform_l8, ee_transform_l9, ee_transform_l10, ee_transform_l11, ee_transform_l12 ]

            # transforms = [ee_transform_0, ee_transform_1, ee_transform_2, ee_transform_3, ee_transform_4, ee_transform_5, ee_transform_6, ee_transform_7, ee_transform_8, ee_transform_9,ee_transform_10, ee_transform_11
            # , ee_transform_l7, ee_transform_l8, ee_transform_l9, ee_transform_l10, ee_transform_l11, ee_transform_l12 ]
            draw_transforms(scene, [env_idx], transforms)


        draw_contacts(scene, scene.env_idxs)

    
    no_contact = True
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

        # print(f" vertex {vertex_pos} ")    
        return vertex_pos

    def get_stiffness():
        coeffecients = np.zeros((2, tree.num_joints)) #K stiffness, d damping
        stiff_k = 600
        damping = 400
        coeffecients[0,:] = np.array( [stiff_k] * tree.num_joints)
        coeffecients[1,:] = np.array( [damping] * tree.num_joints)
        # for i in range(tree.num_joints):
        #     print(f" stiffness {i} ")  

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

        
    
    
    tree_tf1 = tree.get_link_transform(0, tree_name, 'link1')
    tree_tf2 = tree.get_link_transform(0, tree_name, 'link2')
    tree_tf3 = tree.get_link_transform(0, tree_name, 'link3')
    tree_tf4 = tree.get_link_transform(0, tree_name, 'link4')

    # tree_tf5 = tree.get_link_transform(0, tree_name, 'link5')
    # tree_tf6 = tree.get_link_transform(0, tree_name, 'link6')
    tree_tf7 = tree.get_link_transform(0, tree_name, 'link7')
    # tree_tf8 = tree.get_link_transform(0, tree_name, 'link8')
    tree_tf9 = tree.get_link_transform(0, tree_name, 'link9')
    # tree_tf10 = tree.get_link_transform(0, tree_name, 'link10')
    tree_tf11 = tree.get_link_transform(0, tree_name, 'link11')
    # tree_tf12 = tree.get_link_transform(0, tree_name, 'link12')

    tree_tf7L = tree.get_link_transform(0, tree_name, 'link7_leaf')
    tree_tf8L = tree.get_link_transform(0, tree_name, 'link8_leaf')
    tree_tf9L = tree.get_link_transform(0, tree_name, 'link9_leaf')
    tree_tf10L = tree.get_link_transform(0, tree_name, 'link10_leaf')
    tree_tf11L = tree.get_link_transform(0, tree_name, 'link11_leaf')
    tree_tf12L = tree.get_link_transform(0, tree_name, 'link12_leaf')

    global contact_transform
    contact_transform = tree_tf1

    tree_location_list = [tree_tf1, tree_tf2, tree_tf3, tree_tf4,  tree_tf7,  tree_tf9, tree_tf11, tree_tf7L, tree_tf8L , tree_tf9L, tree_tf10L, tree_tf11L, tree_tf12L]

    # tree_location_list = [tree_tf1, tree_tf2, tree_tf3, tree_tf4, tree_tf5, tree_tf6, tree_tf7, tree_tf8, tree_tf9, tree_tf10, tree_tf11, tree_tf12
    # , tree_tf7L, tree_tf8L , tree_tf9L, tree_tf10L, tree_tf11L, tree_tf12L]
    
    loc_tree = tree_tf3.p
    random_index = 1
    
    def policy(scene, env_idx, t_step, t_sim):
        global vertex_init_pos, no_contact, force, loc_tree, vertex_final_pos, force_applied, random_index, contact_transform
  
        
        pass

    scene.run(policy=policy, custom_draws=custom_draws)