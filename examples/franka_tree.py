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
from isaacgym_utils.draw import draw_transforms, draw_contacts, draw_camera

import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--cfg', '-c', type=str, default='cfg/franka_tree.yaml')
    parser.add_argument('--cfg', '-c', type=str, default='cfg/franka_tree_force.yaml')
    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)

 

    scene = GymScene(cfg['scene'])


    # franka = GymFranka(cfg['franka'], scene, actuation_mode='torques')
    tree = GymTree(cfg['tree'], scene, actuation_mode='joints')

    # block = GymBoxAsset(scene, **cfg['block']['dims'],  shape_props=cfg['block']['shape_props'])


    # franka_transform = gymapi.Transform(p=gymapi.Vec3(1, 1, 0))
    tree_transform = gymapi.Transform(p=gymapi.Vec3(0, 0, 0))

    franka_name, tree_name, block_name = 'franka', 'tree', 'block'

    current_iteration = 0
    num_iteration = 10
    force_magnitude = 50
    push_toggle = True
    
    global vertex_init_pos, vertex_final_pos, force_applied
    vertex_init_pos = np.zeros((7,tree.num_links)) #x,y,z,qx,qy,qz,qw
    vertex_final_pos = np.zeros((7,tree.num_links)) #x,y,z,qx,qy,qz,qw
    force_applied = np.zeros((3,tree.num_links)) #fx,fy,fz     

    def setup(scene, _):

        # scene.add_asset(franka_name, franka, franka_transform, collision_filter=1) # avoid self-collisions

        scene.add_asset(tree_name, tree, tree_transform, collision_filter=1) # avoid self-collisions
        # scene.add_asset('block', block, gymapi.Transform(p=gymapi.Vec3(-1, -1, cfg['block']['dims']['sz']/2)) )

    scene.setup_all_envs(setup)    


    def custom_draws(scene):
        for env_idx in scene.env_idxs:

            ee_transform_0 = tree.get_ee_transform_MARK(env_idx, tree_name, 'link1')
            ee_transform_1 = tree.get_ee_transform_MARK(env_idx, tree_name, 'link2')
            ee_transform_2 = tree.get_ee_transform_MARK(env_idx, tree_name, 'link3')

            transforms = [ee_transform_0, ee_transform_1, ee_transform_2]
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
        stiff_k = 30
        damping = 20
        coeffecients[0,:] = np.array( [stiff_k] * tree.num_joints)
        coeffecients[1,:] = np.array( [damping] * tree.num_joints)
        # for i in range(tree.num_joints):
        #     print(f" stiffness {i} ")  

        return coeffecients


    def set_force(force, index):
        force_applied_ = np.zeros((3,tree.num_links))
        force_applied_[:,index] = force

        return force_applied_ 



    def save_data():
        print(f" ********* saving data ********* ")

        edge_def = [(1,2), (2,3)]   
        coeff_stiff_damp = get_stiffness()

        print(f"vertex_init_pos {vertex_init_pos}")
        print(f"vertex_final_pos {vertex_final_pos}")
        print(f"coeff_stiff_damp {coeff_stiff_damp} ")
        print(f"edge_def {edge_def}")
        print(f"force_applied {force_applied}")

        save('X_vertex_init_pose', vertex_init_pos )
        save('X_coeff_stiff_damp', coeff_stiff_damp )
        save('X_edge_def', edge_def )
        save('X_force_applied', force_applied )
        save('Y_vertex_final_pos', vertex_final_pos )



    def policy(scene, env_idx, t_step, t_sim):
        global vertex_init_pos, no_contact, force, loc_tree, vertex_final_pos, force_applied
  
        # #get pose 
        tree_tf = tree.get_link_transform(0, tree_name, 'link3')
        # print(f"t_sim: {t_sim}, tree_pos: {tree_tf.p},  tree rot: {tree_tf.r}, tree.joint_names: {tree.joint_names}")

        ten_sec_interval = t_sim%10

        

        # #create random force
        if t_sim > 5 and t_sim < 10:
            if no_contact == True:
                print(f"===== making contact ========")
                vertex_init_pos = get_link_poses()
                no_contact = False
                force = np_to_vec3([force_magnitude, 0, 0])
                force_applied = set_force(vec3_to_np(force), 0)
                # force = np_to_vec3([np.random.rand()*force_magnitude, np.random.rand()*force_magnitude, np.random.rand()*force_magnitude])
                loc_tree = tree_tf.p

            tree.apply_force(env_idx, tree_name, 'link3', force, loc_tree)

        if t_sim > 10:
            if no_contact == False:
                print(f"===== breaking contact ========")
                vertex_final_pos = get_link_poses()
                save_data()
                no_contact = True
                force = np_to_vec3([0, 0, 0])
                 # # force = np_to_vec3([np.random.rand()*force_magnitude, np.random.rand()*force_magnitude, np.random.rand()*force_magnitude])
                loc_tree = tree_tf.p
            
            tree.apply_force(env_idx, tree_name, 'link3', force, loc_tree)


        # get delta pose

        # release tree

    scene.run(policy=policy, custom_draws=custom_draws)