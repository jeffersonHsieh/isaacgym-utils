import argparse
from numba.core.types.scalars import EnumMember

import numpy as np
from numpy import save 
from autolab_core import YamlConfig, RigidTransform
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import itertools

from scipy.spatial.transform import Rotation


from isaacgym import gymapi
from isaacgym_utils.scene import GymScene
from isaacgym_utils.assets import GymFranka, GymBoxAsset
from isaacgym_utils.assets import GymTree
from isaacgym_utils.camera import GymCamera
from isaacgym_utils.math_utils import RigidTransform_to_transform, np_to_vec3, vec3_to_np, quat_to_np, rot_to_quat
from isaacgym_utils.policy import GraspBlockPolicy, MoveBlockPolicy, GraspTreePolicy, RRTTreeFollowingPolicy
from isaacgym_utils.draw import draw_transforms, draw_contacts, draw_camera, draw_spheres
import pdb
import sys

from collections import OrderedDict

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

def get_grabbable_tree_links():
    # get branch link indices that can be used for interacting with
    grabbable_link_indices = []
    grabbable_link_poses = []

    idx = 0
    for link_name in tree.link_names:
        if not "base" in link_name and not "tip" in link_name: # Exclude base from being a push option
            grabbable_link_indices.append(idx)
        idx += 1
    # print(f"size of grabbable_link_indices {len(grabbable_link_indices)} ")
    # print(f"grabbable_link_indices {grabbable_link_indices} ")

    grabbable_link_poses = get_link_poses()[:,grabbable_link_indices]
    # print(f"grabbable_link_poses {grabbable_link_poses} ")
    # print(f" size of grabbable_link_poses {grabbable_link_poses.shape} ")

    return grabbable_link_indices, grabbable_link_poses


# ====================================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='../cfg/franka_tree.yaml')
    # parser.add_argument('--cfg', '-c', type=str, default='../cfg/franka_tree_force_ocrl.yaml')

    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)


    scene = GymScene(cfg['scene'])


    franka = GymFranka(cfg['franka'], scene, actuation_mode='joints')
    tree = GymTree(cfg['tree'], scene, actuation_mode='joints')

    block = GymBoxAsset(scene, **cfg['block']['dims'], 
                        shape_props=cfg['block']['shape_props'], 
                        rb_props=cfg['block']['rb_props'],
                        asset_options=cfg['block']['asset_options']
                        )

    franka_transform = gymapi.Transform(p=gymapi.Vec3(-0.5, 0, 0))
    tree_transform = gymapi.Transform(p=gymapi.Vec3(0, 0, 0), r=gymapi.Quat(0, 0, 0.707, 0.707))
    block_transform = gymapi.Transform(p=gymapi.Vec3(0, 0, 0.1))
    franka.set_base_offset([-0.5, 0, 0])

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

        scene.add_asset(franka_name, franka, franka_transform, collision_filter=0) # avoid self-collisions

        scene.add_asset(tree_name, tree, tree_transform, collision_filter=1) # avoid self-collisions
        # scene.add_asset(block_name, block, block_transform, collision_filter=1, collision_group=2)

    scene.setup_all_envs(setup)    

    
    #no_contact = True
    #force = np_to_vec3([0, 0, 0])

    policy = RRTTreeFollowingPolicy(franka, franka_name, tree, tree_name, actuator_mode='joints')
    # change to control policy

    # home_joint = policy._franka.get_joints(0, policy._franka_name)[:-2]
    # joint_ranges = np.array([
    #     2.9671*2,
    #     1.8362*2,
    #     2.9671*2,
    #     0.0870+3,
    #     2.9671*2,
    #     3+0.0873,
    #     2.9671*2
    # ])
    # rng = np.random.default_rng(42)
    # offsets_ratio = rng.random([20, 7])*0.3-0.015
    # offsets = np.multiply(offsets_ratio, joint_ranges)

    # rrt_success = np.array([
    #     [-0.2589816,  -0.26682076,  0.69985781, -2.09838354, -0.54362249,   1.5399737, -0.02674972],
    #     [-0.64133424, -1.11601532, -0.87702325, -2.09047383,  0.29347739,   1.76081849, 1.28516885],
    #     [-0.0731406,  -0.70966463, -0.641255,   -2.7131782,   0.29980106,  1.54402597, 0.90153542],
    #     [-0.34544766, -0.6981204,  -0.57542849, -2.02593401,  0.46023198,  1.77406077, 0.66450614],
    #     [-0.30289128, -1.177033 ,  -0.70604583, -2.27502686, -0.58643016 , 1.96453837, 0.92970809]
    # ])
    # intermediate_joints = np.array([
    #     [0.2435, -0.6, 0.3209, -2.0111, -0.3499, 1.4845, 0.3800, 64, 133],
    #     [0.1413, -0.7518, -0.3157, -1.9311, -0.0187, 1.4751, 0.9924, 133, 62],
    #     [-0.3347, -0.8270, -0.3199, -2.1337, 0.0236, 1.4488, 0.9158, 113, 99],
    #     [-0.0609, -0.7706, -0.3396, -1.9461, 0.1367, 1.5599, 0.8180, 107, 101],
    #     [-0.0609, -0.7706, -0.3396, -1.9461, 0.1367, 1.5599, 0.8180, 103, 99]
    # ])
    # rrt_successes = []
    # naive_success = np.load('naive_success.npy')
    # # for i, offset in enumerate(offsets):
    # # for i in range(len(rrt_success)):
    # for i, naive_success_ in enumerate(naive_success):
    for i in range(1):
        print('-'*50)
        print(f'exp {i}')
        # start_joint = home_joint + offset
        # start_joint = both_fail + offset
        # start_joint = rrt_success[i]
        # start_joint = naive_success_[0]
        # target_joint = np.array([0.32009414, -0.49480677, -0.18753628, -1.894689, -0.09001054, 1.4097925, 0.9255461])
        # policy._franka.set_joints(0, policy._franka_name, np.concatenate([start_joint, np.ones(2)*0.104]))
        # policy.actuator_mode = 'attractor'

        # plan_1 = np.linspace(start_joint, intermediate_joints[i, :-2], int(intermediate_joints[i, -2]))
        # plan_2 = np.linspace(intermediate_joints[i, :-2], target_joint, int(intermediate_joints[i, -1]))
        # plan = np.concatenate([plan_1, plan_2])

        # plan = np.linspace(start_joint, target_joint, 200)
        # policy._plan = plan


        policy.reset()
        scene.run(time_horizon=policy.time_horizon, policy=policy)
        #rt_successes.append(policy.plan)
        # breakpoint()

    # rrt_successes = np.array(rrt_successes)
    # np.save(f'rrt_success0.npy', rrt_successes)

    