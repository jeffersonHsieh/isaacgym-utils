import argparse

import numpy as np
from autolab_core import YamlConfig, RigidTransform

from isaacgym import gymapi
from isaacgym_utils.scene import GymScene
from isaacgym_utils.assets import GymFranka, GymBoxAsset
from isaacgym_utils.camera import GymCamera
from isaacgym_utils.math_utils import RigidTransform_to_transform
from isaacgym_utils.policy import GraspBlockPolicy, RRTGraspBlockPolicy
from isaacgym_utils.draw import draw_transforms, draw_contacts, draw_camera

import pdb

import quaternion

import matplotlib.pyplot as plt

def draw_box(ax, fbox_vertices):
    lines_idx_list = [[0, 1], [1, 3], [2, 3], [0, 2], [4, 5], [5, 7], [6, 7], [4, 6], [0, 4], [1, 5], [2, 6], [3, 7]]
    handle = ax.plot(fbox_vertices[lines_idx_list[0], 0], fbox_vertices[lines_idx_list[0], 1], fbox_vertices[lines_idx_list[0], 2])
    color = handle[0]._color
    
    for idxs in lines_idx_list:
        ax.plot(fbox_vertices[idxs, 0], fbox_vertices[idxs, 1], fbox_vertices[idxs, 2], color = color)
    # color = handle[0]._color
    # print(color)
    # ax.plot(fbox_vertices[2:4, 0], fbox_vertices[2:4, 1], fbox_vertices[2:4, 2], color=color)
    # ax.plot(fbox_vertices[4:6, 0], fbox_vertices[4:6, 1], fbox_vertices[4:6, 2])
    # ax.plot(fbox_vertices[:2, 0], fbox_vertices[:2, 1], fbox_vertices[:2, 2])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='/home/jan-malte/OCRL_project/isaacgym-utils/cfg/franka_pick_block.yaml')
    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)

    scene = GymScene(cfg['scene'])

    table = GymBoxAsset(scene, **cfg['table']['dims'], 
                        shape_props=cfg['table']['shape_props'], 
                        asset_options=cfg['table']['asset_options']
                        )
    franka = GymFranka(cfg['franka'], scene, actuation_mode='torques')
    wall = GymBoxAsset(scene, **cfg['wall']['dims'], 
                        shape_props=cfg['wall']['shape_props'], 
                        rb_props=cfg['wall']['rb_props'],
                        asset_options=cfg['wall']['asset_options']
                        )
    block = GymBoxAsset(scene, **cfg['block']['dims'], 
                        shape_props=cfg['block']['shape_props'], 
                        rb_props=cfg['block']['rb_props'],
                        asset_options=cfg['block']['asset_options']
                        )
    table_transform = gymapi.Transform(p=gymapi.Vec3(cfg['table']['dims']['sx']/3, 0, cfg['table']['dims']['sz']/2))
    franka_transform = gymapi.Transform(p=gymapi.Vec3(0, 0, cfg['table']['dims']['sz'] + 0.01))
    wall_transform = gymapi.Transform(p=gymapi.Vec3(0.5, 0, cfg['wall']['dims']['sz']/2 + cfg['table']['dims']['sz'] + 0.01))
    block_transform = gymapi.Transform(p=gymapi.Vec3(1, 0, cfg['table']['dims']['sz'] + 0.01))

    franka.set_base_offset([0, 0, cfg['table']['dims']['sz'] + 0.01])
    # franka.set_base_offset([0, 0, 0, 0, 0, 0, 0])
    franka.precompute_self_collision_box_data()

    # init_joint = franka.INIT_JOINTS[:-2]
    # joints_target = franka.inverse_kinematics(block_pos, joints_target)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    init_joint = franka.INIT_JOINTS[:-2]
    # init_joint = np.array([0.0, 5e-1, 0.0, -2.3, 0.0, 2.8, 7.8e-1])
    ee = franka.ee(init_joint)
    ee_pos = ee[:3]
    # ee_q = quaternion.from_euler_angles(*ee[-3:])
    # print(f'ee pos: {ee_pos}')
    # print(f'ee q: {ee_q}')
    # print(ee)
    # init_joint = np.array([0, 0, 0, 0, 0, 0, 0])
    franka_box_poses = franka.get_collision_boxes_poses(init_joint)
    box_colors = []
    
    for i, franka_box_pose in enumerate(franka_box_poses):
        fbox_pos = franka_box_pose[:3, 3]
        # print(fbox_pos)
        fbox_axes = franka_box_pose[:3, :3]

        fbox_vertex_offsets = franka._collision_box_vertices_offset[i]
        fbox_vertices = fbox_vertex_offsets.dot(fbox_axes.T) + fbox_pos
        # print(fbox_vertices)
        ax.scatter(fbox_vertices[:, 0], fbox_vertices[:, 1], fbox_vertices[:, 2])
        draw_box(ax, fbox_vertices)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)

    plt.show()

    

    # table_name, franka_name, block_name, wall_name = 'table', 'franka', 'block', 'wall'


    # cam = GymCamera(scene, cam_props=cfg['camera'])
    # cam_offset_transform = RigidTransform_to_transform(RigidTransform(
    #     rotation=RigidTransform.z_axis_rotation(np.deg2rad(90)) @ RigidTransform.x_axis_rotation(np.deg2rad(1)),
    #     translation=np.array([-0.083270, -0.046490, 0])
    # ))
    # cam_name = 'hand_cam0'

    # def setup(scene, _):
    #     scene.add_asset(table_name, table, table_transform)
    #     scene.add_asset(franka_name, franka, franka_transform, collision_filter=1) # avoid self-collisions
    #     # scene.add_asset(block_name, block, gymapi.Transform()) # we'll sample block poses later
    #     scene.add_asset(wall_name, wall, wall_transform)
    #     scene.add_asset(block_name, block, block_transform)
    #     scene.attach_camera(cam_name, cam, franka_name, 'panda_hand', offset_transform=cam_offset_transform)
    # scene.setup_all_envs(setup)    

    # # print(scene.env_idxs)
    # print(wall.sx, wall.sy, wall.sz)
    # print(wall.get_rb_poses_as_np_array(0, wall_name))
    # a = wall.get_rb_poses_as_np_array(0, wall_name)[0, :3]
    # b = np.concatenate((a, np.array([0, 0, 0])))
    # print(b)
    # b = quaternion.as_euler_angles(np.array([0.999, 0, 0, 0]))
    # print(b)
    # print(wall.get_rb_poses_as_np_array(0, wall_name).shape)
