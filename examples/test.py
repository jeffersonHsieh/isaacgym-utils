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
    

    table_name, franka_name, block_name, wall_name = 'table', 'franka', 'block', 'wall'


    cam = GymCamera(scene, cam_props=cfg['camera'])
    cam_offset_transform = RigidTransform_to_transform(RigidTransform(
        rotation=RigidTransform.z_axis_rotation(np.deg2rad(90)) @ RigidTransform.x_axis_rotation(np.deg2rad(1)),
        translation=np.array([-0.083270, -0.046490, 0])
    ))
    cam_name = 'hand_cam0'

    def setup(scene, _):
        scene.add_asset(table_name, table, table_transform)
        scene.add_asset(franka_name, franka, franka_transform, collision_filter=1) # avoid self-collisions
        # scene.add_asset(block_name, block, gymapi.Transform()) # we'll sample block poses later
        scene.add_asset(wall_name, wall, wall_transform)
        scene.add_asset(block_name, block, block_transform)
        scene.attach_camera(cam_name, cam, franka_name, 'panda_hand', offset_transform=cam_offset_transform)
    scene.setup_all_envs(setup)    

    # print(scene.env_idxs)
    print(wall.sx, wall.sy, wall.sz)
    print(wall.get_rb_poses_as_np_array(0, wall_name))
    a = wall.get_rb_poses_as_np_array(0, wall_name)[0, :3]
    b = np.concatenate((a, np.array([0, 0, 0])))
    print(b)
    # b = quaternion.as_euler_angles(np.array([0.999, 0, 0, 0]))
    # print(b)
    # print(wall.get_rb_poses_as_np_array(0, wall_name).shape)
