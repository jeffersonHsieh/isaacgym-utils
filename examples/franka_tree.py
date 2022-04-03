import argparse

import numpy as np
from autolab_core import YamlConfig, RigidTransform

from isaacgym import gymapi
from isaacgym_utils.scene import GymScene
from isaacgym_utils.assets import GymFranka, GymBoxAsset
from isaacgym_utils.assets import GymTree
from isaacgym_utils.camera import GymCamera
from isaacgym_utils.math_utils import RigidTransform_to_transform, np_to_vec3, vec3_to_np
from isaacgym_utils.policy import GraspBlockPolicy, MoveBlockPolicy
from isaacgym_utils.draw import draw_transforms, draw_contacts, draw_camera

import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='cfg/franka_tree.yaml')
    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)

    scene = GymScene(cfg['scene'])

    table = GymBoxAsset(scene, **cfg['table']['dims'], 
                        shape_props=cfg['table']['shape_props'], 
                        asset_options=cfg['table']['asset_options']
                        )
    franka = GymFranka(cfg['franka'], scene, actuation_mode='torques')
    tree = GymTree(cfg['tree'], scene)
    block = GymBoxAsset(scene, **cfg['block']['dims'], 
                        shape_props=cfg['block']['shape_props'], 
                        rb_props=cfg['block']['rb_props'],
                        asset_options=cfg['block']['asset_options']
                        )

    table_transform = gymapi.Transform(p=gymapi.Vec3(cfg['table']['dims']['sx']/3, 0, cfg['table']['dims']['sz']/2))
    franka_transform = gymapi.Transform(p=gymapi.Vec3(0, 0, cfg['table']['dims']['sz'] + 0.01))

    tree_transform = gymapi.Transform(p=gymapi.Vec3(1, 1, 0))
    

    table_name, franka_name, block_name, tree_name = 'table', 'franka', 'block', 'tree'


    cam = GymCamera(scene, cam_props=cfg['camera'])
    cam_offset_transform = RigidTransform_to_transform(RigidTransform(
        rotation=RigidTransform.z_axis_rotation(np.deg2rad(90)) @ RigidTransform.x_axis_rotation(np.deg2rad(1)),
        translation=np.array([-0.083270, -0.046490, 0])
    ))
    cam_name = 'hand_cam0'

    def setup(scene, _):
        scene.add_asset(table_name, table, table_transform)
        scene.add_asset(block_name, block, gymapi.Transform()) # we'll sample block poses later

        scene.add_asset(franka_name, franka, franka_transform, collision_filter=1) # avoid self-collisions

        scene.add_asset(tree_name, tree, tree_transform, collision_filter=1) # avoid self-collisions

 
        scene.attach_camera(cam_name, cam, franka_name, 'panda_hand', offset_transform=cam_offset_transform)
    scene.setup_all_envs(setup)    

    def custom_draws(scene):
        for env_idx in scene.env_idxs:
            ee_transform = franka.get_ee_transform(env_idx, franka_name)
            ee_transform_8 = franka.get_ee_transform_MARK(env_idx, franka_name, 'panda_hand')
            ee_transform_0 = franka.get_ee_transform_MARK(env_idx, franka_name, 'panda_link0')
            ee_transform_1 = franka.get_ee_transform_MARK(env_idx, franka_name, 'panda_link1')
            ee_transform_2 = franka.get_ee_transform_MARK(env_idx, franka_name, 'panda_link2')
            ee_transform_3 = franka.get_ee_transform_MARK(env_idx, franka_name, 'panda_link3')
            ee_transform_4 = franka.get_ee_transform_MARK(env_idx, franka_name, 'panda_link4')
            
            tree_tf1 = tree.get_link_transform(env_idx, tree_name, 'link1')
            tree_tf2 = tree.get_link_transform(env_idx, tree_name, 'link2')
            tree_tf3 = tree.get_link_transform(env_idx, tree_name, 'leaf1')

            transforms = [ee_transform, ee_transform_0, ee_transform_1, ee_transform_2, ee_transform_3, ee_transform_4, ee_transform_8, tree_tf1, tree_tf2, tree_tf3]
            draw_transforms(scene, [env_idx], transforms)
            cam_transform = cam.get_transform(env_idx, cam_name)
            draw_camera(scene, [env_idx], cam_transform, length=0.04)
        draw_contacts(scene, scene.env_idxs)



    # policy = GraspBlockPolicy(franka, franka_name, block, block_name)
    policy = MoveBlockPolicy(franka, franka_name, block, block_name)


    while True:
        # sample block poses
        block_transforms = [gymapi.Transform(p=gymapi.Vec3(
            (np.random.rand()*2 - 1) * 0.1 + 0.5, 
            (np.random.rand()*2 - 1) * 0.2,
            cfg['table']['dims']['sz'] + cfg['block']['dims']['sz'] / 2 + 0.1
        )) for _ in range(scene.n_envs)]

        # set block poses
        for env_idx in scene.env_idxs:
            block.set_rb_transforms(env_idx, block_name, [block_transforms[env_idx]])

        print(f"resetting policy")
        policy.reset()
        print(f"running scene again")
        scene.run(time_horizon=policy.time_horizon, policy=policy, custom_draws=custom_draws)
        # scene.run(policy=policy, custom_draws=custom_draws)