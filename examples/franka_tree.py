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
    # parser.add_argument('--cfg', '-c', type=str, default='cfg/franka_tree.yaml')
    parser.add_argument('--cfg', '-c', type=str, default='cfg/franka_tree_force.yaml')
    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)

    initial_push = True

    scene = GymScene(cfg['scene'])


    franka = GymFranka(cfg['franka'], scene, actuation_mode='torques')
    tree = GymTree(cfg['tree'], scene, actuation_mode='joints')

    block = GymBoxAsset(scene, **cfg['block']['dims'],  shape_props=cfg['block']['shape_props'])


    franka_transform = gymapi.Transform(p=gymapi.Vec3(1, 1, 0))
    tree_transform = gymapi.Transform(p=gymapi.Vec3(0, 0, 0))

    franka_name, tree_name, block_name = 'franka', 'tree', 'block'

    

    def setup(scene, _):

        # scene.add_asset(franka_name, franka, franka_transform, collision_filter=1) # avoid self-collisions

        scene.add_asset(tree_name, tree, tree_transform, collision_filter=1) # avoid self-collisions
        scene.add_asset('block', block, gymapi.Transform(p=gymapi.Vec3(-1, -1, cfg['block']['dims']['sz']/2)) )

    scene.setup_all_envs(setup)    


    


    def custom_draws(scene):
        for env_idx in scene.env_idxs:
            # ee_transform = franka.get_ee_transform(env_idx, franka_name)
            # ee_transform_8 = franka.get_ee_transform_MARK(env_idx, franka_name, 'panda_hand')
            # ee_transform_0 = franka.get_ee_transform_MARK(env_idx, franka_name, 'panda_link0')
            # ee_transform_1 = franka.get_ee_transform_MARK(env_idx, franka_name, 'panda_link1')
            # ee_transform_2 = franka.get_ee_transform_MARK(env_idx, franka_name, 'panda_link2')
            # ee_transform_3 = franka.get_ee_transform_MARK(env_idx, franka_name, 'panda_link3')
            # ee_transform_4 = franka.get_ee_transform_MARK(env_idx, franka_name, 'panda_link4')

            ee_transform_0 = tree.get_ee_transform_MARK(env_idx, tree_name, 'link1')
            ee_transform_1 = tree.get_ee_transform_MARK(env_idx, tree_name, 'link2')
            ee_transform_2 = tree.get_ee_transform_MARK(env_idx, tree_name, 'leaf1')



            

            transforms = [ee_transform_0, ee_transform_1, ee_transform_2]
            draw_transforms(scene, [env_idx], transforms)

        draw_contacts(scene, scene.env_idxs)

    def policy(scene, env_idx, t_step, t_sim):

        force = np_to_vec3([10, 0, 0])
        force_block = np_to_vec3([-.5, 0, 0])

        if t_sim > 5:
            print("t_sim {t_sim}")
            force = np_to_vec3([0, 0, 0])
            force_block = np_to_vec3([0, 0, 0])


        tree_tf = tree.get_link_transform(0, tree_name, 'leaf1')


        block_tf = block.get_rb_transforms(0,block_name)[0]

        

        # loc = ee_transform.p
        loc_tree = tree_tf.p
        loc_block = block_tf.p

        print(f"loc_block {loc_block}")

        # print(f"link location:, {loc}")
        # franka.apply_force(env_idx, 'franka', 'panda_link2', force, loc)
        tree.apply_force(env_idx, tree_name, 'link2', force, loc_tree)
        block.apply_force(env_idx, block_name, 'box', force_block, loc_block)



    scene.run(policy=policy, custom_draws=custom_draws)