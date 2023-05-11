import argparse

import numpy as np
from pathlib import Path
from autolab_core import YamlConfig, RigidTransform

from isaacgym import gymapi
from isaacgym_utils.scene import GymScene
from isaacgym_utils.assets import GymFranka, GymBoxAsset
from isaacgym_utils.camera import GymCamera
from isaacgym_utils.math_utils import RigidTransform_to_transform
from isaacgym_utils.policy import GraspBlockPolicy, RRTGraspBlockPolicy,FrankaJointWayPointPolicy,RRTFollowingPolicy
from isaacgym_utils.draw import draw_transforms, draw_contacts, draw_camera


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='../cfg/franka_pick_block.yaml')
    parser.add_argument('--kp',default=10,type=float)
    parser.add_argument('--kd',default=1,type=float)
    parser.add_argument('--log_file',default=None,type=Path)
    parser.add_argument('--traj',default="plan-correct.npy",type=Path)
    parser.add_argument('--time_horizon',default=1000,type=int)
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
    wall_transform = gymapi.Transform(p=gymapi.Vec3(0.2, 0, cfg['wall']['dims']['sz']/2 + cfg['table']['dims']['sz'] + 0.1))
    block_transform = gymapi.Transform(p=gymapi.Vec3(0.5, 0, cfg['table']['dims']['sz'] + cfg['block']['dims']['sz'] / 2 + 0.1))
    
    # change the collision box in franka robot
    franka.set_base_offset([0, 0, cfg['table']['dims']['sz'] + 0.01])
    # franka.precompute_self_collision_box_data()

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

    def custom_draws(scene):
        for env_idx in scene.env_idxs:
            ee_transform = franka.get_ee_transform(env_idx, franka_name)
            ee_transform_8 = franka.get_ee_transform_MARK(env_idx, franka_name, 'panda_hand')
            ee_transform_0 = franka.get_ee_transform_MARK(env_idx, franka_name, 'panda_link0')
            ee_transform_1 = franka.get_ee_transform_MARK(env_idx, franka_name, 'panda_link1')
            ee_transform_2 = franka.get_ee_transform_MARK(env_idx, franka_name, 'panda_link2')
            ee_transform_3 = franka.get_ee_transform_MARK(env_idx, franka_name, 'panda_link3')
            ee_transform_4 = franka.get_ee_transform_MARK(env_idx, franka_name, 'panda_link4')
            

            transforms = [ee_transform, ee_transform_0, ee_transform_1, ee_transform_2, ee_transform_3, ee_transform_4, ee_transform_8]
            draw_transforms(scene, [env_idx], transforms)
            cam_transform = cam.get_transform(env_idx, cam_name)
            draw_camera(scene, [env_idx], cam_transform, length=0.04)
        draw_contacts(scene, scene.env_idxs)

    # policy = GraspBlockPolicy(franka, franka_name, block, block_name)
    # policy = RRTGraspBlockPolicy(franka, franka_name, block, block_name, wall, wall_name)
    P_GAIN = np.diag([args.kp]*7)
    # P_GAIN = np.diag([0.3,0.3,0.3,1,1,1,1])
    D_GAIN = np.diag([args.kd]*7)
    traj = np.load(str(args.traj))
    policy = FrankaJointWayPointPolicy(franka,franka_name,init_joint_pos=traj[0],goal_joint_pos = traj[-1],
                                       traj=traj,P_gain=P_GAIN,D_gain=D_GAIN,T=max(len(traj),1))
    # policy = RRTFollowingPolicy(franka,franka_name,traj=traj)
    
    # import pdb;pdb.set_trace()
    for i in range(1):
        # sample block poses
        # block_transforms = [gymapi.Transform(p=gymapi.Vec3(
        #     0.8, 
        #     0,
        #     cfg['table']['dims']['sz'] + cfg['block']['dims']['sz'] / 2 + 0.1
        # )) for _ in range(scene.n_envs)]

        # set block poses
        # for env_idx in scene.env_idxs:
        #     block.set_rb_transforms(env_idx, block_name, [block_transforms[env_idx]])

        print(f"resetting policy")
        policy.reset()
        print(f"running scene again")
        scene.run(time_horizon=args.time_horizon, policy=policy, custom_draws=custom_draws)
    print("Goal joint state (degs)", np.degrees(traj[-1]))
    print("Goal ee pos", franka.ee(traj[-1]))
    print("Actual joint states (degs)",np.degrees(policy.actual_traj[-1]))
    print("Actual ee pos", franka.ee(policy.actual_traj[-1]))
    ee_error = np.linalg.norm(franka.ee(policy.actual_traj[-1])[:3] - franka.ee(traj[-1])[:3])
    print("ee error (2-norm)", ee_error)
    print("joint error (inf norm degree)", np.linalg.norm(policy.actual_traj[-1] - traj[-1],np.inf))
    print("joint error (-inf norm degree)", np.linalg.norm(policy.actual_traj[-1] - traj[-1],-np.inf))

    if args.log_file!=None:
        with args.log_file.open("a+") as f:
            f.write(f"{args.traj.name},{args.kp},{args.kd},{ee_error}\n")
    
    assert np.linalg.norm(franka.ee(policy.actual_traj[-1])[:3] - franka.ee(traj[-1])[:3]) < 0.01