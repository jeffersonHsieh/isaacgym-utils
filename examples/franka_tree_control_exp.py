import argparse

import numpy as np
from pathlib import Path
from autolab_core import YamlConfig, RigidTransform

from isaacgym import gymapi
from isaacgym_utils.scene import GymScene
from isaacgym_utils.assets import GymFranka, GymBoxAsset
from isaacgym_utils.assets import GymTree
from isaacgym_utils.math_utils import RigidTransform_to_transform, np_to_vec3, vec3_to_np, quat_to_np, rot_to_quat
from isaacgym_utils.policy import GraspBlockPolicy, RRTGraspBlockPolicy,FrankaJointWayPointPolicy,RRTFollowingPolicy
from isaacgym_utils.draw import draw_transforms, draw_contacts, draw_camera


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='../cfg/franka_tree_force_ocrl.yaml')
    parser.add_argument('--kp',default=10,type=float)
    parser.add_argument('--kd',default=1,type=float)
    parser.add_argument('--log_file',default=None,type=Path)
    parser.add_argument('--traj',default="../exp/traj/rrt/rrt_traj_1.npy",type=Path)
    parser.add_argument('--time_horizon',default=1000,type=int)

    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)


    scene = GymScene(cfg['scene'])


    franka = GymFranka(cfg['franka'], scene, actuation_mode='torques')
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


    
    global vertex_init_pos, vertex_final_pos, force_applied 

    vertex_init_pos = np.zeros((7,tree.num_links)) #x,y,z,qx,qy,qz,qw
    vertex_final_pos = np.zeros((7,tree.num_links)) #x,y,z,qx,qy,qz,qw
    force_applied = np.zeros((3,tree.num_links)) #fx,fy,fz     

    

    def setup(scene, _):

        scene.add_asset(franka_name, franka, franka_transform, collision_filter=0) # avoid self-collisions

        scene.add_asset(tree_name, tree, tree_transform, collision_filter=1) # avoid self-collisions
        # scene.add_asset(block_name, block, block_transform, collision_filter=1, collision_group=2)

    scene.setup_all_envs(setup)    

    # policy = GraspBlockPolicy(franka, franka_name, block, block_name)
    # policy = RRTGraspBlockPolicy(franka, franka_name, block, block_name, wall, wall_name)
    P_GAIN = np.diag([args.kp]*7)
    # P_GAIN = np.diag([0.3,0.3,0.3,1,1,1,1])
    D_GAIN = np.diag([args.kd]*7)

    traj = np.load(str(args.traj))

    policy = FrankaJointWayPointPolicy(franka,franka_name,init_joint_pos=traj[0],goal_joint_pos = traj[-1],
                                       traj=traj,P_gain=P_GAIN,D_gain=D_GAIN,T=max(len(traj),1))
    # policy = RRTFollowingPolicy(franka,franka_name,traj=traj)
    start_joint = traj[0]
    policy._franka.set_joints(0, policy._franka_name, np.concatenate([start_joint, np.ones(2)*0.104]))
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
        scene.run(time_horizon=args.time_horizon, policy=policy)
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