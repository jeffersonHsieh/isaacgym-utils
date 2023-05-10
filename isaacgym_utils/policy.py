from abc import ABC, abstractmethod
from os import initgroups
import numpy as np
import quaternion

from isaacgym import gymapi
from numpy.core.shape_base import block
from .math_utils import min_jerk, slerp_quat, vec3_to_np, np_to_vec3, np_to_quat,\
                    project_to_line, compute_task_space_impedance_control, quat_to_rpy

from .rrt import RRT


class Policy(ABC):

    def __init__(self):
        self._time_horizon = -1

    @abstractmethod
    def __call__(self, scene, env_idx, t_step, t_sim):
        pass

    def reset(self):
        pass

    @property
    def time_horizon(self):
        return self._time_horizon


class RandomDeltaJointPolicy(Policy):

    def __init__(self, franka, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._franka = franka
        self._name = name

    def __call__(self, scene, env_idx, _, __):
        delta_joints = (np.random.random(self._franka.n_dofs) * 2 - 1) * ([0.05] * 7 + [0.005] * 2)
        self._franka.apply_delta_joint_targets(env_idx, self._name, delta_joints)


class MoveBlockPolicy(Policy):

    def __init__(self, franka, franka_name, block, block_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._franka = franka
        self._franka_name = franka_name
        self._block = block
        self._block_name = block_name

        self._time_horizon = 1000

        self.reset()

    def reset(self):
        self._pre_grasp_transforms = []
        self._grasp_transforms = []
        self._init_ee_transforms = []
        self._ee_waypoint_policies = []

    def __call__(self, scene, env_idx, t_step, t_sim):
        ee_transform = self._franka.get_ee_transform(env_idx, self._franka_name)

        if t_step == 0:
            self._init_ee_transforms.append(ee_transform)
            self._ee_waypoint_policies.append(
                EEImpedanceWaypointPolicy(self._franka, self._franka_name, ee_transform, ee_transform, T=20)
            )

        if t_step == 20:
            block_transform = self._block.get_rb_transforms(env_idx, self._block_name)[0]
            grasp_transform = gymapi.Transform(p=block_transform.p, r=self._init_ee_transforms[env_idx].r)
            pre_grasp_transfrom = gymapi.Transform(p=grasp_transform.p + gymapi.Vec3(0, 0, 0.2), r=grasp_transform.r)

            self._grasp_transforms.append(grasp_transform)
            self._pre_grasp_transforms.append(pre_grasp_transfrom)

            self._ee_waypoint_policies[env_idx] = \
                EEImpedanceWaypointPolicy(
                    self._franka, self._franka_name, ee_transform, self._pre_grasp_transforms[env_idx], T=180
                )
            print(f"------------- move 1 ------------- ")

        if t_step == 200:
            self._ee_waypoint_policies[env_idx] = \
                EEImpedanceWaypointPolicy(
                    self._franka, self._franka_name, self._pre_grasp_transforms[env_idx], self._grasp_transforms[env_idx], T=100
                )
            print(f"------------- move 2 ------------- ")

        if t_step == 300:
            block_transform = self._block.get_rb_transforms(env_idx, self._block_name)[0]
            loc = block_transform.p

            force = np_to_vec3([-np.sin(t_sim), 0, 0])

            self._block.apply_force(env_idx, self._block_name, 'box', force, loc)

            print(f"------------- move 3 ------------- ")


        self._ee_waypoint_policies[env_idx](scene, env_idx, t_step, t_sim)

class RRTTreeFollowingPolicy(Policy):
    def __init__(self, franka, franka_name, tree, tree_name, actuator_mode = 'joints', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._franka = franka
        self._franka_name = franka_name
        self.actuator_mode = actuator_mode
        self._tree = tree
        self._tree_name = tree_name
        self._time_horizon = 1000
        self._plan = None

        self.reset()

    def reset(self):
        self._init_ee_transforms = []
        self._ee_waypoint_policies = []
        self.boxes = []
    
    def is_in_collision(self, joints):
        for box in self.boxes:
            if self._franka.check_box_collision(joints, box):
                return True
        return False

    def __call__(self, scene, env_idx, t_step, t_sim):
        ee_transform = self._franka.get_ee_transform(env_idx, self._franka_name)

        if t_step == 0:
            if self._plan is not None:
                self.plan = self._plan
            else:
                joints_start = self._franka.get_joints(env_idx, self._franka_name)[:-2]
                rrt = RRT(self._franka, self.is_in_collision)

                # self.boxes.append(np.array([0, 0, 0, 0, 0, 0, 0.5, 0.5, 2]))

                # print(f'issac starting pos: {ee_transform.p}, {ee_transform.r.x}')

                # get ee [-1.93109447e-01 -8.18154653e-17  5.93882073e-01 -2.09060298e-16  -4.37113853e-08 -1.11022310e-16]

                target_idx = 2

                targets = self._tree.get_tips_transforms(env_idx, self._tree_name)
                target_pos = vec3_to_np(targets[target_idx].p)
                target_pos_rpy = np.concatenate((target_pos + [0, 0, 0.12], np.array([0, 0, 0])))
                # print(f'target pos: {target_pos_rpy}')

                # joints_target = np.array([ 0.08159304, -0.7904955,  -0.16469808, -1.7779573,   0.6305357,   1.7729826, 0.861042  ])
                joints_target = self._franka.inverse_kinematics(target_pos_rpy, joints_start)
                print(f'joint target: {joints_target}')

                self.plan = rrt.plan(joints_start, joints_target)
                # np.save(f'tree_plan.npy', self.plan)

            self.i = 0
            self._ee_waypoint_policies.append(
                EEImpedanceWaypointPolicy(self._franka, self._franka_name, ee_transform, ee_transform, T=15)
            )

        if t_step > 15 and t_step % 2 == 0:
            # print(self.actuator_mode)
            # if self.actuator_mode is not 'joints':
            #     self._ee_waypoint_policies[env_idx] = EEImpedanceWaypointPolicy(self._franka, self._franka_name, ee_transform, ee_transform, T=15)
            # else:
            try:
                self._ee_waypoint_policies[env_idx] = SetJointPolicy(self._franka, self._franka_name, np.concatenate([self.plan[self.i], np.ones(2)*0.104]), T=15)
                self.i += 1
            except IndexError:
                self._franka.close_grippers(env_idx, self._franka_name)

        self._ee_waypoint_policies[env_idx](scene, env_idx, t_step, t_sim)

class RRTGraspBlockPolicy(Policy):
    def __init__(self, franka, franka_name, block, block_name, wall, wall_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._franka = franka
        self._franka_name = franka_name
        self._block = block
        self._block_name = block_name
        self._wall = wall
        self._wall_name = wall_name

        self._time_horizon = 1000

        self.reset()

    
    def ee_upright_constraint(self, q):
        '''
        TODO: Implement constraint function and its gradient. 
        
        This constraint should enforce the end-effector stays upright.
        Hint: Use the roll and pitch angle in desired_ee_rp. The end-effector is upright in its home state.
        Input:
            q - a joint configuration
        Output:
            err - a non-negative scalar that is 0 when the constraint is satisfied
            grad - a vector of length 6, where the ith element is the derivative of err w.r.t. the ith element of ee
        '''
        desired_ee_rp = self._franka.ee(self._franka.home_joints)[3:5]
        ee = self._franka.ee(q)
        err = np.sum((np.asarray(desired_ee_rp) - np.asarray(ee[3:5]))**2)
        grad = np.asarray([0, 0, 0, 2*(ee[3]-desired_ee_rp[0]), 2*(ee[4]-desired_ee_rp[1]), 0])
        return err, grad

    def reset(self):
        self._pre_grasp_transforms = []
        self._grasp_transforms = []
        self._init_ee_transforms = []
        self._ee_waypoint_policies = []
        self.boxes = []
    
    def is_in_collision(self, joints):
        for box in self.boxes:
            if self._franka.check_box_collision(joints, box):
                return True
        return False

    def __call__(self, scene, env_idx, t_step, t_sim):
        ee_transform = self._franka.get_ee_transform(env_idx, self._franka_name)

        if t_step == 0:
            wall_pos = self._wall.get_rb_poses_as_np_array(env_idx, self._wall_name)
            wall_size = np.array([[self._wall.sx, self._wall.sy, self._wall.sz]])
            self.boxes = np.concatenate((wall_pos, wall_size), axis=1)
            rrt = RRT(self._franka, self.is_in_collision)
            self._init_ee_transforms.append(ee_transform)

            block_pos = self._block.get_rb_poses_as_np_array(env_idx, self._block_name)[0] 
            block_pos_rpy = np.concatenate((block_pos[:3], np.array([np.pi, 0, np.pi/2])))
            print(f'block pos: {block_pos_rpy}')

            joints_start = self._franka.get_joints(env_idx, self._franka_name)[:-2]

            # joints_target = np.array([0, 3.43e-1, 0, -2.235, 0, 2.567, 7.8e-1])
            joints_target = [ 0.3783648,   0.7361812,   0.01813878, -1.8424159,  -1.239651,    1.3556952,  1.8316574 ]
            # joints_target = self._franka.inverse_kinematics(block_pos_rpy, joints_start)

            ee = self._franka.ee(joints_start)
            print(f'start ee: {ee}')
            ee = self._franka.ee(joints_target)
            print(f'target ee: {ee}')

            # constraint = self.ee_upright_constraint
            # self.plan = rrt.plan(joints_start, joints_target, constraint)
            print(f'joints_start: {joints_start}')
            print(f'joints_target: {joints_target}')
            self.plan = rrt.plan(joints_start, joints_target)
            np.save(f'plan.npy', self.plan)
            self.i = 0

            self._ee_waypoint_policies.append(
                EEImpedanceWaypointPolicy(self._franka, self._franka_name, ee_transform, ee_transform, T=15)
            )

        if t_step > 15 and t_step % 2 == 0:
            # self._init_ee_transforms.append(ee_transform)
            try:
                self._ee_waypoint_policies[env_idx] = SetJointPolicy(self._franka, self._franka_name, np.concatenate([self.plan[self.i], np.ones(2)*0.104]), T=15)
                self.i += 1
            except IndexError:
                # self._ee_waypoint_policies[env_idx] = SetJointPolicy(self._franka, self._franka_name, np.concatenate([self.plan[self.i], np.ones(2)*0.104]), T=15)
                self._franka.close_grippers(env_idx, self._franka_name)

        self._ee_waypoint_policies[env_idx](scene, env_idx, t_step, t_sim)

class GraspBlockPolicy(Policy):

    def __init__(self, franka, franka_name, block, block_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._franka = franka
        self._franka_name = franka_name
        self._block = block
        self._block_name = block_name

        self._time_horizon = 1000

        self.reset()

    def reset(self):
        self._pre_grasp_transforms = []
        self._grasp_transforms = []
        self._init_ee_transforms = []
        self._ee_waypoint_policies = []

    def __call__(self, scene, env_idx, t_step, t_sim):
        ee_transform = self._franka.get_ee_transform(env_idx, self._franka_name)

        if t_step == 0:
            self._franka.open_grippers(env_idx, self._franka_name)
            self._init_ee_transforms.append(ee_transform)
            self._ee_waypoint_policies.append(
                EEImpedanceWaypointPolicy(self._franka, self._franka_name, ee_transform, ee_transform, T=20)
            )
        # self._ee_waypoint_policies[env_idx] = EEImpedanceWaypointPolicy(self._franka, self._franka_name, ee_transform, ee_transform, T=20)
        

        # if t_step == 20:
        #     block_transform = self._block.get_rb_transforms(env_idx, self._block_name)[0]
        #     # print(self._init_ee_transforms[env_idx].r)
        #     grasp_transform = gymapi.Transform(p=block_transform.p, r=gymapi.Quat(0.707, 0, 0, 0.707))
        #     pre_grasp_transfrom = gymapi.Transform(p=grasp_transform.p + gymapi.Vec3(0, 0.1, 0), r=grasp_transform.r)

        #     self._grasp_transforms.append(grasp_transform)
        #     self._pre_grasp_transforms.append(pre_grasp_transfrom)

        #     self._ee_waypoint_policies[env_idx] = \
        #         EEImpedanceWaypointPolicy(
        #             self._franka, self._franka_name, ee_transform, self._pre_grasp_transforms[env_idx], T=180
        #         )

        # if t_step == 200:
        #     self._ee_waypoint_policies[env_idx] = \
        #         EEImpedanceWaypointPolicy(
        #             self._franka, self._franka_name, self._pre_grasp_transforms[env_idx], self._grasp_transforms[env_idx], T=100
        #         )

        # if t_step == 300:
        #     self._franka.close_grippers(env_idx, self._franka_name)
        
        # if t_step == 400:
        #     self._ee_waypoint_policies[env_idx] = \
        #         EEImpedanceWaypointPolicy(
        #             self._franka, self._franka_name, self._grasp_transforms[env_idx], self._pre_grasp_transforms[env_idx], T=100
        #         )

        # if t_step == 500:
        #     self._ee_waypoint_policies[env_idx] = \
        #         EEImpedanceWaypointPolicy(
        #             self._franka, self._franka_name, self._pre_grasp_transforms[env_idx], self._grasp_transforms[env_idx], T=100
        #         )

        # if t_step == 600:
        #     self._franka.open_grippers(env_idx, self._franka_name)

        # if t_step == 700:
        #     self._ee_waypoint_policies[env_idx] = \
        #         EEImpedanceWaypointPolicy(
        #             self._franka, self._franka_name, self._grasp_transforms[env_idx], self._pre_grasp_transforms[env_idx], T=100
        #         )

        # if t_step == 800:
        #     self._ee_waypoint_policies[env_idx] = \
        #         EEImpedanceWaypointPolicy(
        #             self._franka, self._franka_name, self._pre_grasp_transforms[env_idx], self._init_ee_transforms[env_idx], T=100
        #         )

        self._ee_waypoint_policies[env_idx](scene, env_idx, t_step, t_sim)


class GraspPointPolicy(Policy):

    def __init__(self, franka, franka_name, grasp_transform, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._franka = franka
        self._franka_name = franka_name
        self._grasp_transform = grasp_transform

        self._time_horizon = 710

        self.reset()

    def reset(self):
        self._pre_grasp_transforms = []
        self._grasp_transforms = []
        self._init_ee_transforms = []

    def __call__(self, scene, env_idx, t_step, _):
        t_step = t_step % self._time_horizon

        if t_step == 0:
            self._init_joints = self._franka.get_joints(env_idx, self._franka_name)
            self._init_rbs = self._franka.get_rb_states(env_idx, self._franka_name)

        if t_step == 20:
            ee_transform = self._franka.get_ee_transform(env_idx, self._franka_name)
            self._init_ee_transforms.append(ee_transform)

            pre_grasp_transfrom = gymapi.Transform(p=self._grasp_transform.p, r=self._grasp_transform.r)
            pre_grasp_transfrom.p.z += 0.2

            self._grasp_transforms.append(self._grasp_transform)
            self._pre_grasp_transforms.append(pre_grasp_transfrom)

            self._franka.set_ee_transform(env_idx, self._franka_name, self._pre_grasp_transforms[env_idx])

        if t_step == 100:
            self._franka.set_ee_transform(env_idx, self._franka_name, self._grasp_transforms[env_idx])

        if t_step == 150:
            self._franka.close_grippers(env_idx, self._franka_name)
        
        if t_step == 250:
            self._franka.set_ee_transform(env_idx, self._franka_name, self._pre_grasp_transforms[env_idx])

        if t_step == 350:
            self._franka.set_ee_transform(env_idx, self._franka_name, self._grasp_transforms[env_idx])

        if t_step == 500:
            self._franka.open_grippers(env_idx, self._franka_name)

        if t_step == 550:
            self._franka.set_ee_transform(env_idx, self._franka_name, self._pre_grasp_transforms[env_idx])

        if t_step == 600:
            self._franka.set_ee_transform(env_idx, self._franka_name, self._init_ee_transforms[env_idx])

        if t_step == 700:
            self._franka.set_joints(env_idx, self._franka_name, self._init_joints)
            self._franka.set_rb_states(env_idx, self._franka_name, self._init_rbs)


class FrankaEEImpedanceController:

    def __init__(self, franka, franka_name):
        self._franka = franka
        self._franka_name = franka_name
        self._elbow_joint = 3

        Kp_0, Kr_0 = 200, 8
        Kp_1, Kr_1 = 200, 5
        self._Ks_0 = np.diag([Kp_0] * 3 + [Kr_0] * 3)
        self._Ds_0 = np.diag([4 * np.sqrt(Kp_0)] * 3 + [2 * np.sqrt(Kr_0)] * 3)
        self._Ks_1 = np.diag([Kp_1] * 3 + [Kr_1] * 3)
        self._Ds_1 = np.diag([4 * np.sqrt(Kp_1)] * 3 + [2 * np.sqrt(Kr_1)] * 3)

    def compute_tau(self, env_idx, target_transform):
        # primary task - ee control
        ee_transform = self._franka.get_ee_transform(env_idx, self._franka_name)

        J = self._franka.get_jacobian(env_idx, self._franka_name)
        q_dot = self._franka.get_joints_velocity(env_idx, self._franka_name)[:7]
        x_vel = J @ q_dot

        tau_0 = compute_task_space_impedance_control(J, ee_transform, target_transform, x_vel, self._Ks_0, self._Ds_0)

        # secondary task - elbow straight
        link_transforms = self._franka.get_links_transforms(env_idx, self._franka_name)
        elbow_transform = link_transforms[self._elbow_joint]

        u0 = vec3_to_np(link_transforms[0].p)[:2]
        u1 = vec3_to_np(link_transforms[-1].p)[:2]
        curr_elbow_xyz = vec3_to_np(elbow_transform.p)
        goal_elbow_xy = project_to_line(curr_elbow_xyz[:2], u0, u1)
        elbow_target_transform = gymapi.Transform(
            p=gymapi.Vec3(goal_elbow_xy[0], goal_elbow_xy[1], curr_elbow_xyz[2] + 0.2),
            r=elbow_transform.r
        )

        J_elb = self._franka.get_jacobian(env_idx, self._franka_name, target_joint=self._elbow_joint)
        x_vel_elb = J_elb @ q_dot

        tau_1 = compute_task_space_impedance_control(J_elb, elbow_transform, elbow_target_transform, x_vel_elb, self._Ks_1, self._Ds_1)
        
        # nullspace projection
        JT_inv = np.linalg.pinv(J.T)
        Null = np.eye(7) - J.T @ (JT_inv)
        tau = tau_0 + Null @ tau_1

        return tau


class EEImpedanceWaypointPolicy(Policy):

    def __init__(self, franka, franka_name, init_ee_transform, goal_ee_transform, T=300):
        self._franka = franka
        self._franka_name = franka_name

        self._T = T
        self._ee_impedance_ctrlr = FrankaEEImpedanceController(franka, franka_name)

        init_ee_pos = vec3_to_np(init_ee_transform.p)
        goal_ee_pos = vec3_to_np(goal_ee_transform.p)
        self._traj = [
            gymapi.Transform(
                p=np_to_vec3(min_jerk(init_ee_pos, goal_ee_pos, t, self._T)),
                r=slerp_quat(init_ee_transform.r, goal_ee_transform.r, t, self._T),
            )
            for t in range(self._T)
        ]

    @property
    def horizon(self):
        return self._T

    def __call__(self, scene, env_idx, t_step, t_sim):
        target_transform = self._traj[min(t_step, self._T - 1)]
        tau = self._ee_impedance_ctrlr.compute_tau(env_idx, target_transform)
        self._franka.apply_torque(env_idx, self._franka_name, tau)

class SetJointPolicy(Policy):

    def __init__(self, franka, franka_name, joints, T=15):
        self._franka = franka
        self._franka_name = franka_name
        self._joints = joints

        self._T = T

    @property
    def horizon(self):
        return self._T

    def __call__(self, scene, env_idx, t_step, t_sim):
        self._franka.set_joints(env_idx, self._franka_name, self._joints)

class FrankaJointController:
    """
    returns joint torques to reach target joint positions
    """
    def __init__(self, franka, franka_name, P_gain, D_gain):
        self._franka = franka
        self._franka_name = franka_name
        self._elbow_joint = 3
        self._P_gain = P_gain
        self._D_gain = D_gain
        self._prev_delta = None

    def compute_tau(self, env_idx, target_joints_pos,dt):

        delta_joints = target_joints_pos - self._franka.get_joints(env_idx, self._franka_name)[:self._franka.num_dof]
        if self._prev_delta is None:
            self._prev_delta = delta_joints
        error_rate = (delta_joints - self._prev_delta) / dt
        # error_rate = (delta_joints) / dt
        self._prev_delta = delta_joints
        # print(delta_joints)
        # self._franka.apply_delta_joint_targets(env_idx, self._name, delta_joints)
        return self._P_gain @ delta_joints + self._D_gain @ error_rate


class FrankaJointWayPointPolicy(Policy):
    """
    follows a joint trajectory if given, 
    otherwise linearly interpolates between init and goal joint positions
    """
    def __init__(self, franka, franka_name, init_joint_pos, goal_joint_pos, 
            traj,
            P_gain,
            D_gain,
            T=300):
        self._franka = franka
        self._franka_name = franka_name

        self._T = T
        self._time_horizon = T
        self._joint_ctrlr = FrankaJointController(franka, franka_name,P_gain,D_gain)
        if traj is None:
            self._traj = np.linspace(init_joint_pos, goal_joint_pos, num=T)
        else:
            self._traj = traj
            assert len(traj) == T and len(traj[0]) == 7 \
                and (traj[0] == init_joint_pos).all() and (traj[-1] == goal_joint_pos).all()
        self.actual_traj = []

    @property
    def horizon(self):
        return self._T
    @property
    def time_horizon(self):
        return self._time_horizon

    def __call__(self, scene, env_idx, t_step, t_sim):
        ndof = self._franka.num_dof
        target_joint_pos = self._traj[min(t_step, self._T - 1)]
        tau = self._joint_ctrlr.compute_tau(env_idx, target_joint_pos,scene.dt)
        tau = np.clip(tau,self._franka.joint_limits_lower[:ndof],self._franka.joint_limits_upper[:ndof])
        self._franka.apply_torque(env_idx, self._franka_name, tau)
        self.actual_traj.append(self._franka.get_joints(env_idx, self._franka_name)[:ndof])


class RRTFollowingPolicy(Policy):
    def __init__(self, franka, franka_name, traj, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._franka = franka
        self._franka_name = franka_name

        self._traj = traj
        self._time_horizon = 1000
        self.WAYPT_DURATION = 100
        self.INCREASE_TAU_THRESHOLD = 1
        self.P_GAIN = np.diag([10,10,10,10,10,10,10])
        self.D_GAIN = np.diag([0.1,0.1,0.1,0.1,0.1,0.1,0.1])
        self.reset()


    def reset(self):
        self._joint_pos = []

        self._joint_waypoint_policies = []


    def __call__(self, scene, env_idx, t_step, t_sim):
        tgt_joint_pos = self._traj[min(t_step, self._time_horizon - 1)]
        joint_pos = self._franka.get_joints(env_idx, self._franka_name)[:self._franka.num_dof]
        self._joint_pos.append(joint_pos)
        # initialize a waypoint policy
        if t_step == 0:
            self._joint_waypoint_policies = [FrankaJointWayPointPolicy(self._franka, 
            self._franka_name, joint_pos, tgt_joint_pos,
            P_gain=self.P_GAIN, D_gain=self.D_GAIN,
            T=self.WAYPT_DURATION)]
        
        # check every WAYPT_DURATION steps to update the waypoint policy
        elif t_step % self.WAYPT_DURATION == 0:
            exp_init_joint_pos = self._traj[min(t_step-1, self._time_horizon - 1)]
            diff = np.linalg.norm(joint_pos-exp_init_joint_pos)
            # use a bigger gain (tau_factor) if the starting position is far 
            # from the expected starting position
            tau_factor = self.INIT_TAU_FACTOR #* (1+diff/self.INCREASE_TAU_THRESHOLD)
            self._joint_waypoint_policies[env_idx] = FrankaJointWayPointPolicy(self._franka,
             self._franka_name, joint_pos, tgt_joint_pos, 
             P_gain=self.P_GAIN*tau_factor, D_gain=self.D_GAIN*tau_factor, 
             T=self.WAYPT_DURATION)

        # call the actual policy to apply torque
        self._joint_waypoint_policies[env_idx](scene, env_idx, t_step, t_sim)



   
class GraspTreePolicy(Policy):

    def __init__(self, franka, franka_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._franka = franka
        self._franka_name = franka_name

        self._time_horizon = 1000

        self.reset()

    def reset(self):
        self._pre_grasp_transforms = []
        self._grasp_transforms = []
        self._init_ee_transforms = []
        self._ee_waypoint_policies = []

    def set_grasp_goal(self, grasp_goal):
        self._grasp_goal = grasp_goal

    def __call__(self, scene, env_idx, t_step, t_sim):
        ee_transform = self._franka.get_ee_transform(env_idx, self._franka_name)

        if t_step == 0:
            self._init_ee_transforms.append(ee_transform)
            self._ee_waypoint_policies.append(
                EEImpedanceWaypointPolicy(self._franka, self._franka_name, ee_transform, ee_transform, T=20)
            )

       

        if t_step == 20:
            #convert grasp goal to transform
            grasp_pos = np_to_vec3(self._grasp_goal[0:3])
            grasp_rot = np_to_quat(self._grasp_goal[3:7])


            grasp_transform = gymapi.Transform(p=grasp_pos, r=grasp_rot)

            self._grasp_transforms.append(grasp_transform)
      

            self._ee_waypoint_policies[env_idx] = \
                EEImpedanceWaypointPolicy(
                    self._franka, self._franka_name, ee_transform, self._grasp_transforms[env_idx], T=180
                )

   
        self._ee_waypoint_policies[env_idx](scene, env_idx, t_step, t_sim)

