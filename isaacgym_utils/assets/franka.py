import numpy as np
from pathlib import Path
import quaternion
from scipy.spatial.transform import Rotation as R
from itertools import product

from isaacgym import gymapi
from isaacgym_utils.constants import isaacgym_utils_ASSETS_PATH
from isaacgym_utils.math_utils import transform_to_RigidTransform, vec3_to_np, quat_to_rot, np_to_vec3, rot_to_np_quat, quat_to_np, angle_axis_between_quats

from .assets import GymURDFAsset
from .franka_numerical_utils import get_franka_mass_matrix


class GymFranka(GymURDFAsset):

    INIT_JOINTS = np.array([0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4, 0.04, 0.04])
    _LOWER_LIMITS = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0, 0])
    _UPPER_LIMITS = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04])
    _VEL_LIMITS = None

    _URDF_PATH = 'franka_description/robots/franka_panda.urdf'
    _URDF_PATH_WITH_DYNAMICS = 'franka_description/robots/franka_panda_dynamics.urdf'

    # a d alpha theta
    dh_params = np.array([
                    [0, 0.333, 0, 0],
                    [0, 0, -np.pi/2, 0],
                    [0, 0.316, np.pi/2, 0],
                    [0.0825, 0, np.pi/2, 0],
                    [-0.0825, 0.384, -np.pi/2, 0],
                    [0, 0, np.pi/2, 0],
                    [0.088, 0, np.pi/2, 0],
                    [0, 0.107, 0, -0.785398163397],
                    [0, 0.1034, 0, 0]])
                    # [0.088, 0, np.pi/2, 0],
                    # [0, 0.107, 0, 0],
                    # [0, 0.1034, 0, 0]])

    _dh_alpha_rot = np.array([
                        [1, 0, 0, 0],
                        [0, -1, -1, 0],
                        [0, -1, -1, 0],
                        [0, 0, 0, 1]
                        ], dtype=np.float32)
    _dh_a_trans = np.array([
                        [1, 0, 0, -1],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]
                        ], dtype=np.float32)
    _dh_d_trans = np.array([
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, -1],
                        [0, 0, 0, 1]
                        ], dtype=np.float32)
    _dh_theta_rot = np.array([
                        [-1, -1, 0, 0],
                        [-1, -1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]
                        ], dtype=np.float32)

    collision_box_shapes = np.array([
        [0.23, 0.2, 0.1],
        [0.13, 0.12, 0.1], 
        [0.12, 0.1, 0.2],
        [0.15, 0.27, 0.11],
        [0.12, 0.1, 0.2],
        [0.13, 0.12, 0.25],
        [0.13, 0.23, 0.15],
        [0.12, 0.12, 0.4],
        [0.12, 0.12, 0.25],
        [0.13, 0.23, 0.12],
        [0.12, 0.12, 0.2],
        [0.08, 0.22, 0.17]
    ])
    collision_box_shapes *= 0.2
    _collision_box_links = [1, 1, 1, 1, 1, 3, 4, 5, 5, 5, 7, 7]
    _collision_box_poses_raw = np.array([
        [-.04, 0, -0.283, 1, 0, 0, 0],
        [-0.009, 0, -0.183, 1, 0, 0, 0],
        [0, -0.032, -0.082, 0.95141601, 0.30790838, 0, 0],
        [-0.008, 0, 0, 1, 0, 0, 0],
        [0, .042, .067, 0.95141601, 0.30790838, 0, 0],
        [0.00687, 0, -0.139, 1, 0, 0, 0],
        [-0.008, 0.004, 0, 0.70710678, -0.70710678, 0, 0],
        [0.00422, 0.05367, -0.121, 0.9961947, -0.08715574, 0, 0],
        [0.00422,  0.00367, -0.263, 1, 0, 0, 0],
        [0.00328, 0.0176, -0.0055, 1, 0, 0, 0],
        [-0.0136, 0.0092, 0.0083, 0, 1, 0, 0],
        [0.0136,  -0.0092,  0.1457, 0.92387953, 0, 0, -0.38268343]
    ])

    @staticmethod
    def _key(env_idx, name):
        return (env_idx, name)

    def __init__(self, cfg, *args, actuation_mode='joints'):
        self.num_dof = 7

        if 'urdf' in cfg:
            urdf_path = cfg['urdf']
            assets_root = Path(cfg['assets_root'])
        else:
            urdf_path = GymFranka._URDF_PATH_WITH_DYNAMICS
            assets_root = isaacgym_utils_ASSETS_PATH
        super().__init__(urdf_path, *args,
                        shape_props=cfg['shape_props'],
                        dof_props=cfg['dof_props'],
                        asset_options=cfg['asset_options'],
                        assets_root=assets_root
                        )

        self._use_custom_ee = False
        if 'custom_ee_rb_name' in cfg:
            self._use_custom_ee = True
            self._custom_ee_rb_name = cfg['custom_ee_rb_name']

        self._left_finger_rb_name = cfg.get('custom_left_finger_rb_name', 'panda_leftfinger')
        self._right_finger_rb_name = cfg.get('custom_right_finger_rb_name', 'panda_rightfinger')

        self._ee_tool_offset = gymapi.Transform()
        if 'custom_ee_offset' in cfg:
            self._ee_tool_offset = gymapi.Transform((np_to_vec3(cfg['custom_ee_offset'])))

        self._gripper_offset = gymapi.Transform(gymapi.Vec3(0, 0, 0.1034))
        self._finger_offset = gymapi.Transform(gymapi.Vec3(0, 0, 0.045))

        self._actuation_mode = actuation_mode
        self._attractor_handles_map = {}
        self._attractor_transforms_map = {}

        if actuation_mode == 'attractors':
            self._attractor_stiffness = cfg['attractor_props']['stiffness']
            self._attractor_damping = cfg['attractor_props']['damping']
        
    def precompute_self_collision_box_data(self):
        self._collision_boxes_data = np.zeros((len(self.collision_box_shapes), 10))
        self._collision_boxes_data[:, -3:] = self.collision_box_shapes

        # Precompute things and preallocate np memory for collision checking
        self._collision_box_poses = []
        for pose in self._collision_box_poses_raw:
            T = np.eye(4)
            T[:3, 3] = pose[:3]
            T[:3, :3] = quaternion.as_rotation_matrix(quaternion.quaternion(*pose[3:]))
            self._collision_box_poses.append(T)

        self._collision_box_hdiags = []
        self._collision_box_vertices_offset = []
        self._vertex_offset_signs = np.array(list(product([1, -1],[1,-1], [1,-1])))
        for sizes in self.collision_box_shapes:
            hsizes = sizes/2

            self._collision_box_vertices_offset.append(self._vertex_offset_signs * hsizes)
            self._collision_box_hdiags.append(np.linalg.norm(sizes/2))
        self._collision_box_vertices_offset = np.array(self._collision_box_vertices_offset)
        self._collision_box_hdiags = np.array(self._collision_box_hdiags)

        self._collision_proj_axes = np.zeros((3, 15))
        self._box_vertices_offset = np.ones([8, 3])
        self._box_transform = np.eye(4)

    def set_gripper_width_target(self, env_idx, name, width):
        joints_targets = self.get_joints_targets(env_idx, name)
        joints_targets[-2:] = width
        self.set_joints_targets(env_idx, name, joints_targets)

    def open_grippers(self, env_idx, name):
        self.set_gripper_width_target(env_idx, name, 0.04)

    def close_grippers(self, env_idx, name):
        self.set_gripper_width_target(env_idx, name, 0)

    def set_gripper_width(self, env_idx, name, width):
        width = np.clip(width, self._LOWER_LIMITS[-1], self._UPPER_LIMITS[-1])
        self.set_gripper_width_target(env_idx, name, width)

        joints = self.get_joints(env_idx, name)
        joints[-2] = width
        self.set_joints(env_idx, name, joints)

    def get_gripper_width(self, env_idx, name):
        return self.get_joints(env_idx, name)[-1]

    def get_base_transform(self, env_idx, name):
        return self.get_rb_transform(env_idx, name, 'panda_link0')

    def get_ee_transform(self, env_idx, name, offset=True):
        ee_transform = self.get_rb_transform(env_idx, name, 'panda_hand')
        if offset:
            ee_transform = ee_transform * self._gripper_offset * self._ee_tool_offset
        return ee_transform

    def get_ee_transform_MARK(self, env_idx, name, link_name, offset=True):
        ee_transform = self.get_rb_transform(env_idx, name, link_name)
        if offset:
            ee_transform = ee_transform 
        return ee_transform


    def get_ee_rigid_transform(self, env_idx, name, offset=True):
        return transform_to_RigidTransform(self.get_ee_transform(env_idx, name, offset=offset),
                                                from_frame='panda_ee', to_frame='panda_link0')

    def get_finger_transforms(self, env_idx, name, offset=True):
        lf_transform = self.get_rb_transform(env_idx, name, self._left_finger_rb_name)
        rf_transform = self.get_rb_transform(env_idx, name, self._right_finger_rb_name)

        if offset:
            lf_transform = lf_transform * self._finger_offset
            rf_transform = rf_transform * self._finger_offset

        return lf_transform, rf_transform

    def get_desired_ee_transform(self, env_idx, name):
        if self._actuation_mode != 'attractors':
            raise ValueError('Can\'t get desired ee transform when not using attractors!')

        key = self._key(env_idx, name)
        return self._attractor_transforms_map[key]

    def get_left_finger_ct_forces(self, env_idx, name):
        rbi = self.rb_names_map[self._left_finger_rb_name]
        return self.get_rb_ct_forces(env_idx, name)[rbi]

    def get_right_finger_ct_forces(self, env_idx, name):
        rbi = self.rb_names_map[self._right_finger_rb_name]
        return self.get_rb_ct_forces(env_idx, name)[rbi]

    def get_ee_ct_forces(self, env_idx, name):
        if self._use_custom_ee:
            rbi = self.rb_names_map[self._custom_ee_rb_name]
            ct_forces = self.get_rb_ct_forces(env_idx, name)[rbi]
        else:
            ct_forces_lf = self.get_left_finger_ct_forces(env_idx, name)
            ct_forces_rf = self.get_right_finger_ct_forces(env_idx, name)
            ct_forces = ct_forces_lf + ct_forces_rf

        return ct_forces


    def apply_force(self, env_idx, name, rb_name, force, loc):
        env_ptr = self._scene.env_ptrs[env_idx]
        ah = self._scene.ah_map[env_idx][name]
        bh = self._scene.gym.get_actor_rigid_body_index(env_ptr, ah, self.rb_names_map[rb_name], gymapi.DOMAIN_ENV)

        if self._scene.use_gpu_pipeline:
            for i, k in enumerate('xyz'):
                self._scene.tensors['forces'][env_idx, bh, i] = getattr(force, k)
                self._scene.tensors['forces_pos'][env_idx, bh, i] = getattr(loc, k)
            self._scene.register_actor_tensor_to_update(env_idx, name, 'forces')
            return True
        else:
            return self._scene.gym.apply_body_force(env_ptr, bh, force, loc)



    @property
    def joint_limits_lower(self):
        return self._LOWER_LIMITS

    @property
    def joint_limits_upper(self):
        return self._UPPER_LIMITS

    @property
    def joint_max_velocities(self):
        return self._VEL_LIMITS

    def set_actuation_mode(self, mode, env_idx, name):
        self._actuation_mode = mode
        env_ptr = self._scene.env_ptrs[env_idx]
        if self._actuation_mode == 'attractors':
            self.set_dof_props(env_idx, name, {
                'driveMode': [gymapi.DOF_MODE_NONE] * 7 + [gymapi.DOF_MODE_POS] * 2
            })

            key = self._key(env_idx, name)
            if key not in self._attractor_handles_map:
                attractor_props = gymapi.AttractorProperties()
                attractor_props.stiffness = self._attractor_stiffness
                attractor_props.damping= self._attractor_damping
                attractor_props.axes = gymapi.AXIS_ALL

                gripper_handle = self._scene.gym.get_rigid_handle(env_ptr, name, 'panda_hand')
                attractor_props.rigid_handle = gripper_handle
                attractor_props.offset = self._gripper_offset * self._ee_tool_offset

                attractor_handle = self._scene.gym.create_rigid_body_attractor(env_ptr, attractor_props)
                self._attractor_handles_map[key] = attractor_handle

            gripper_transform = self.get_ee_transform(env_idx, name)
            self.set_ee_transform(env_idx, name, gripper_transform)
        elif self._actuation_mode == 'joints':
            self.set_dof_props(env_idx, name, {
                'driveMode': [gymapi.DOF_MODE_POS] * 9
            })
        elif self._actuation_mode == 'torques':
            self.set_dof_props(env_idx, name, {
                'driveMode': [gymapi.DOF_MODE_EFFORT] * 7 + [gymapi.DOF_MODE_POS] * 2
            })
        else:
            raise ValueError('Unknown actuation mode! Must be attractors, joints, or torques!')

    def _post_create_actor(self, env_idx, name):
        super()._post_create_actor(env_idx, name)
        self.set_joints(env_idx, name, self.INIT_JOINTS)
        self.set_joints_targets(env_idx, name, self.INIT_JOINTS)

        if self._LOWER_LIMITS is None or self._UPPER_LIMITS is None or self._VEL_LIMITS is None:
            dof_props = self.get_dof_props(env_idx, name)
            self._LOWER_LIMITS = dof_props['lower']
            self._UPPER_LIMITS = dof_props['upper']
            self._VEL_LIMITS = dof_props['velocity']

        self.set_actuation_mode(self._actuation_mode, env_idx, name)

    def set_attractor_props(self, env_idx, name, props):
        if self._actuation_mode != 'attractors':
            raise ValueError('Not using attractors!')
        env_ptr = self._scene.env_ptrs[env_idx]

        key = self._key(env_idx, name)
        ath = self._attractor_handles_map[key]
        attractor_props = self._scene.gym.get_attractor_properties(env_ptr, ath)

        for key, val in props.items():
            setattr(attractor_props, key, val)

        self._scene.gym.set_attractor_properties(env_ptr, ath, attractor_props)

    def set_ee_transform(self, env_idx, name, transform):
        if self._actuation_mode != 'attractors':
            raise ValueError('Can\'t set ee transform when not using attractors!')
        key = self._key(env_idx, name)
        attractor_handle = self._attractor_handles_map[key]

        self._attractor_transforms_map[key] = transform

        env_ptr = self._scene.env_ptrs[env_idx]
        self._scene.gym.set_attractor_target(env_ptr, attractor_handle, transform)

    def set_delta_ee_transform(self, env_idx, name, transform):
        ''' This performs delta translation in the global frame and
            delta rotation in the end-effector frame.
        '''
        current_transform = self.get_ee_transform(env_idx, name)
        desired_transform = gymapi.Transform(p=current_transform.p, r=current_transform.r)
        desired_transform.p = desired_transform.p + transform.p
        desired_transform.r = transform.r * desired_transform.r

        self.set_ee_transform(env_idx, name, desired_transform)

    def apply_torque(self, env_idx, name, tau):
        if len(tau) == 7:
            tau = np.concatenate([tau, np.zeros(2)])

        self.apply_actor_dof_efforts(env_idx, name, tau)

    def get_links_transforms(self, env_idx, name):
        return [
            self.get_rb_transform(env_idx, name, f'panda_link{i}')
            for i in range(1, 8)
        ]

    def get_links_rigid_transforms(self, env_idx, name):
        transforms = self.get_links_transforms(env_idx, name)
        return [transform_to_RigidTransform(transform,
                                        from_frame='panda_link{}'.format(i+1),
                                        to_frame='panda_link0')
                for i, transform in enumerate(transforms)]

    def get_jacobian(self, env_idx, name, target_joint=7):
        transforms = self.get_links_transforms(env_idx, name)

        if target_joint == 7:
            ee_pos = vec3_to_np(self.get_ee_transform(env_idx, name).p)
        else:
            ee_pos = vec3_to_np(transforms[target_joint].p)

        joints_pos, axes = np.zeros((7, 3)), np.zeros((7, 3))
        for i, transform in enumerate(transforms[:target_joint]):
            joints_pos[i] = vec3_to_np(transform.p)
            axes[i] = quat_to_rot(transform.r)[:, 2]
        J = np.r_[np.cross(axes, ee_pos - joints_pos).T, axes.T]

        return J

    def get_mass_matrix(self, env_idx, name):
        q = self.get_joints(env_idx, name)[:7]
        return get_franka_mass_matrix(q)

    def reset_joints(self, env_idx, name):
        self.set_joints(env_idx, name, self.INIT_JOINTS)
    
    def set_base_offset(self, offset):
        self.base_offset = np.array(offset)

    def ee(self, joints):
        '''
        Arguments: array of joint positions (rad)
        Returns: A numpy array that contains the [x, y, z, roll, pitch, yaw] location of the end-effector.
        '''
        fk = self.forward_kinematics(joints)
        ee_frame = fk[-2,:,:]

        x, y, z = ee_frame[:-1,3]
        roll = np.arctan2(ee_frame[2,1], ee_frame[2,2])
        pitch = np.arcsin(-ee_frame[2,0])
        yaw = np.arctan2(ee_frame[1,0], ee_frame[0,0])

        # r = R.from_matrix(ee_frame[:3,:3])
        # r = r.as_euler('xyz')

        # ee = np.array([x, y, z, *r])
        ee = np.array([x, y, z, roll, pitch, yaw])

        # ee = np.zeros(7)
        # ee[:3] = ee_frame[:-1,3]
        # ee[-4:] = rot_to_np_quat(ee_frame[:3,:3]) # x y z w

        ee[:3] += self.base_offset

        return ee
    
    def get_links_poses(self, joints):
        links_poses = []
        fk = self.forward_kinematics(joints)
        for i, fk_i in enumerate(fk):
            frame = fk[i,:,:]

            x, y, z = frame[:-1,3]
            roll = np.arctan2(frame[2,1], frame[2,2])
            pitch = np.arcsin(-frame[2,0])
            yaw = np.arctan2(frame[1,0], frame[0,0])

            link_pose = np.array([x, y, z, roll, pitch, yaw])
            link_pose[:3] += self.base_offset

            links_poses.append(link_pose)
        
        links_poses = np.array(links_poses)

        return links_poses

    def jacobian(self, joints):
        '''
        Calculate the jacobians analytically using your forward kinematics
        Arguments: array of joint positions (rad)
        Returns: A numpy array that contains the 6 x num_dof end-effector jacobian.
        '''
        jacobian = np.zeros((6,self.num_dof))
        fk = self.forward_kinematics(joints)
        ee_pos = fk[-1,:3,3]

        for i in range(self.num_dof):
            joint_pos = fk[i,:3,3]
            joint_axis = fk[i,:3,2]
            jacobian[:3,i] = np.cross(joint_axis, (ee_pos - joint_pos)).T
            jacobian[3:,i] = joint_axis.T

        return jacobian

    def ee_error(self, desired_ee_pos, current_ee_pos):
        x_pos = current_ee_pos[:3]
        x_quat = quaternion.from_euler_angles(current_ee_pos[-3:])

        xd_pos = desired_ee_pos[:3]
        xd_quat = quaternion.from_euler_angles(desired_ee_pos[-3:])

        xe_pos = x_pos - xd_pos
        xe_ang_axis = angle_axis_between_quats(x_quat, xd_quat)
        ee_error = np.concatenate([xe_pos, xe_ang_axis])
        
        return ee_error


    def inverse_kinematics(self, desired_ee_pos, current_joints):
        '''
        Arguments: desired_ee_pos which is a np array of [x, y, z, r, p, y] which represent the desired end-effector position of the robot
                   current_joints which represents the current location of the robot
        Returns: A numpy array that contains the joints required in order to achieve the desired end-effector position.
        '''
        joints = current_joints.copy()
        current_ee_pos = self.ee(joints)

        # ee_error = desired_ee_pos - current_ee_pos
        ee_error = self.ee_error(desired_ee_pos, current_ee_pos)

        alpha = 0.1

        for i in range(10000):
            jacob = self.jacobian(joints)
            # joints += alpha * jacob.T.dot(ee_error.T)
            joints -= alpha * np.linalg.pinv(jacob).dot(ee_error.T)
            
            current_ee_pos = self.ee(joints)
            # ee_error = desired_ee_pos - current_ee_pos
            ee_error = self.ee_error(desired_ee_pos, current_ee_pos)
            if np.linalg.norm(ee_error) < 1e-3:
                print(f'IK solved, pos: {current_ee_pos}')
                print(np.linalg.norm(ee_error))
                return joints

        print('cannot solve inverse kinematics for target pose')
        print(f'current pos: {current_ee_pos}')
        print(np.linalg.norm(ee_error))
        return joints

    def forward_kinematics(self, joints):
        '''
        Calculate the position of each joint using the dh_params
        Arguments: array of joint positions (rad)
        Returns: A numpy array that contains the 4x4 transformation matrices from the base to the position of each joint.
        '''
        forward_kinematics = np.zeros((len(self.dh_params), 4, 4))
        previous_transformation = np.eye(4)

        for i in range(len(self.dh_params)):
            a, d, alpha, theta = self.dh_params[i]

            if i < self.num_dof:
                theta = theta + joints[i]

            ca, sa = np.cos(alpha), np.sin(alpha)
            ct, st = np.cos(theta), np.sin(theta)
            self._dh_alpha_rot[1, 1] = ca
            self._dh_alpha_rot[1, 2] = -sa
            self._dh_alpha_rot[2, 1] = sa
            self._dh_alpha_rot[2, 2] = ca

            self._dh_a_trans[0, 3] = a
            self._dh_d_trans[2, 3] = d

            self._dh_theta_rot[0, 0] = ct
            self._dh_theta_rot[0, 1] = -st
            self._dh_theta_rot[1, 0] = st
            self._dh_theta_rot[1, 1] = ct

            transform = self._dh_alpha_rot.dot(self._dh_a_trans).dot(self._dh_d_trans).dot(self._dh_theta_rot)

            forward_kinematics[i] = previous_transformation.dot(transform)
            previous_transformation = forward_kinematics[i]

        return forward_kinematics

    def check_box_collision(self, joints, box):
        '''
        Arguments: joints represents the current location of the robot
                   box contains the position of the center of the box [x, y, z, r, p, y] and the length, width, and height [l, w, h]
        Returns: A boolean where True means the box is in collision with the arm and false means that there are no collisions.
        '''
        # box_pos, box_rpy, box_hsizes = box[:3], box[3:6], box[6:]/2
        # box_q = quaternion.from_euler_angles(box_rpy)
        box_pos, box_q, box_hsizes = box[:3], box[3:7], box[7:]/2
        box_axes = quaternion.as_rotation_matrix(np.quaternion(*box_q))

        self._box_vertices_offset[:,:] = self._vertex_offset_signs * box_hsizes
        box_vertices = (box_axes.dot(self._box_vertices_offset.T) + np.expand_dims(box_pos, 1)).T

        box_hdiag = np.linalg.norm(box_hsizes)
        min_col_dists = box_hdiag + self._collision_box_hdiags

        franka_box_poses = self.get_collision_boxes_poses(joints)
        for i, franka_box_pose in enumerate(franka_box_poses):
            fbox_pos = franka_box_pose[:3, 3]
            fbox_axes = franka_box_pose[:3, :3]

            # coarse collision check
            if np.linalg.norm(fbox_pos - box_pos) > min_col_dists[i]:
                continue

            fbox_vertex_offsets = self._collision_box_vertices_offset[i]
            fbox_vertices = fbox_vertex_offsets.dot(fbox_axes.T) + fbox_pos

            # construct axes
            cross_product_pairs = np.array(list(product(box_axes.T, fbox_axes.T)))
            cross_axes = np.cross(cross_product_pairs[:,0], cross_product_pairs[:,1]).T
            self._collision_proj_axes[:, :3] = box_axes
            self._collision_proj_axes[:, 3:6] = fbox_axes
            self._collision_proj_axes[:, 6:] = cross_axes

            # projection
            box_projs = box_vertices.dot(self._collision_proj_axes)
            fbox_projs = fbox_vertices.dot(self._collision_proj_axes)
            min_box_projs, max_box_projs = box_projs.min(axis=0), box_projs.max(axis=0)
            min_fbox_projs, max_fbox_projs = fbox_projs.min(axis=0), fbox_projs.max(axis=0)

            # check if no separating planes exist
            if np.all([min_box_projs <= max_fbox_projs, max_box_projs >= min_fbox_projs]):
                return True
        
        return False

    def get_collision_boxes_poses(self, joints):
        fk = self.forward_kinematics(joints)
        # print(fk)

        box_poses_world = []
        for i, link in enumerate(self._collision_box_links):
            link_transform = fk[link - 1]
            box_pose_world = link_transform.dot(self._collision_box_poses[i])
            box_pose_world[:3, 3] += self.base_offset
            box_poses_world.append(box_pose_world)

        return box_poses_world

        # code for checking if each transform is consistent with isaac's answer 
        # fr_links_poses = self._franka.get_links_poses(joints_start)
        # isaac_link_transforms = self._franka.get_rb_transforms(env_idx, self._franka_name)
        # for i in range(len(fr_links_poses)):
        #     print(f'joint {i}')
        #     print('fr')
        #     print(fr_links_poses[i])
        #     print('iasac')
        #     print(isaac_link_transforms[i+1].p)
        #     print(isaac_link_transforms[i+1].r)

        # joints_target = np.array([0.0, 5e-1, 0.0, -2.3, 0.0, 2.8, 7.8e-1])
        # joints_target = np.array([0, 3.43e-1, 0, -2.235, 0, 2.567, 7.8e-1])
