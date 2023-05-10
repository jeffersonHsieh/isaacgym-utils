import numpy as np
from pathlib import Path

from isaacgym import gymapi
from isaacgym_utils.constants import isaacgym_utils_ASSETS_PATH
from isaacgym_utils.math_utils import transform_to_RigidTransform, vec3_to_np, quat_to_rot, np_to_vec3

from .assets import GymURDFAsset
from .franka_numerical_utils import get_franka_mass_matrix


class GymTree(GymURDFAsset):

    global num_joints, joint_names, num_links, link_names

    joint_names = ['joint0_x_to_1', 'joint0_y_to_1', 'joint0_z_to_1', 'joint1_x_to_2', 'joint1_y_to_2', 'joint1_z_to_2', 'joint1_x_to_4', 
    'joint1_y_to_4', 'joint1_z_to_4', 'joint1_x_to_8', 'joint1_y_to_8', 'joint1_z_to_8', 'joint2_x_to_3', 'joint2_y_to_3', 'joint2_z_to_3', 
    'joint2_x_to_6', 'joint2_y_to_6', 'joint2_z_to_6', 'joint3_x_to_5', 'joint3_y_to_5', 'joint3_z_to_5', 'joint3_x_to_9', 'joint3_y_to_9', 
    'joint3_z_to_9', 'joint4_x_to_7', 'joint4_y_to_7', 'joint4_z_to_7', 'joint6_x_to_10', 'joint6_y_to_10', 'joint6_z_to_10']

    link_names = ['base_link', 'link_0_to_1', 'link_1_to_2', 'link_1_to_4', 
    'link_1_to_8', 'link_2_to_3', 'link_2_to_6', 'link_3_to_5', 'link_3_to_9', 'link_4_to_7', 
    'link5_tip', 'link_6_to_10', 'link7_tip', 'link8_tip', 'link9_tip', 'link10_tip']
    num_joints = len(joint_names)
    num_links = len(link_names)
    min_angle = -np.pi/16
    max_angle = np.pi/16
    print(f" num_joints: {num_joints} and num_links: {num_links} ")

    INIT_JOINTS = np.zeros(num_joints)
    # INIT_JOINTS = np.array(np.random.uniform(min_angle,max_angle,num_joints))
    _LOWER_LIMITS = None
    _UPPER_LIMITS = None
    _VEL_LIMITS = None

    # _URDF_PATH = '/home/marklee/github/build_sdf/generated_urdf/tree_pruned.urdf'
    _URDF_PATH = 'franka_description/robots/[10]tree0_ocrl.urdf'
    # _URDF_PATH = 'franka_description/robots/tree_test.urdf'

    @staticmethod
    def _key(env_idx, name):
        return (env_idx, name)

    def __init__(self, cfg, *args, actuation_mode='joints'):
        if 'urdf' in cfg:
            urdf_path = cfg['urdf']
            assets_root = Path(cfg['assets_root'])
        else:
            urdf_path = GymTree._URDF_PATH
            assets_root = isaacgym_utils_ASSETS_PATH

        super().__init__(urdf_path, *args,
                        shape_props=cfg['shape_props'],
                        dof_props=cfg['dof_props'],
                        asset_options=cfg['asset_options'],
                        assets_root=assets_root
                        )

        # print(f"init tree func {(cfg['dof_props'])} ")   


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

        self.joint_names = joint_names
        self.link_names = link_names
        self.num_joints = num_joints
        self.num_links = num_links

        

        if actuation_mode == 'attractors':
            self._attractor_stiffness = cfg['attractor_props']['stiffness']
            self._attractor_damping = cfg['attractor_props']['damping']

    def get_tips_transforms(self, env_idx, name):
        return [
            self.get_rb_transform(env_idx, name, f'link{i}_tip')
            for i in [5, 7, 8, 9, 10]
        ]

    def get_tips_rigid_transforms(self, env_idx, name):
        transforms = self.get_tips_transforms(env_idx, name)
        return [transform_to_RigidTransform(transform,
                                        from_frame=f'link{i}_tip'.format(i+1),
                                        to_frame='world')
                for i, transform in zip([5, 7, 8, 9, 10], transforms)]

    def get_link_transform(self, env_idx, name, link_name):
        transform = self.get_rb_transform(env_idx, name, link_name) 
        return transform

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
                'driveMode': [gymapi.DOF_MODE_NONE] * num_joints
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
                'driveMode': [gymapi.DOF_MODE_POS] * num_joints
            })
        elif self._actuation_mode == 'torques':
            self.set_dof_props(env_idx, name, {
                'driveMode': [gymapi.DOF_MODE_EFFORT] * num_joints
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

        # print(f" ----------- inside post create actor --------- ")

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


