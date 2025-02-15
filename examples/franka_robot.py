import numpy as np
import quaternion
from itertools import product

# import rospy
# from sensor_msgs.msg import JointState
# from collision_boxes_publisher import CollisionBoxesPublisher

class FrankaRobot:

    INIT_JOINTS = np.array([0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4,])
    joint_limits_low = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    joint_limits_high = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    home_joints = np.array([0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4])
    dh_params = np.array([[0, 0.333, 0, 0],
                            [0, 0, -np.pi/2, 0],
                            [0, 0.316, np.pi/2, 0],
                            [0.0825, 0, np.pi/2, 0],
                            [-0.0825, 0.384, -np.pi/2, 0],
                            [0, 0, np.pi/2, 0],
                            [0.088, -0.107, np.pi/2, 0],
                            [0, 0.107, 0, -0.785398163397],
                            [0, -0.1034, np.pi, 0]])
                            # [0.088, 0, np.pi/2, 0],
                            # [0, 0.107, 0, 0],
                            # [0, 0.1034, 0, 0]])
    num_dof = 7

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
    collision_box_shapes *= 1
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

    def __init__(self):
        # self._joint_state_pub = rospy.Publisher('joint_states', JointState, queue_size=10)
        # self._collision_boxes_pub = CollisionBoxesPublisher('franka_collision_boxes')

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

    def set_base_offset(self, offset):
        self.base_offset = np.array(offset)

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

    def ee(self, joints):
        '''
        Arguments: array of joint positions (rad)
        Returns: A numpy array that contains the [x, y, z, roll, pitch, yaw] location of the end-effector.
        '''
        fk = self.forward_kinematics(joints)
        ee_frame = fk[-1,:,:]

        x, y, z = ee_frame[:-1,3]
        roll = np.arctan2(ee_frame[2,1], ee_frame[2,2])
        pitch = np.arcsin(-ee_frame[2,0])
        yaw = np.arctan2(ee_frame[1,0], ee_frame[0,0])

        ee = np.array([x, y, z, roll, pitch, yaw])
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

    def inverse_kinematics(self, desired_ee_pos, current_joints):
        '''
        Arguments: desired_ee_pos which is a np array of [x, y, z, r, p, y] which represent the desired end-effector position of the robot
                   current_joints which represents the current location of the robot
        Returns: A numpy array that contains the joints required in order to achieve the desired end-effector position.
        '''
        joints = current_joints.copy()
        current_ee_pos = self.ee(joints)
        ee_error = desired_ee_pos - current_ee_pos
        alpha = 0.1

        while np.linalg.norm(ee_error) > 1e-3:
            jacob = self.jacobian(joints)
            joints += alpha * jacob.T.dot(ee_error.T)
            
            current_ee_pos = self.ee(joints)
            ee_error = desired_ee_pos - current_ee_pos

        return joints
    
    def get_obstacle_vertices(self, box):
        box_pos, box_rpy, box_hsizes = box[:3], box[3:6], box[6:]/2
        box_q = quaternion.from_euler_angles(box_rpy)
        box_axes = quaternion.as_rotation_matrix(box_q)

        self._box_vertices_offset[:,:] = self._vertex_offset_signs * box_hsizes
        box_vertices = (box_axes.dot(self._box_vertices_offset.T) + np.expand_dims(box_pos, 1)).T
        return box_vertices


    def check_box_collision(self, joints, box):
        '''
        Arguments: joints represents the current location of the robot
                   box contains the position of the center of the box [x, y, z, r, p, y] and the length, width, and height [l, w, h]
        Returns: A boolean where True means the box is in collision with the arm and false means that there are no collisions.
        '''
        box_pos, box_rpy, box_hsizes = box[:3], box[3:6], box[6:]/2
        box_q = quaternion.from_euler_angles(box_rpy)
        box_axes = quaternion.as_rotation_matrix(box_q)

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

        box_poses_world = []
        for i, link in enumerate(self._collision_box_links):
            # if i in [7, 8, 9]:
            #     continue 
            link_transform = fk[link - 1]
            box_pose_world = link_transform.dot(self._collision_box_poses[i])
            box_pose_world[:3, 3] += self.base_offset
            box_poses_world.append(box_pose_world)

        return box_poses_world
