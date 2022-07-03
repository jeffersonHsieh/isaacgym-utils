import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
from xml.dom import minidom
import os
from scipy.spatial.transform import Rotation

import yaml

BATCH_SIZE = 1
STIFFNESS_BASE = 11000
SIMULATION_STEP_SIZE = 0.01
GUI_ON = 0

def sphere(pt, a=1.0, b=1.0, c=1.0):
    r = (pt[0]-0.5)**2*a + (pt[1]-0.5)**2*b + (pt[2]-0.5)**2*c
    return r <= 0.25


class TreeGenerator(object):

    def __init__(self, att_pts_max, scaling, offset, da, dt, max_steps, step_width, max_tree_points, tip_radius, att_env_shape_funct=sphere, tree_id=0, pipe_model_exponent=2, att_pts_min=None, x_strech=1, y_strech=1, z_strech=1, step_width_scaling=1, env_num=1):
        """

        :param att_pts_max: maximum number of attraction points. for further info see initialize_att_pts
        :param scaling: defines tree crown size. for further info see initialize_att_pts
        :param offset: defines tree crown position. for further info see initialize_att_pts
        :param da: attraction distance
        :param dt: termination distance
        :param max_steps: maximum amount of steps taken to generate tree model
        :param step_width: the distance that parent and child nodes have to each other
        :param max_tree_points: algorithm stops after creating the specified number of tree points
        :param att_env_shape_funct: defines tree crown shape. for further info see initialize_att_pts
        """
        self.tree_points = np.array([[0,0,0]])
        if att_pts_min is None:
            att_pts_min = att_pts_max
        self.att_pts = self.initialize_att_pts(att_pts_max, att_pts_min, lambda pt: att_env_shape_funct(pt, x_strech, y_strech, z_strech), scaling, offset)
        min_da = self.min_da()
        if da < min_da:
            da = min_da
        self.da = da
        self.dt = dt
        self.max_steps = max_steps
        self.step_width = step_width
        self.closest_tp = self.initialize_closest_tp_list()
        self.edges = {} # dictionary with keys as parents referencing np.arrays of [children, edge_thickness]
        self.branch_thickness_dict = {}
        self.max_tree_points = max_tree_points
        self.tip_radius = tip_radius
        self.tree_id = tree_id
        self.scaling = scaling
        self.pipe_model_exponent = pipe_model_exponent
        self.step_width_scaling = step_width_scaling
        self.name_dict = {"joints":[], "links":[]}
        self.env_num = env_num
        self.edge_list = []

    def min_da(self):
        min_da = None
        for point in self.att_pts:
            if min_da is None:
                min_da = np.linalg.norm(point)
            else:
                if min_da > np.linalg.norm(point):
                    min_da = np.linalg.norm(point)
        return math.ceil(min_da)


    def initialize_closest_tp_list(self):
        closest_tp = []
        for ap in self.att_pts:
            closest_tp.append((0,np.linalg.norm(ap-[0,0,0])))
        return closest_tp

    def update_closest_tp_list(self, new_points, point_index):
        to_delete = []
        for new_point in new_points:
            ap_index = 0
            for ap in self.att_pts:
                if np.linalg.norm(ap-new_point) < self.closest_tp[ap_index][1]:
                    if np.linalg.norm(ap-new_point) <= self.dt:
                        #print("deleted")
                        to_delete.append(ap_index)
                    else:
                        self.closest_tp[ap_index] = (point_index, np.linalg.norm(ap-new_point))
                ap_index += 1
            point_index += 1

        deleted=0
        to_delete = list(set(to_delete))
        to_delete.sort()
        for ap_index in to_delete:
            self.att_pts = np.delete(self.att_pts, ap_index-deleted, 0)
            self.closest_tp.pop(ap_index-deleted)
            deleted+=1

        #print("att_pts \t\t" + str(len(self.att_pts)))
        #print("closest_tp  \t" + str(len(self.closest_tp)))

    @staticmethod
    def initialize_att_pts(att_pts_max, att_pts_min, att_env_shape_funct, scaling, offset):
        """
        function that initializes the attraction points within the envelope defined
        by att_env_shape_funct
        :param att_pts_max: number of the maximum amount of attraction points to be generated. Due to rejection
                            sampling this number will rarely be reached
        :param att_env_shape_funct: function returning a boolean value. True, if a given point is within the
                                    desired Envelope, false if not. Should consider, that the values checked against
                                    this are between 0 and 1 in each dimension
        :param scaling: scalar, that determines by how much we want to scale the points outwards (original position is
                        between 1 and 0). determines overall volume of the envelope
        :param offset:  vector that determines, by how much we shift the initial position of the envelope.
        :return: returns (3,n) array with the n attraction points in it
        """
        ret_pts = [[]]
        initial = True
        while len(ret_pts)/3 + BATCH_SIZE < att_pts_max and len(ret_pts)/3 < att_pts_min:
            rng = np.random.default_rng()
            pts = rng.random((BATCH_SIZE,3))
            for pt in pts:
                if att_env_shape_funct(pt):
                    if initial:
                        ret_pts = (pt+offset)*scaling
                        initial = False
                    else:
                        ret_pts = np.concatenate((ret_pts, (pt+offset)*scaling), axis=0)
        ret_pts = np.reshape(ret_pts, (int(len(ret_pts)/3),3))
        return ret_pts

    def generate_sv_sets(self):
        sv_sets = {}
        ap_index = 0
        for candidate in self.closest_tp:
            if candidate[1] <= self.da:
                if candidate[0] not in sv_sets.keys():
                    sv_sets[candidate[0]] = [ap_index]
                else:
                    sv_sets[candidate[0]].append(ap_index)
            ap_index += 1
        return sv_sets

    def generate_tree(self):
        #self.tree_points = np.array([[0,0,0],[0,0,1],[0,0.25,2],[0,0.2,2],[0,0.3,2],[0,0.15,2],[0,0.1,2],[0,0.05,2], [0,0,2],[0,0.35,2],[0,0.4,2],[0,0.5,2],[0,0.45,2]])
        #self.edges = {0:[1],1:[2,3,4,5,6,7,8,9,10,11,12]} #1:[2,3,4,5,6,7,8,9,10,11,12]
        #return self.tree_points
        i = 0
        while len(self.att_pts) >= 1 and i < self.max_steps and len(self.tree_points) < self.max_tree_points:
            sv_sets = self.generate_sv_sets()
            new_tps = []
            point_index = len(self.tree_points)
            for key in sv_sets.keys():
                new_tps.append(self.generate_new_point(key, sv_sets[key]))
                if len(self.tree_points) > self.max_tree_points:
                    break
            self.update_closest_tp_list(new_tps, point_index)
            self.step_width = self.step_width * self.step_width_scaling
            i += 1
        return self.tree_points

    def generate_new_point(self, tp_index, sv_set):
        active_att_pts = self.att_pts[sv_set]
        tp = self.tree_points[tp_index]
        vec = np.array([0,0,0])
        for ap in active_att_pts:
            tmp = (ap - tp)/np.linalg.norm((ap - tp))
            vec = vec + tmp
        vec = vec/np.linalg.norm(vec)
        new_tp = tp + self.step_width*vec

        self.tree_points = np.vstack((self.tree_points, new_tp))
        if tp_index in self.edges.keys():
            self.edges[tp_index] = np.append(self.edges[tp_index], (len(self.tree_points) - 1))
        else:
            self.edges[tp_index] = np.array([(len(self.tree_points) - 1)])
        return new_tp

    def find_leaves(self):
        tree_node_indices = range(0,len(self.tree_points)-1)
        leaves = []
        for index in tree_node_indices:
            if len(self.find_children(index)) == 0:
                leaves.append(index)
        return leaves

    def find_parent(self, node):
        for key in self.edges.keys():
            if node in self.edges[key]:
                return key
        return None

    def find_children(self, node):
        if node in self.edges.keys():
            return self.edges[node]
        else:
            return []

    def calculate_branch_thickness(self):
        self.assign_thickness(0)

    def assign_thickness(self, node_id):
        children = self.find_children(node_id)
        parent = self.find_parent(node_id)
        diameter = 0
        if len(children) == 0:
            if parent is not None:
                for child_index in range(0, len(self.edges[parent])):
                    if self.edges[parent][child_index] == node_id:
                        self.branch_thickness_dict[node_id] = self.tip_radius
                        diameter = self.tip_radius
                        break
        else:
            for child in children:
                diameter += self.assign_thickness(child)**self.pipe_model_exponent
            diameter = diameter**(1/self.pipe_model_exponent)
            if parent is not None:
                for child_index in range(0,len(self.edges[parent])):
                    if self.edges[parent][child_index] == node_id:
                        self.branch_thickness_dict[node_id] = diameter
                        break
        return diameter

    def generate_urdf(self):
        urdf = minidom.Document()
        robot = urdf.createElement('robot')
        robot.setAttribute('name', "tree%s"%self.tree_id)
        urdf.appendChild(robot)
        self.generate_color_definitions(urdf, robot)
        self.generate_ground(urdf, robot)
        for node_index, _ in enumerate(self.tree_points):
            children = self.find_children(node_index)
            self.generate_spherical_joint(urdf, robot, node_index, children)
            for child in children:
                self.generate_link(urdf, robot, node_index, child)

        self.clean_edge_list()
        tree_string = urdf.toprettyxml(indent='\t')
        save_path_file = "tree%s.urdf" % self.tree_id

        with open(save_path_file, "w") as f:
            f.write(tree_string)

        return self.name_dict, self.edge_list, os.path.abspath(save_path_file)

    def clean_edge_list(self):
        for parent, child in self.edge_list:
            if isinstance(parent, str):
                print("removed (%s, %s)"%(parent, child))
                self.edge_list.remove((parent,child))
                parent_idx = self.name_dict["links"].index(parent) 
                self.edge_list.append((parent_idx,child))
                print("added (%s, %s)"%(parent_idx, child))


    def generate_color_definitions(self, urdf, robot):
        for name, rgba in [("blue", "0 0 0.8 1"), ("green", "0 0.6 0 0.8"), ("brown", "0.3 0.15 0.05 1.0")]:
            material = urdf.createElement('material')
            material.setAttribute('name', name)
            robot.appendChild(material)
            color = urdf.createElement('color')
            color.setAttribute('rgba', rgba)
            material.appendChild(color)

    def add_limits(self, urdf, parent):
        limit1 = urdf.createElement('limit')
        limit1.setAttribute('lower', '-3.1416')
        limit1.setAttribute('upper', '3.1416')
        limit1.setAttribute('effort', '10')
        limit1.setAttribute('velocity', '3')
        parent.appendChild(limit1)

        limit2 = urdf.createElement('limit')
        limit2.setAttribute('lower', '-2.9671')
        limit2.setAttribute('upper', '2.9671')
        limit2.setAttribute('effort', '87')
        limit2.setAttribute('velocity', '2.1750')
        parent.appendChild(limit2)

    def add_dynamics(self, urdf, parent):
        dynamics = urdf.createElement('dynamics')
        dynamics.setAttribute('damping', '10.0')
        parent.appendChild(dynamics)

    def add_safety_controller(self, urdf, parent):
        safety_controller = urdf.createElement('safety_controller')
        safety_controller.setAttribute('k_position', '100.0')
        safety_controller.setAttribute('k_velocity', '40.0')
        safety_controller.setAttribute('soft_lower_limit', '-2.8973')
        safety_controller.setAttribute('soft_upper_limit', '2.8973')
        parent.appendChild(safety_controller)

    def add_inertia(self, urdf, parent):
        inertia = urdf.createElement('inertia')
        inertia.setAttribute('ixx', '0.001')
        inertia.setAttribute('ixy', '0')
        inertia.setAttribute('ixz', '0')
        inertia.setAttribute('iyy', '0.001')
        inertia.setAttribute('iyz', '0')
        inertia.setAttribute('izz', '0.001')
        parent.appendChild(inertia)

    def add_mass(self, urdf, parent):
        mass = urdf.createElement('mass')
        mass.setAttribute('value', '0.001')
        parent.appendChild(mass)

    def add_inertial(self, urdf, parent):
        inertial = urdf.createElement('inertial')
        self.add_mass(urdf, inertial)
        self.add_inertia(urdf, inertial)
        parent.appendChild(inertial)

    def generate_spherical_joint(self, urdf, robot, tree_node, children):
        jointbase = None
        joint_one_offset = [0,0,0.01]
        parent = self.find_parent(tree_node)
        if parent is not None:
            joint_one_offset = self.tree_points[tree_node] - self.tree_points[parent]

        jointbase = urdf.createElement('joint')
        jointbase.setAttribute('name', 'joint%s_base'%tree_node)
        #self.name_dict["joints"].append('joint%s_base'%tree_node)
        jointbase.setAttribute('type', 'fixed')
        joint_parent = urdf.createElement('parent')
        if parent is None:
                joint_parent.setAttribute('link', 'base_link')
        else:
            joint_parent.setAttribute('link', 'link_%s_to_%s'%(parent,tree_node))
        jointbase.appendChild(joint_parent)

        origin = urdf.createElement('origin')
        origin.setAttribute('xyz', '%s %s %s'%(joint_one_offset[0], joint_one_offset[1], joint_one_offset[2]))
        origin.setAttribute('rpy', '0 0 0')
        jointbase.appendChild(origin)

        if len(children) == 0:
            joint_child = urdf.createElement('child')
            joint_child.setAttribute('link', 'link%s_tip'%(tree_node))
            jointbase.appendChild(joint_child)
            robot.appendChild(jointbase)

            link_base = urdf.createElement('link')
            link_base.setAttribute('name', 'link%s_tip'%(tree_node))
            
            visual = urdf.createElement('visual')
            self.add_origin(urdf, visual, [0,0,0], [0,0,0])
            geometry = urdf.createElement('geometry')
            box = urdf.createElement('box')
            box.setAttribute('size', '%s %s %s' % (self.tip_radius, self.tip_radius, self.tip_radius))
            geometry.appendChild(box)
            visual.appendChild(geometry)

            material = urdf.createElement('material')
            material.setAttribute('name', 'green')
            visual.appendChild(material)

            link_base.appendChild(visual)

            collision = urdf.createElement('collision')
            self.add_origin(urdf, collision, [0,0,0], [0,0,0])
            geometry = urdf.createElement('geometry')
            box = urdf.createElement('box')
            box.setAttribute('size', '%s %s %s' % (self.tip_radius, self.tip_radius, self.tip_radius))
            geometry.appendChild(box)
            collision.appendChild(geometry)

            link_base.appendChild(collision)

            robot.appendChild(link_base)

            self.name_dict["links"].append('link%s_tip'%(tree_node)) 

            incoming_edge_name = 'link_%s_to_%s'%(parent,tree_node)
            if incoming_edge_name in self.name_dict["links"]:
                incoming_edge_idx = self.name_dict["links"].index(incoming_edge_name) # extract index of incoming edge
                my_edge_idx = len(self.name_dict["links"])-1 # extract index of added link (last added element)
                self.edge_list.append((incoming_edge_idx, my_edge_idx))
            else:
                print("edge name not found. Adding dirty edge")
                my_edge_idx = len(self.name_dict["links"])-1 # extract index of added link (last added element)
                self.edge_list.append((incoming_edge_name,my_edge_idx))
                parent_idx = self.edge_list.index(incoming_edge_name)

        else:
            joint_child = urdf.createElement('child')
            joint_child.setAttribute('link', 'link%s_base'%(tree_node))
            jointbase.appendChild(joint_child)
            robot.appendChild(jointbase)

            link_base = urdf.createElement('link')
            link_base.setAttribute('name', 'link%s_base'%(tree_node))
            self.add_inertial(urdf, link_base)
            robot.appendChild(link_base) 

        for child in children:
            jointx = urdf.createElement('joint')
            jointx.setAttribute('name', 'joint%s_x_to_%s'%(tree_node,child))
            self.name_dict["joints"].append('joint%s_x_to_%s'%(tree_node,child))
            jointx.setAttribute('type', 'revolute')
            self.add_safety_controller(urdf, jointx)
            joint_parent = urdf.createElement('parent')
            joint_parent.setAttribute('link', 'link%s_base'%tree_node)
            jointx.appendChild(joint_parent)
            #jointx.appendChild(joint_parent)

            joint_child = urdf.createElement('child')
            joint_child.setAttribute('link', 'link%s_x_to_%s'%(tree_node,child))
            jointx.appendChild(joint_child)

            origin = urdf.createElement('origin')
            origin.setAttribute('xyz', '0 0 0')
            origin.setAttribute('rpy', '0 0 0')
            jointx.appendChild(origin)

            axis = urdf.createElement('axis')
            axis.setAttribute('xyz', '1 0 0')
            jointx.appendChild(axis)

            self.add_dynamics(urdf, jointx)
            self.add_limits(urdf, jointx)
            robot.appendChild(jointx)

            linkx = urdf.createElement('link')
            linkx.setAttribute('name', 'link%s_x_to_%s'%(tree_node,child))
            self.add_inertial(urdf, linkx)
            robot.appendChild(linkx)

            jointy = urdf.createElement('joint')
            jointy.setAttribute('name', 'joint%s_y_to_%s' % (tree_node,child))
            self.name_dict["joints"].append('joint%s_y_to_%s'%(tree_node,child))
            jointy.setAttribute('type', 'revolute')
            self.add_safety_controller(urdf, jointy)
            joint_parent = urdf.createElement('parent')
            joint_parent.setAttribute('link', 'link%s_x_to_%s' % (tree_node,child))
            jointy.appendChild(joint_parent)

            joint_child = urdf.createElement('child')
            joint_child.setAttribute('link', 'link%s_y_to_%s' % (tree_node,child))
            jointy.appendChild(joint_child)

            origin = urdf.createElement('origin')
            origin.setAttribute('xyz', '0 0 0')
            origin.setAttribute('rpy', '0 0 0')
            jointy.appendChild(origin)

            axis = urdf.createElement('axis')
            axis.setAttribute('xyz', '0 1 0')
            jointy.appendChild(axis)

            self.add_dynamics(urdf, jointy)
            self.add_limits(urdf, jointy)
            robot.appendChild(jointy)

            linky = urdf.createElement('link')
            linky.setAttribute('name', 'link%s_y_to_%s' % (tree_node,child))
            self.add_inertial(urdf, linky)
            robot.appendChild(linky)

            jointz = urdf.createElement('joint')
            jointz.setAttribute('name', 'joint%s_z_to_%s' % (tree_node,child))
            self.name_dict["joints"].append('joint%s_z_to_%s' % (tree_node,child))
            jointz.setAttribute('type', 'revolute')
            self.add_safety_controller(urdf, jointz)
            joint_parent = urdf.createElement('parent')
            joint_parent.setAttribute('link', 'link%s_y_to_%s' % (tree_node,child))
            jointz.appendChild(joint_parent)

            joint_child = urdf.createElement('child')
            joint_child.setAttribute('link', 'link_%s_to_%s' %(tree_node, child))
            jointz.appendChild(joint_child)

            origin = urdf.createElement('origin')
            origin.setAttribute('xyz', '0 0 0')
            origin.setAttribute('rpy', '0 0 0')
            jointz.appendChild(origin)

            axis = urdf.createElement('axis')
            axis.setAttribute('xyz', '0 0 1')
            jointz.appendChild(axis)

            self.add_dynamics(urdf, jointz)
            self.add_limits(urdf, jointz)
            robot.appendChild(jointz)

    def generate_link(self, urdf, robot, parent, child):
        xyz_offset = (self.tree_points[child] - self.tree_points[parent])
        link_length = np.linalg.norm(xyz_offset)
        xyz_offset = xyz_offset/2
        rpy_rotations = self.calculate_rpy(parent, child)
        cylinder_radius = self.tip_radius
        idx = 0
        while idx < len(self.edges[parent]):
            cylinder_radius = self.branch_thickness_dict[child]
            idx += 1

        link = urdf.createElement('link')
        link.setAttribute('name', 'link_%s_to_%s'%(parent, child))
        self.name_dict["links"].append('link_%s_to_%s'%(parent, child))
        parents_parent = self.find_parent(parent) # extract ID of the parents parent
        if parents_parent is not None:
            incoming_edge_name = "link_%s_to_%s"%(parents_parent, parent)
        else:
            incoming_edge_name = "base_link"

        if incoming_edge_name in self.name_dict["links"]:
            incoming_edge_idx = self.name_dict["links"].index(incoming_edge_name) # extract index of incoming edge
            my_edge_idx = len(self.name_dict["links"])-1 # extract index of added link (last added element)
            self.edge_list.append((incoming_edge_idx, my_edge_idx))
        else:
            print("edge name not found. Adding dirty edge")
            my_edge_idx = len(self.name_dict["links"])-1 # extract index of added link (last added element)
            self.edge_list.append((incoming_edge_name,my_edge_idx))

        visual = urdf.createElement('visual')
        self.add_origin(urdf, visual, xyz_offset, rpy_rotations)
        self.add_geometry_cylinder(urdf, visual, cylinder_radius, link_length)

        material = urdf.createElement('material')
        material.setAttribute('name', 'brown')
        visual.appendChild(material)

        link.appendChild(visual)

        collision = urdf.createElement('collision')
        self.add_origin(urdf, collision, xyz_offset, rpy_rotations)
        self.add_geometry_cylinder(urdf, collision, cylinder_radius, link_length)

        link.appendChild(collision)
        robot.appendChild(link)

    def add_origin(self, urdf, parent, xyz_offset, rpy_rotations):
        origin = urdf.createElement('origin')
        origin.setAttribute('xyz', '%s %s %s' % (xyz_offset[0], xyz_offset[1], xyz_offset[2]))
        origin.setAttribute('rpy', '%s %s %s' % (rpy_rotations[0], rpy_rotations[1], rpy_rotations[2]))
        parent.appendChild(origin)

    def add_geometry_cylinder(self, urdf, parent, cylinder_radius, link_length):
        geometry = urdf.createElement('geometry')
        cylinder = urdf.createElement('cylinder')
        cylinder.setAttribute('radius', '%s' % (cylinder_radius*self.tip_radius))
        cylinder.setAttribute('length', '%s' % link_length)
        geometry.appendChild(cylinder)
        parent.appendChild(geometry)

    # CURRENT THEORY: Isaacgym interprets the rpy rotations as rotation around the GLOBAL x y and z axis
    def calculate_rpy(self, parent, child):
        #Z_tmp = np.array([self.tree_points[child][1], self.tree_points[child][0], self.tree_points[child][2]]) - np.array([self.tree_points[parent][1], self.tree_points[parent][0], self.tree_points[parent][2]]) 
        Z_tmp = self.tree_points[child] - self.tree_points[parent]
        Z = Z_tmp/np.linalg.norm(Z_tmp)

        if Z[2] == 1 and Z[1] == 0 and Z[0] == 0:
            X = np.array([1,0,0])
        else:
            X_tmp = np.cross(Z, np.array([0,0,1])) #np.array([-Z[1]/Z[2], 1, 0])
            X = X_tmp/np.linalg.norm(X_tmp)

        Y_tmp = np.cross(Z, X)
        Y = Y_tmp/np.linalg.norm(Y_tmp)

        R = np.vstack((X,Y,Z))
        R = np.transpose(R)
        #print(R)
        #print(R[2][0])

        rot = Rotation.from_matrix(R)
        rot_eul = rot.as_euler("xyz")

        #print(rot_eul)

        #p = np.arcsin(-R[2][0])
        #r = np.arcsin(R[2][1]/np.cos(p))
        #y = np.arcsin(R[1][0]/np.cos(p))

        #theta = np.arccos((np.trace(R)-1)/2)

        #if np.sin(theta) != 0:
        #    p = -((Y[2]-Z[1])/2*np.sin(theta))*theta
        #    y = ((Z[0]-X[2])/2*np.sin(theta))*theta
        #    r = -((X[1]-Y[0])/2*np.sin(theta))*theta
        #else:
        #    r = 0
        #    p = 0
        #    y = 0

        # X-Z-Y 
        #r = np.arctan2(Y[2], Y[1])
        #y = np.arctan2(-Y[0], np.sqrt(1-Y[0]**2)) #np.arcsin(-Y[0])
        #p = np.arctan2(Z[0], X[0])

        # X-Y-Z (Incorrect order)
        #r = np.arctan2(-Z[1], Z[2])
        #y = np.arctan2(Z[0], np.sqrt(1-Z[0]**2)) #np.arcsin(Z[0])
        #p = np.arctan2(-Y[0], X[0])

        # Y-X-Z (Incorrect order)
        #r = np.arctan2(Z[0], Z[2])
        #y = np.arctan2(-Z[1], np.sqrt(1-Z[1]**2)) #np.arcsin(-Z[1])
        #p = np.arctan2(X[1], Y[1])

        # Y-Z-X
        #r = np.arctan2(-X[2], X[0])
        #y = np.arctan2(X[1], np.sqrt(1-X[1]**2)) #np.arcsin(X[1])
        #p = np.arctan2(-Z[1], Y[1])

        # Z-Y-X
        #r = np.arctan2(X[1], X[0])
        #y = np.arctan2(-X[2], np.sqrt(1-X[2]**2)) #np.arcsin(-X[2])
        #p = np.arctan2(Y[2], Z[2])

        # Z-X-Y
        #r = np.arctan2(-Y[0], Y[1])
        #y = np.arctan2(Y[2], np.sqrt(1-Y[2]**2)) #np.arcsin(Y[2])
        #p = np.arctan2(-X[2], Z[2])

        # X-Z-X
        #r = np.arctan2(X[2], X[1])
        #y = np.arctan2(np.sqrt(1-X[0]**2), X[0])
        #p = np.arctan2(Z[0], -Y[0])

        # X-Y-X
        #r = np.arctan2(X[1], -X[2])
        #y = np.arctan2(np.sqrt(1-X[0]**2), X[0])
        #p = np.arctan2(Y[0], Z[0])

        # Y-X-Y
        #r = np.arctan2(Y[0], Y[2])
        #y = np.arctan2(np.sqrt(1-Y[1]**2), Y[1])
        #p = np.arctan2(X[1], -Y[2])

        # Y-Z-Y
        #r = np.arctan2(Y[2], -Y[0])
        #y = np.arctan2(np.sqrt(1-Y[1]**2), Y[1])
        #p = np.arctan2(Z[1], X[1])

        # Z-Y-Z
        #r = np.arctan2(Z[1], Z[0])ww
        #y = np.arctan2(np.sqrt(1-Z[2]**2), Z[2])
        #p = np.arctan2(Y[2], -X[2])

        # Z-X-Z
        #r = np.arctan2(Z[0], -Z[1])
        #y = np.arctan2(np.sqrt(1-Z[2]**2), Z[2])
        #p = np.arctan2(X[2], Y[2])

        #offsets = self.tree_points[child] - self.tree_points[parent]
        #offsets = offsets/2
        #if offsets[2] < 0:
        #    if offsets[0] < 0:
        #        r = - (self.calculate_angle(offsets[1], np.linalg.norm(offsets)) - np.pi/2)
        #        p = - self.calculate_angle(offsets[0], np.linalg.norm(offsets))
        #        y = 0
        #    else:
        #        r = - (self.calculate_angle(offsets[1], np.linalg.norm(offsets)) + np.pi/2) 
        #        p = - self.calculate_angle(offsets[0], np.linalg.norm(offsets))
        #        y = 0
        #else:
        #    r = -self.calculate_angle(offsets[1], np.linalg.norm(offsets))
        #    p = self.calculate_angle(offsets[0], np.linalg.norm(offsets))
        #    y = 0
        r = rot_eul[0]
        p = rot_eul[1]
        y = rot_eul[2]

        return [r,p,y]
        #return [r,p,-y]
        #return [r,-p,y]
        #return [r,-p,-y]
        #return [-r,p,y]
        #return [-r,p,-y]
        #return [-r,-p,y]
        #return [-r,-p,-y]

    def calculate_angle(self, delta, dist):
        rot_vec = 0
        if dist != 0:
            rot_vec = np.arcsin(delta/dist)
        #    if offsets[0] < 0:
        #        if offsets[1] > 0:
        #            rot_vec = rot_vec - np.pi/2
        #        else:
        #            rot_vec += np.pi/2
        return rot_vec

    def generate_ground(self, urdf, robot):
        link = urdf.createElement('link')
        link.setAttribute('name', 'base_link')
        robot.appendChild(link)
        self.name_dict["links"].append("base_link")

        visual = urdf.createElement('visual')
        link.appendChild(visual)
        origin = urdf.createElement('origin')
        origin.setAttribute('xyz', '0 0 0')
        origin.setAttribute('rpy', '0 0 0')
        visual.appendChild(origin)
        geometry = urdf.createElement('geometry')
        visual.appendChild(geometry)
        box = urdf.createElement('box')
        box.setAttribute('size', '%s %s 0.02'%(1*self.scaling, 1*self.scaling))
        geometry.appendChild(box)
        material = urdf.createElement('material')
        material.setAttribute('name', 'green')
        visual.appendChild(material)

        collision = urdf.createElement('collision')
        link.appendChild(collision)
        originc = urdf.createElement('origin')
        originc.setAttribute('xyz', '0 0 0')
        originc.setAttribute('rpy', '0 0 0')
        collision.appendChild(originc)
        geometryc = urdf.createElement('geometry')
        visual.appendChild(geometryc)
        boxc = urdf.createElement('box')
        boxc.setAttribute('size', '%s %s 0.02' % (1 * self.scaling, 1 * self.scaling))
        geometryc.appendChild(boxc)
        collision.appendChild(geometryc)

    def calc_edge_tuples(self):
        edge_tuples = []
        for parent in self.edges.keys():
            for child in self.edges[parent]:
                edge_tuples.append((parent, child))
        return edge_tuples

    # has to be executed after the urdf was created
    def generate_yaml(self):
        file_object = {}
        file_object["scene"] = {}
        file_object["scene"]["n_envs"] = self.env_num
        file_object["scene"]["es"] = 1
        file_object["scene"]["gui"] = GUI_ON

        file_object["scene"]["cam"] = {}
        file_object["scene"]["cam"]["cam_pos"] = [5, 0, 5]
        file_object["scene"]["cam"]["look_at"] = [0, 0, 0]

        file_object["scene"]["gym"] = {}
        file_object["scene"]["gym"]["dt"] = SIMULATION_STEP_SIZE
        file_object["scene"]["gym"]["substeps"] = 2
        file_object["scene"]["gym"]["up_axis"] = "z"
        file_object["scene"]["gym"]["type"] = "physx"
        file_object["scene"]["gym"]["use_gpu_pipeline"] = True

        file_object["scene"]["gym"]["physx"] = {}
        file_object["scene"]["gym"]["physx"]["solver_type"] = 1
        file_object["scene"]["gym"]["physx"]["num_position_iterations"] = 8
        file_object["scene"]["gym"]["physx"]["num_velocity_iterations"] = 1
        file_object["scene"]["gym"]["physx"]["rest_offset"] = 0.0
        file_object["scene"]["gym"]["physx"]["contact_offset"] = 0.001
        file_object["scene"]["gym"]["physx"]["friction_offset_threshold"] = 0.001
        file_object["scene"]["gym"]["physx"]["friction_correlation_distance"] = 0.0005
        file_object["scene"]["gym"]["physx"]["use_gpu"] = True

        file_object["scene"]["gym"]["device"] = {}
        file_object["scene"]["gym"]["device"]["compute"] = 0
        file_object["scene"]["gym"]["device"]["graphics"] = 0

        file_object["scene"]["gym"]["plane"] = {}
        file_object["scene"]["gym"]["plane"]["dynamic_friction"] = 0.4
        file_object["scene"]["gym"]["plane"]["static_friction"] = 0
        file_object["scene"]["gym"]["plane"]["restitution"] = 0

        file_object["tree"] = {}

        file_object["tree"]["asset_options"] = {}
        file_object["tree"]["asset_options"]["fix_base_link"] = True
        file_object["tree"]["asset_options"]["flip_visual_attachments"] = True
        file_object["tree"]["asset_options"]["armature"] = 0.01
        file_object["tree"]["asset_options"]["max_linear_velocity"] = 100.0
        file_object["tree"]["asset_options"]["max_angular_velocity"] = 40.0
        file_object["tree"]["asset_options"]["disable_gravity"] = True

        file_object["tree"]["attractor_props"] = {}
        file_object["tree"]["attractor_props"]["stiffness"] = 0
        file_object["tree"]["attractor_props"]["damping"] = 0

        file_object["tree"]["shape_props"] = {}
        file_object["tree"]["shape_props"]["thickness"] = 1e-3

        file_object["tree"]["dof_props"] = {}
        stiffness_list = []
        for name in self.name_dict["joints"]:
            name_lst = name.split("_")
            joint_idx = int(name_lst[0][5:])
            child_idx = int(name_lst[-1])
            parent = self.find_parent(joint_idx)
            
            length = np.linalg.norm(self.tree_points[child_idx] - self.tree_points[joint_idx])
            #print("lenght for node %s: %s"%(joint_idx,length))

            thickness = self.branch_thickness_dict[child_idx]*2 #use thickness of outgoing edge for stiffness calc

            stiffness_factor = thickness**4 #/(length**3) # <-- lenght factor should be taken care of by physical simulation from isaacgym
            stiffness = STIFFNESS_BASE * stiffness_factor
            #print(stiffness)
            stiffness_list.append(stiffness)

        damping_list = [25] * (len(self.name_dict["joints"])) 
        file_object["tree"]["dof_props"]["stiffness"] = stiffness_list #[30] * (len(self.name_dict["joints"])) # -len(self.tree_points)
        file_object["tree"]["dof_props"]["damping"] = damping_list # -len(self.tree_points)
        file_object["tree"]["dof_props"]["effort"] = [87] * (len(self.name_dict["joints"])) # -len(self.tree_points)

        with open("tree%s.yaml"%self.tree_id, "w") as f:
            yaml.dump(file_object, f)

        return os.path.abspath("tree%s.yaml"%self.tree_id), stiffness_list, damping_list



#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#tg = TreeGenerator(max_steps=10000, att_pts_max=1000, da=50, dt=0.25, step_width=0.5, offset=[-0.5, -0.5, 0.25], scaling=5, max_tree_points=200, tip_radius=0.1, z_strech=0.25, x_strech=2, y_strech=2)
#tg.generate_tree()
#ax.scatter(tg.tree_points[:,0], tg.tree_points[:,1], tg.tree_points[:,2])
#plt.show()
#tg.calculate_branch_thickness()
#tg.generate_urdf()
#print(len(tg.tree_points))
