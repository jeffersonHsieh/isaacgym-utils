import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
from xml.dom import minidom
import os
from scipy.spatial.transform import Rotation

import yaml

BATCH_SIZE = 1 # size of batches for random point generation. Lower values might speed up the process, but lead to variations in the final number of attraction points
STIFFNESS_BASE = 220000 # factor used to calculate stiffness. 
SIMULATION_STEP_SIZE = 0.01 # internal value for isaacgym simulation

def sphere(pt, a=1.0, b=1.0, c=1.0):
    r = (pt[0]-0.5)**2*a + (pt[1]-0.5)**2*b + (pt[2]-0.5)**2*c
    return r <= 0.25


class TreeGenerator(object):

    def __init__(self,path, att_pts_max, scaling, offset, da, dt, max_steps, step_width, max_tree_points, tip_radius, gui_on, att_env_shape_funct=sphere, tree_id=0, pipe_model_exponent=2, att_pts_min=None, x_strech=1, y_strech=1, z_strech=1, step_width_scaling=1, env_num=1):
        """
        Sets up the tree generator with all of its parameters.
        :param att_pts_max: maximum number of attraction points. for further info see initialize_att_pts
        :param scaling: defines tree crown size. for further info see initialize_att_pts
        :param offset: defines tree crown position. for further info see initialize_att_pts
        :param da: attraction distance
        :param dt: termination distance
        :param max_steps: maximum amount of steps taken to generate tree model
        :param tip_radius: radius of the branch tips in the generated tree.
        :param gui_on: 1 or 0. Determines whether Isaac Gym is run with gui or without it.
        :param step_width: the distance that parent and child nodes have to each other
        :param max_tree_points: algorithm stops after creating the specified number of tree points
        :param att_env_shape_funct: defines tree crown shape. for further info see initialize_att_pts
        :param gui_on: controls wether or not isaac gym starts with or without gui (1 is on 0 is off)
        :param tree_id: index of generated tree
        :param pipe_model_exponent: exponent for the pipe model used to generate the branch thickness. should be 2 or 3. Higher values lead to lower increas in radius between segments
        :param att_pts_min: minimum number of attraction points to be generated. Irellevant if BATCH_SIZE is 1
        :param x_strech, y_strech, z_strech: strech values used to change the shape of the attraction point cloud. Each streches the shape along its respective axis
        :param step_width_scaling: factor that the step width is multiplied with after each point generation round.
        :param env_num: number of environments to be run in parallel
        """
        self.tree_points = np.array([[0,0,0]]) # initilize root tree point
        if att_pts_min is None:
            att_pts_min = att_pts_max
        self.att_pts = self.initialize_att_pts(att_pts_max, att_pts_min, lambda pt: att_env_shape_funct(pt, x_strech, y_strech, z_strech), scaling, offset)
        min_da = self.min_da() # calculate the minimum da that still allows the root point to be attracted to at least one attraction point
        if da < min_da: # if da is so small that no tree generation would happen, increase it to minimum value
            da = min_da
        self.da = da
        self.dt = dt
        self.max_steps = max_steps
        self.step_width = step_width
        self.closest_tp = self.initialize_closest_tp_list() # closest tp is a list that holds the closest tree point for every attraction point (index of attraction point is the index on this list) as well as the distance between ap and tp
        self.edges = {} # dictionary with keys as parents referencing np.arrays of children indices
        self.branch_thickness_dict = {} # dictionar holding the branch thickness of the incoming edge. key: tp id, value: thickness
        self.max_tree_points = max_tree_points
        self.tip_radius = tip_radius
        self.tree_id = tree_id
        self.scaling = scaling
        self.pipe_model_exponent = pipe_model_exponent
        self.step_width_scaling = step_width_scaling

        self.name_dict = {"joints":[], "links":[]} # dictionary holding the names of the generated urdf elements
        self.env_num = env_num
        self.edge_list = []
        self.gui_on = gui_on

    def infer_edges_from_list(self):
        """
        this function should allow us to generate the edges dictionary required by many processing functions
        in this class from a list of edge tuples. It is assumed that edge tuples are unique and that no tuple
        (a, b) exists, if tuple (b, a) is also part of the edge tuples list.

        further it is assumed the root node has index 0 and that the edge tuples reference indices that align
        with the tree points array
        """
        frontier = [0] # assuming root node has index 0
        edges_dict = {}
        edge_list = self.edge_list
        while len(frontier) > 0:
            current_node = frontier.pop()
            children = []

            for node_a, node_b in enumerate(edge_list):
                if node_a == current_node:
                    children.append(node_b)
                    frontier.append(node_a)
                    edge_list.remove((node_a, node_b))
                if node_b == current_node:
                    children.append(node_a)
                    frontier.append(node_b)
                    edge_list.remove((node_a, node_b))

            edges_dict[current_node] = np.array(children)

        self.edges = edges_dict



    def min_da(self):
        """
        calculates the minimum attraction distance necessary to allow tree generation
        """
        min_da = None
        for point in self.att_pts:
            if min_da is None:
                min_da = np.linalg.norm(point)
            else:
                if min_da > np.linalg.norm(point):
                    min_da = np.linalg.norm(point)
        return math.ceil(min_da)


    def initialize_closest_tp_list(self):
        """
        initializes the list holding the closest tp for each ap. In the beginning that is always tp0
        :return: initialized list of point_index-distance tuples
        """
        closest_tp = []
        for ap in self.att_pts:
            closest_tp.append((0,np.linalg.norm(ap-[0,0,0])))
        return closest_tp

    def update_closest_tp_list(self, new_points, point_index):
        """
        :param new_points: list of newly generated points as cartesian coordinates
        :param point_index: first index of the newly generated points
        """
        to_delete = []
        for new_point in new_points:
            ap_index = 0 # running index to reference attraction points
            for ap in self.att_pts:
                if np.linalg.norm(ap-new_point) < self.closest_tp[ap_index][1]: #check if a newly generated point is closer to given attraction point than the closest 'old' tree point
                    if np.linalg.norm(ap-new_point) <= self.dt: # if so check if new point is within termination range
                        to_delete.append(ap_index) # if so then mark attraction point to be deleted
                    else:
                        self.closest_tp[ap_index] = (point_index, np.linalg.norm(ap-new_point)) # if no deletion took place update the closest tp list
                ap_index += 1
            point_index += 1 # increase the point index to move on to next point

        deleted=0 # term to adjust for index shifting
        to_delete = list(set(to_delete)) # Remove duplicates in delete list
        to_delete.sort() # order delete list. this way we delete the small indices first, allowing us to easily adjust delete indices
        for ap_index in to_delete:
            self.att_pts = np.delete(self.att_pts, ap_index-deleted, 0) # delete attraction point. adjust delete index by the number of already deleted lower indices
            self.closest_tp.pop(ap_index-deleted) # delete entry in closest_tp list. adjust delete index by the number of already deleted lower indices
            deleted+=1

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
        while len(ret_pts)/3 + BATCH_SIZE < att_pts_max and len(ret_pts)/3 < att_pts_min: # repeat until we generated a total number of points larger than the minimum att_pts number and smaller than the maximum number
            rng = np.random.default_rng() # setup rng
            pts = rng.random((BATCH_SIZE,3)) # generate a batch of random points
            for pt in pts:
                if att_env_shape_funct(pt): # check if the given point is within the attraction point envelope. If so add it to the ap array
                    if initial:
                        ret_pts = (pt+offset)*scaling # offset and scaling is applied
                        initial = False
                    else:
                        ret_pts = np.concatenate((ret_pts, (pt+offset)*scaling), axis=0) # offset and scaling is applied
        ret_pts = np.reshape(ret_pts, (int(len(ret_pts)/3),3))
        return ret_pts

    def generate_sv_sets(self):
        """
        generates a dictionary containing lists of attraction points attracting a given tree point. keys:tree point index, value:list of attraction points
        :returns: dictionary of attraction point lists. key: tree point index, value: list of attraction point indices
        """
        sv_sets = {}
        ap_index = 0 # running attraction point index used to reference the ap's
        for candidate in self.closest_tp: #only tree points that are closest to at least one attraction point and therefore in the closest_tp list need to be considered
            if candidate[1] <= self.da:
                if candidate[0] not in sv_sets.keys(): # if we encounter an entirely new tp index, add a dictionary entry
                    sv_sets[candidate[0]] = [ap_index]
                else:
                    sv_sets[candidate[0]].append(ap_index) # otherwise add the attraction point to the list of ap's
            ap_index += 1
        return sv_sets

    def generate_tree(self):
        """
        main function for generating tree points. Generates tree points, until either no attraction points remain, a
        maximum number of point generation rounds is reached or ideally until the expected number of tree points was
        generated.
        :return: list of tree points.
        """
        i = 0
        while len(self.att_pts) >= 1 and i < self.max_steps and len(self.tree_points) < self.max_tree_points:
            sv_sets = self.generate_sv_sets() # generate the attraction point sets, that are in play this round
            new_tps = [] # list of tree points to be generated this round as list of xyz coordinates
            point_index = len(self.tree_points) # point index is assigned as highest index so far +1
            for key in sv_sets.keys():
                new_tps.append(self.generate_new_point(key, sv_sets[key])) # generate new tree point. self.edges and self.tree_points is updated in this function
                if len(self.tree_points) > self.max_tree_points: # stop immediately after the max number of tree points was reaches
                    break
            self.update_closest_tp_list(new_tps, point_index) # update the closest tp list
            self.step_width = self.step_width * self.step_width_scaling # reduce step width
            i += 1
        return self.tree_points # return tree points

    def generate_new_point(self, tp_index, sv_set):
        """
        function that generates one new tree point with the help of the parent index and the set of active attraction points
        :param tp_index: index of the parent of the tree point to be generated
        :param sv_set: list of attraction points as indices
        :return:
        """
        active_att_pts = self.att_pts[sv_set] # retrieve coordinates of attraction points (list indexing)
        tp = self.tree_points[tp_index] # retrieve xyz coordinates of parent
        vec = np.array([0,0,0])
        for ap in active_att_pts:
            tmp = (ap - tp)/np.linalg.norm((ap - tp)) # normalized vector between given attraction point and parent
            vec = vec + tmp # sum all tmp vectors
        vec = vec/np.linalg.norm(vec) # normalize result
        new_tp = tp + self.step_width*vec # see formula for SCA point generation

        self.tree_points = np.vstack((self.tree_points, new_tp)) # add new tree point
        if tp_index in self.edges.keys(): # add edge
            self.edges[tp_index] = np.append(self.edges[tp_index], (len(self.tree_points) - 1))
        else:
            self.edges[tp_index] = np.array([(len(self.tree_points) - 1)])
        return new_tp # return xyz coordinates of new tree point

    def find_leaves(self):
        """
        utility function for finding nodes without children
        returns: list of leaf indices
        """
        tree_node_indices = range(0,len(self.tree_points)-1)
        leaves = []
        for index in tree_node_indices:
            if len(self.find_children(index)) == 0:
                leaves.append(index)
        return leaves

    def find_parent(self, node):
        """
        utility function for finding parent of given node. If no parent exists (root node) this function returns None
        :param node: index of node we want to find the parent of
        :return: index of parent or None if no parent exists
        """
        for key in self.edges.keys():
            if node in self.edges[key]:
                return key
        return None

    def find_children(self, node):
        """
        utility funtction to find the children of a node. If no children exist returns an empty list
        :param node: index of node we want to find children of
        :return: list of child indices (might be empty)
        """
        if node in self.edges.keys():
            return self.edges[node]
        else:
            return []

    def calculate_branch_thickness(self):
        """
        wrapper function that starts recursive branch radius assignment
        """
        self.assign_thickness(0)

    def assign_thickness(self, node_id):
        """
        recursively adds the branch radius of given node and all of its decendants.
        :param node_id: index of the node that we want to assign a radius to
        """
        children = self.find_children(node_id)
        parent = self.find_parent(node_id)
        radius = 0
        if len(children) == 0: # if no children are found assign tip thickness to edge between parent and given node. recursion break condition
            if parent is not None:
                for child_index in range(0, len(self.edges[parent])):
                    if self.edges[parent][child_index] == node_id: # once we found the correct edge, assign radius
                        self.branch_thickness_dict[node_id] = self.tip_radius
                        radius = self.tip_radius
                        break
        else:
            for child in children: # assign radius recursively for every child
                radius += self.assign_thickness(child)**self.pipe_model_exponent # take the results to the power of pipe_model_exponent and sum them
            radius = radius**(1/self.pipe_model_exponent) # take the nth root of the result where n is the pipe_model_exponent
            if parent is not None: # assign the result to the corresponding edge
                for child_index in range(0,len(self.edges[parent])):
                    if self.edges[parent][child_index] == node_id:
                        self.branch_thickness_dict[node_id] = radius
                        break
        return radius # return calculated radius for recursive calculation

    def generate_urdf(self):
        """
        main function to generate the urdf with. calls all other urdf creation functions
        :returns name_dict: dictionary of all relevant names (links and joints). keys: 'links' and 'joints', values: lists of names
        :returns edge_list: list of index tuples. Indices correspond with name dictionary lists
        :returns urdf_path: the path to the generated urdf file
        """
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
        save_path_file = "%s[%s]tree%s.urdf" % (self.path, self.att_pts_max, self.tree_id)

        with open(save_path_file, "w") as f:
            f.write(tree_string)

        return self.name_dict, self.edge_list, os.path.abspath(save_path_file)

    def clean_edge_list(self):
        for parent, child in self.edge_list:
            """
            removes all instances of string parents in the edge list. Translates them to index of parent string. String
            parents can occur when temporarily adding "dirty edges" at time of edge list generation. 
            """
            if isinstance(parent, str):
                print("removed (%s, %s)"%(parent, child))
                self.edge_list.remove((parent,child))
                parent_idx = self.name_dict["links"].index(parent) 
                self.edge_list.append((parent_idx,child))
                print("added (%s, %s)"%(parent_idx, child))


    def generate_color_definitions(self, urdf, robot):
        """
        generates color definition section of urdf
        """
        for name, rgba in [("blue", "0 0 0.8 1"), ("green", "0 0.6 0 0.8"), ("brown", "0.3 0.15 0.05 1.0")]:
            material = urdf.createElement('material')
            material.setAttribute('name', name)
            robot.appendChild(material)
            color = urdf.createElement('color')
            color.setAttribute('rgba', rgba)
            material.appendChild(color)

    def add_limits(self, urdf, parent):
        """
        adds limits to urdf element
        :param urdf: the urdf instance
        :param parent: the parent that section is added to
        """
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
        """
        adds dynamics to urdf element
        :param urdf: the urdf instance
        :param parent: the parent that section is added to
        """
        dynamics = urdf.createElement('dynamics')
        dynamics.setAttribute('damping', '10.0')
        parent.appendChild(dynamics)

    def add_safety_controller(self, urdf, parent):
        """
        adds safety controller to urdf element
        :param urdf: the urdf instance
        :param parent: the parent that section is added to
        """
        safety_controller = urdf.createElement('safety_controller')
        safety_controller.setAttribute('k_position', '100.0')
        safety_controller.setAttribute('k_velocity', '40.0')
        safety_controller.setAttribute('soft_lower_limit', '-2.8973')
        safety_controller.setAttribute('soft_upper_limit', '2.8973')
        parent.appendChild(safety_controller)

    def add_inertia(self, urdf, parent):
        """
        adds inertia to urdf element
        :param urdf: the urdf instance
        :param parent: the parent that section is added to
        """
        inertia = urdf.createElement('inertia')
        inertia.setAttribute('ixx', '0.001')
        inertia.setAttribute('ixy', '0')
        inertia.setAttribute('ixz', '0')
        inertia.setAttribute('iyy', '0.001')
        inertia.setAttribute('iyz', '0')
        inertia.setAttribute('izz', '0.001')
        parent.appendChild(inertia)

    def add_mass(self, urdf, parent):
        """
        adds mass to urdf element
        :param urdf: the urdf instance
        :param parent: the parent that section is added to
        """
        mass = urdf.createElement('mass')
        mass.setAttribute('value', '0.001')
        parent.appendChild(mass)

    def add_inertial(self, urdf, parent):
        """
        adds inertial to urdf element
        :param urdf: the urdf instance
        :param parent: the parent that section is added to
        """
        inertial = urdf.createElement('inertial')
        self.add_mass(urdf, inertial)
        self.add_inertia(urdf, inertial)
        parent.appendChild(inertial)

    def generate_spherical_joint(self, urdf, robot, tree_node, children):
        """
        urdf generation: generates all sperical joints that have a given tree node as parent
        :param urdf: urdf instance
        :param robot: urdf robot element we are working on
        :param tree_node: the tree node we want to generate joints for. given as index
        :param children: the children of the given tree node
        """
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
        """
        urdf generation: generates a link between parent and child
        :param urdf: urdf instance
        :param robot: urdf robot element we are working on
        :param parent: id of the parent element (int)
        :param child: id of the child element (int)
        """
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
        """
        adds origin segment to a given element
        :param urdf: urdf instance we are working on
        :param parent: element we are adding the origin section to
        :param xyz_offset: list of xyz coordinates representing the offset
        :param rpy_rotations: list of roll pitch and yaw values representing the elements rotation
        """
        origin = urdf.createElement('origin')
        origin.setAttribute('xyz', '%s %s %s' % (xyz_offset[0], xyz_offset[1], xyz_offset[2]))
        origin.setAttribute('rpy', '%s %s %s' % (rpy_rotations[0], rpy_rotations[1], rpy_rotations[2]))
        parent.appendChild(origin)

    def add_geometry_cylinder(self, urdf, parent, cylinder_radius, link_length):
        """
        adds the geometric section for a cylinder to a given parent element
        :param urdf: the urdf instance we are working on
        :param parent: the parent element we want to add the geometry section to
        :param cylinder_radius: radius of the cylinder
        :param link_length: length of the cylinder
        """
        geometry = urdf.createElement('geometry')
        cylinder = urdf.createElement('cylinder')
        cylinder.setAttribute('radius', '%s' % (cylinder_radius)) #*self.tip_radius <- why was that here?
        cylinder.setAttribute('length', '%s' % link_length)
        geometry.appendChild(cylinder)
        parent.appendChild(geometry)

    def calculate_rpy(self, parent, child):
        """
        calulates the rpy needed to orient a link to go from parent to child
        :param parent: tree point index of the parent
        :param child: tree point index of the child
        :returns: list of rpy (in that order) values
        """
        Z_tmp = self.tree_points[child] - self.tree_points[parent]
        Z = Z_tmp/np.linalg.norm(Z_tmp)

        if Z[2] == 1 and Z[1] == 0 and Z[0] == 0:
            X = np.array([1,0,0])
        else:
            X_tmp = np.cross(Z, np.array([0,0,1]))
            X = X_tmp/np.linalg.norm(X_tmp)

        Y_tmp = np.cross(Z, X)
        Y = Y_tmp/np.linalg.norm(Y_tmp)

        R = np.vstack((X,Y,Z))
        R = np.transpose(R)

        rot = Rotation.from_matrix(R)
        rot_eul = rot.as_euler("xyz")

        r = rot_eul[0]
        p = rot_eul[1]
        y = rot_eul[2]

        return [r,p,y]

    def generate_ground(self, urdf, robot):
        """
        adds ground section to urdf
        :param urdf: urdf instance we are working on
        :param robot: robot section that ground is added to
        """
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
        """
        makes a list of edge tuples out of the edges dictionary
        """
        edge_tuples = []
        for parent in self.edges.keys():
            for child in self.edges[parent]:
                edge_tuples.append((parent, child))
        return edge_tuples

    # has to be executed after the urdf was created
    def generate_yaml(self):
        """
        generates the yaml file for the generated tree. Has to be executed after the urdf was created
        :returns path: path to the yaml file
        :returns stiffness list: list of stiffness values for all joints
        :returns damping list: list of damping values for all joints
        """
        file_object = {}
        file_object["scene"] = {}
        file_object["scene"]["n_envs"] = self.env_num
        file_object["scene"]["es"] = 1
        file_object["scene"]["gui"] = self.gui_on

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

            thickness = self.branch_thickness_dict[child_idx]*2 #use thickness of outgoing edge for stiffness calc

            stiffness_factor = thickness**4
            stiffness = STIFFNESS_BASE * stiffness_factor
            stiffness_list.append(stiffness)

        stiffness_list[0] = stiffness_list[0] * 2
        damping_list = [25] * (len(self.name_dict["joints"])) 
        file_object["tree"]["dof_props"]["stiffness"] = stiffness_list #[30] * (len(self.name_dict["joints"])) # -len(self.tree_points)
        file_object["tree"]["dof_props"]["damping"] = damping_list # -len(self.tree_points)
        file_object["tree"]["dof_props"]["effort"] = [87] * (len(self.name_dict["joints"])) # -len(self.tree_points)

        location = "%s[%s]tree%s.yaml" % (self.path, self.att_pts_max, self.tree_id)
        with open(location, "w") as f:
            yaml.dump(file_object, f)

        return os.path.abspath(location), stiffness_list, damping_list



#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#tg = TreeGenerator(max_steps=10000, att_pts_max=1000, da=50, dt=0.25, step_width=0.5, offset=[-0.5, -0.5, 0.25], scaling=5, max_tree_points=200, tip_radius=0.1, z_strech=0.25, x_strech=2, y_strech=2)
#tg.generate_tree()
#ax.scatter(tg.tree_points[:,0], tg.tree_points[:,1], tg.tree_points[:,2])
#plt.show()
#tg.calculate_branch_thickness()
#tg.generate_urdf()
#print(len(tg.tree_points))
