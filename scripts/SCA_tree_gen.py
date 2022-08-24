import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
from xml.dom import minidom

BATCH_SIZE = 1

def sphere(pt, a=1.0, b=1.0, c=1.0):
    r = (pt[0]-0.5)**2*a + (pt[1]-0.5)**2*b + (pt[2]-0.5)**2*c
    return r <= 0.25


class TreeGenerator(object):

    def __init__(self, att_pts_max, scaling, offset, da, dt, max_steps, step_width, max_tree_points, tip_radius, att_env_shape_funct=sphere, tree_id=0, pipe_model_exponent=2, att_pts_min=None, x_strech=1, y_strech=1, z_strech=1, step_width_scaling=1):
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
        self.names =

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


        tree_string = urdf.toprettyxml(indent='\t')
        save_path_file = "tree%s.urdf" % self.tree_id

        with open(save_path_file, "w") as f:
            f.write(tree_string)

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
        joint_one_offset = [0,0,0.01]
        parent = self.find_parent(tree_node)
        if parent is not None:
            joint_one_offset = self.tree_points[tree_node] - self.tree_points[parent]

        jointx = urdf.createElement('joint')
        jointx.setAttribute('name', 'joint%s_x'%tree_node)
        jointx.setAttribute('type', 'revolute')
        self.add_safety_controller(urdf, jointx)
        joint_parent = urdf.createElement('parent')
        if parent is None:
            joint_parent.setAttribute('link', 'base_link')
        else:
            joint_parent.setAttribute('link', 'link_%s_to_%s'%(parent,tree_node))
        jointx.appendChild(joint_parent)
        jointx.appendChild(joint_parent)

        joint_child = urdf.createElement('child')
        joint_child.setAttribute('link', 'link%s_x'%tree_node)
        jointx.appendChild(joint_child)

        origin = urdf.createElement('origin')
        origin.setAttribute('xyz', '%s %s %s'%(joint_one_offset[0], joint_one_offset[1], joint_one_offset[2]))
        origin.setAttribute('rpy', '0 0 0')
        jointx.appendChild(origin)

        axis = urdf.createElement('axis')
        axis.setAttribute('xyz', '1 0 0')
        jointx.appendChild(axis)

        self.add_dynamics(urdf, jointx)
        self.add_limits(urdf, jointx)
        robot.appendChild(jointx)

        linkx = urdf.createElement('link')
        linkx.setAttribute('name', 'link%s_x'%tree_node)
        self.add_inertial(urdf, linkx)
        robot.appendChild(linkx)

        jointy = urdf.createElement('joint')
        jointy.setAttribute('name', 'joint%s_y' % tree_node)
        jointy.setAttribute('type', 'revolute')
        self.add_safety_controller(urdf, jointy)
        joint_parent = urdf.createElement('parent')
        joint_parent.setAttribute('link', 'link%s_x'%tree_node)
        jointy.appendChild(joint_parent)

        joint_child = urdf.createElement('child')
        joint_child.setAttribute('link', 'link%s_y' % tree_node)
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
        linky.setAttribute('name', 'link%s_y' % tree_node)
        self.add_inertial(urdf, linky)
        robot.appendChild(linky)

        for child in children:
            jointz = urdf.createElement('joint')
            jointz.setAttribute('name', 'joint%s_z_to_%s' % (tree_node,child))
            jointz.setAttribute('type', 'revolute')
            self.add_safety_controller(urdf, jointz)
            joint_parent = urdf.createElement('parent')
            joint_parent.setAttribute('link', 'link%s_y' % tree_node)
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

    def calculate_rpy(self, parent, child):
        Z_tmp = self.tree_points[child] - self.tree_points[parent]
        Z = Z_tmp/np.linalg.norm(Z_tmp)

        X_tmp = np.array([-Z[1]/Z[2], 1, 0])
        X = X_tmp/np.linalg.norm(X_tmp)

        Y_tmp = np.cross(Z, X)
        Y = Y_tmp/np.linalg.norm(Y_tmp)

        r = np.arctan2(-Z[1], Z[2])
        p = np.arcsin(Z[0])
        y = np.arctan2(-Y[0], X[0])

        return [r,p,y]

    def calculate_rotational_vector(self, offsets):
        # offset[1] = gegenkathete
        # offset[0] = ankathete
        rot_vec = 0
        if offsets[0] != 0:
            rot_vec = np.arctan(offsets[1]/offsets[0])
            if offsets[0] < 0:
                if offsets[1] > 0:
                    rot_vec = rot_vec-np.pi/2
                else:
                    rot_vec += np.pi/2 #rotate everything by 180 degrees

        """
        else:
            rot_vec = 0
        if offsets[0] < 0:
            if offsets[1] < 0:
                rot_vec += 3.14159
            elif offsets[1] == 0:
                rot_vec = 4.71239
            else:
                rot_vec += 4.71239
        elif offsets[0] == 0:
            if offsets[1] < 0:
                rot_vec = 3.14159
            else:
                rot_vec = 0
        else:
            if offsets[1] < 0:
                rot_vec += 1.5708
            elif offsets[1] == 0:
                rot_vec = 1.5708
            else:
                rot_vec += 0
        """
        return rot_vec


    def generate_ground(self, urdf, robot):
        link = urdf.createElement('link')
        link.setAttribute('name', 'base_link')
        robot.appendChild(link)

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


#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#tg = TreeGenerator(max_steps=10000, att_pts_max=1000, da=50, dt=0.25, step_width=0.5, offset=[-0.5, -0.5, 0.25], scaling=5, max_tree_points=200, tip_radius=0.1, z_strech=0.25, x_strech=2, y_strech=2)
#tg.generate_tree()
#ax.scatter(tg.tree_points[:,0], tg.tree_points[:,1], tg.tree_points[:,2])
#plt.show()
#tg.calculate_branch_thickness()
#tg.generate_urdf()
#print(len(tg.tree_points))
