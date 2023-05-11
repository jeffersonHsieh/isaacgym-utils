import argparse

import numpy as np
from numpy import save 
from autolab_core import YamlConfig, RigidTransform
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import itertools

from scipy.spatial.transform import Rotation


from isaacgym import gymapi
from isaacgym_utils.scene import GymScene
from isaacgym_utils.assets import GymFranka, GymBoxAsset
from isaacgym_utils.assets import GymTree
from isaacgym_utils.camera import GymCamera
from isaacgym_utils.math_utils import RigidTransform_to_transform, np_to_vec3, vec3_to_np, quat_to_np
from isaacgym_utils.policy import GraspBlockPolicy, MoveBlockPolicy, GraspTreePolicy
from isaacgym_utils.draw import draw_transforms, draw_contacts, draw_camera, draw_spheres
import pdb
import sys

from pathlib import Path

def draw_box(XYZ_center, LWH_list):
    '''
    XYZ_center: list of center points for each box
    LWH_list: length, width, height for each box
    output: plot 3D of each box with center point and rotation
    '''

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
   
    # Plot each box
    for center, LWH in zip(XYZ_center, LWH_list):
        x, y, z = center
        l, w, h = LWH
       
        # Define the vertices of the box
        vertices = [[x - l/2, y - w/2, z - h/2],
                    [x - l/2, y + w/2, z - h/2],
                    [x + l/2, y + w/2, z - h/2],
                    [x + l/2, y - w/2, z - h/2],
                    [x - l/2, y - w/2, z + h/2],
                    [x - l/2, y + w/2, z + h/2],
                    [x + l/2, y + w/2, z + h/2],
                    [x + l/2, y - w/2, z + h/2]]
       
        # Define the faces of the box
        faces = [[0,1,2,3],
                 [0,1,5,4],
                 [1,2,6,5],
                 [2,3,7,6],
                 [3,0,4,7],
                 [4,5,6,7]]
       

        print(f"vertices: {vertices}")
        print(f"faces: {faces}")
        poly3d_list = []
        for face in faces:
            
            verticies_list = []
            for edge in face:
                verticies_list.append(vertices[edge])

            print(f"for face {face} ,verticies_list: {verticies_list}")
            poly3d_list.append(verticies_list)


        obj = Poly3DCollection(poly3d_list,facecolors='blue', linewidths=1, edgecolors='black', alpha=.25)
        print(f"obj: {obj} ")


        # Plot the box
        ax.add_collection3d(obj)
       
    # Set the limits for the plot
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(0, 1)
   
    # Set the labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
   
    # Show the plot
    plt.show()


def draw_box_rotation(XYZ_center, RPY_list, LWH_list):
    '''
    XYZ_center: list of center points for each box
    RPY_list: angle rotation in radians for each box
    LWH_list: length, width, height for each box
    output: plot 3D of each box with center point and rotation
    '''

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
   
    # Plot each box
    for center, RPY, LWH in zip(XYZ_center, RPY_list, LWH_list):
        x, y, z = center
        l, w, h = LWH

        # convert RPY radians to degrees
        RPY_deg = [np.rad2deg(RPY[0]), np.rad2deg(RPY[1]), np.rad2deg(RPY[2])]
        # print(f"RPY: {RPY_deg}")
       
        # Define the vertices of the box
        vertices = [[x - l/2, y - w/2, z - h/2],
                    [x - l/2, y + w/2, z - h/2],
                    [x + l/2, y + w/2, z - h/2],
                    [x + l/2, y - w/2, z - h/2],
                    [x - l/2, y - w/2, z + h/2],
                    [x - l/2, y + w/2, z + h/2],
                    [x + l/2, y + w/2, z + h/2],
                    [x + l/2, y - w/2, z + h/2]]
       
        # Convert the RPY angles to a rotation matrix
        R = Rotation.from_euler( seq = 'xyz', angles= [RPY[0], RPY[1], RPY[2]], degrees=False )

        # convert spatial transform object to np matrix
        R = R.as_matrix()
       
        # Rotate the vertices of the box
        vertices_rotated = []
        for vertex in vertices:
            # offset vertex by center by doing np subtraction
            centerd_vertex = np.array(vertex) - np.array(center)
            transformed_vertex = R.dot(centerd_vertex)
            vertex_rotated = transformed_vertex + center
            vertices_rotated.append(vertex_rotated)

        # Define the faces of the box
        faces = [[0,1,2,3],
                 [0,1,5,4],
                 [1,2,6,5],
                 [2,3,7,6],
                 [3,0,4,7],
                 [4,5,6,7]]
       

        # print(f"vertices: {vertices_rotated}")
        # print(f"faces: {faces}")
        poly3d_list = []
        for face in faces:
            
            verticies_list = []
            for edge in face:
                verticies_list.append(vertices_rotated[edge])

            # print(f"for face {face} ,verticies_list: {verticies_list}")
            poly3d_list.append(verticies_list)


        obj = Poly3DCollection(poly3d_list,facecolors='blue', linewidths=1, edgecolors='black', alpha=.25)


        # Plot the box
        ax.add_collection3d(obj)
       
    # Set the limits for the plot
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(0, 1)
   
    # Set the labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
   
    # Show the plot
    plt.show()

    



def visualize_box_collision_for_tree():
    #load npy file for box collision
    filename = str(Path(__file__).parent.parent/'assets/franka_description/robots/tree0_box_link.npy')
    box_link_np = np.load(filename)
    print(f"box_link_np shape {box_link_np.shape} ")
    edge_def = [(0, 1), (1, 2), (1, 3), (1, 4), (2, 5), (2, 6), (3, 7), (5, 8), (4, 9), (7, 10), (8, 11), (6, 12), (9, 13), (10, 14)] #hardcoded for tree0

    XYZ_offset_list = []
    RPY_list = []
    LWH_list = []
    

    num_links = box_link_np.shape[0]
    #append to list by reading from box_link_np
    for i in range(num_links):
        # print(f" box_link_np[{i}]: {box_link_np[i,0:3]} ")
        XYZ_offset_list.append(box_link_np[i,0:3])
        RPY_list.append(box_link_np[i,3:6])
        LWH_list.append(box_link_np[i,6:9])


    # since box_link_np is cenetered in parent branch farme, need to get the world coordinate of each link and then add the box_link_np[i] to it
    grabbable_link_indices, grabbable_link_poses = get_grabbable_tree_links()

    x = grabbable_link_poses[0,:]
    y = grabbable_link_poses[1,:]
    z = grabbable_link_poses[2,:]
    

    num_of_grabbable_links = grabbable_link_poses.shape[1]
    print(f"num_of_grabbable_links: {num_of_grabbable_links}")

    initx_array = np.zeros((3,num_of_grabbable_links))

    XYZ_center = []
    for i in range(num_of_grabbable_links):
        x[i] = x[i] + box_link_np[i,0]
        y[i] = y[i] + box_link_np[i,1]
        z[i] = z[i] + box_link_np[i,2]
        print(f" xyz of link {i} is {x[i]}, {y[i]}, {z[i]} ")
        XYZ_center.append([x[i],y[i],z[i]])

        # initx_array[0,i] = x[i]
        # initx_array[1,i] = y[i]
        # initx_array[2,i] = z[i]

    draw_box_rotation(XYZ_center, RPY_list, LWH_list)
    # draw_box(XYZ_center, LWH_list)
    

    

    
    sys.exit()


def get_link_poses():
    vertex_pos = np.zeros((7,tree.num_links)) #x,y,z,qx,qy,qz,qw

    for i in range(tree.num_links):
        link_tf = tree.get_link_transform(0, tree_name, tree.link_names[i])
        pos = vec3_to_np(link_tf.p)
        quat = quat_to_np(link_tf.r)

        vertex_pos[0,i] = pos[0]
        vertex_pos[1,i] = pos[1]
        vertex_pos[2,i] = pos[2]
        vertex_pos[3,i] = quat[0]
        vertex_pos[4,i] = quat[1]
        vertex_pos[5,i] = quat[2]
        vertex_pos[6,i] = quat[3]

    # print(f" vertex {vertex_pos} ")    
    return vertex_pos

def get_grabbable_tree_links():
    # get branch link indices that can be used for interacting with
    grabbable_link_indices = []
    grabbable_link_poses = []

    idx = 0
    for link_name in tree.link_names:
        if not "base" in link_name and not "tip" in link_name: # Exclude base from being a push option
            grabbable_link_indices.append(idx)
        idx += 1
    # print(f"size of grabbable_link_indices {len(grabbable_link_indices)} ")
    # print(f"grabbable_link_indices {grabbable_link_indices} ")

    grabbable_link_poses = get_link_poses()[:,grabbable_link_indices]
    # print(f"grabbable_link_poses {grabbable_link_poses} ")
    # print(f" size of grabbable_link_poses {grabbable_link_poses.shape} ")

    return grabbable_link_indices, grabbable_link_poses


# ====================================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--cfg', '-c', type=str, default='cfg/franka_tree.yaml')
    parser.add_argument('--cfg', '-c', type=str, default='../cfg/franka_tree_force_ocrl.yaml')

    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)


    scene = GymScene(cfg['scene'])


    franka = GymFranka(cfg['franka'], scene, actuation_mode='torques')
    tree = GymTree(cfg['tree'], scene, actuation_mode='joints')


    franka_transform = gymapi.Transform(p=gymapi.Vec3(-0.5, 0, 0))
    tree_transform = gymapi.Transform(p=gymapi.Vec3(0, 0, 0))

    franka_name, tree_name, block_name = 'franka', 'tree', 'block'

    current_iteration = 0
    num_iteration = 100
    force_magnitude = 50
    push_toggle = True
    
    global vertex_init_pos, vertex_final_pos, force_applied 

    vertex_init_pos = np.zeros((7,tree.num_links)) #x,y,z,qx,qy,qz,qw
    vertex_final_pos = np.zeros((7,tree.num_links)) #x,y,z,qx,qy,qz,qw
    force_applied = np.zeros((3,tree.num_links)) #fx,fy,fz     

    

    def setup(scene, _):

        scene.add_asset(franka_name, franka, franka_transform, collision_filter=0) # avoid self-collisions

        scene.add_asset(tree_name, tree, tree_transform, collision_filter=1) # avoid self-collisions
        # scene.add_asset('block', block, gymapi.Transform(p=gymapi.Vec3(-1, -1, cfg['block']['dims']['sz']/2)) )

    scene.setup_all_envs(setup)    

    
    no_contact = True
    force = np_to_vec3([0, 0, 0])

    policy = GraspTreePolicy(franka, franka_name)



    while True:
        # get grabble tree link poses
        grabbable_link_indices, grabbable_link_poses = get_grabbable_tree_links()
        
        visualize_box_collision_for_tree()


        #randomly choose index to grab
        idx = np.random.randint(0, len(grabbable_link_indices))
        idx = -1 #hardcode to grab the last link

        goal_grab_pose = grabbable_link_poses[:,idx]

        goal_grab_pose[0] = goal_grab_pose[0] + 0.2 # arbitrary offset to see if robot will collide
        goal_grab_pose[1] = 0# arbitrary offset to see if robot will collide

        print(f"grabbing link idx {idx}, goal_grab_pose {goal_grab_pose} ")
        policy.set_grasp_goal(goal_grab_pose)

        print(f"resetting policy")
        policy.reset()
        print(f"running policy")
        scene.run(time_horizon=policy.time_horizon, policy=policy)

    