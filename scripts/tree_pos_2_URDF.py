import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
import pandas as pd

from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits import mplot3d

import os
import sys
import yaml

import SCA_tree_gen as sca


''' 
##############################################################
This script is for Interactive Perception 2023 paper.
This script generated URDF tree files from real 3D position data and explicit edge definition
input: 3D position data
input: edge definition
output: URDF file
##############################################################
''' 


path = '/home/mark/github/isaacgym-utils/scripts/dataset_mark/real_URDF/'
TREE_POINTS = 10

class real3D_to_URDF(object):
    def __init__(self, path = path):
        self.path = path

        #load pos and edge npy files
        pos_path = path + 'X_total.npy'
        edge_path = path + 'X_edge_def1.npy'

        self.X_pos_data = np.load(pos_path, allow_pickle=True)
        self.X_edge = np.load(edge_path, allow_pickle=True)

        print(f"X_pos.shape: {self.X_pos_data.shape}") #NUM_PUSH X NUM_NODES

        self.NUM_PUSH = self.X_pos_data.shape[0]
        self.NUM_NODES = self.X_pos_data.shape[1] #NUM_NODES
        self.NODE_DIM = 7 #xyzqxyzw


    def convert_X_list_to_np(self, X_list):
        '''
        convert list of list into np matrix
        '''
        node_pose_np = np.zeros((self.NUM_PUSH, self.NUM_NODES, self.NODE_DIM))

        for push_idx in range(self.NUM_PUSH):
            
            for node_idx in range (self.NUM_NODES):
                node_pos = np.array(X_list[push_idx][node_idx])
                # print(f" node_pos {node_pos}")

                if pd.isnull(node_pos).any() or node_pos is None:
                    # print(f"node_pos is nan or None")
                    node_pose_np[push_idx][node_idx][:] = 0

                else:
                    node_pose_np[push_idx][node_idx][:] = node_pos


        #remove rows with all zeros
        # print(f"X_node_pos contains  0 {np.any(X_node_pos == 0)}")    

        print(f"node_pose_np.shape: {node_pose_np.shape}") #NUM_PUSH X NUM_NODES x 7
        return node_pose_np

    def convert_edgelist_to_edgedic(self, edgelist_in):
        '''
        convert list to dictionary of edges
        # edge_def = [(0, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 6), (2, 7), (3, 8), (6, 9), (6, 10), (4, 11), (9, 12), (8, 13), (7, 14), (5, 15), (10, 16)]
        # edge_dict =  {0: np.array([1]), 1: np.array([2, 3, 5, 9]), 2: np.array([4, 8]), 4: np.array([ 6, 10]), 3: np.array([7])} 
        '''

        #create dictionary of edges with empty list
        edge_dict =  {new_list: [] for new_list in range(self.NUM_NODES)}

        # print(f"X_edge {X_edge}")

        #add edges to dictionary
        for edge in edgelist_in:
            edge_dict[edge[0]].append(edge[1])

        #convert list to np array
        for k, v in edge_dict.items():
            # print(f" k {k} v {v}")
            edge_dict[k] = np.array(v)
        
        # print(f"edge_dict {edge_dict}")
        # #remove empty dictionary
        edge_dict = {k: v for k, v in edge_dict.items() if len(v) > 0}

        print(f"edge_dict {edge_dict}")
        return edge_dict

    def plot_tree_from_real(self, tree_points, X_edge):
        '''
        plot 3D scatter and connected edges
        input: edge def list
        input: 3D position of all nodes
        '''

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        x = tree_points[:,0]
        y = tree_points[:,1]
        z = tree_points[:,2]

        ax.scatter(x,y,z)

        ax.axes.set_xlim3d(left=-2, right=2) 
        ax.axes.set_ylim3d(bottom=-2, top=2) 
        ax.axes.set_zlim3d(bottom=-2, top=2) 

        #plot edges in between nodes
        line_3D_list = []
        for idx,edge in enumerate(X_edge):
            edge_a = edge[0]
            edge_b = edge[1]
            # print(f"idx {idx} with {edge_a,edge_b}")

            line_3D_list.append([ tree_points[edge_a,0:3] , tree_points[edge_b,0:3]])


        x0_lc = Line3DCollection(line_3D_list, colors=[1,0,0,1], linewidths=1)
        ax.add_collection(x0_lc)

        plt.show()


def main():
    print(f" ================ starting sample script ================  ")

    urdfCreator = real3D_to_URDF(path)
    X_pos_list = urdfCreator.X_pos_data
    X_pos_np = urdfCreator.convert_X_list_to_np(X_pos_list)


    #get XYZ of nodes
    X_node_pose = X_pos_np[700]
    X_node_pos = X_node_pose[:, :3]
    print(f"X_node_pos {X_node_pos}")

    #convert edge list to dictionary for URDF generation
    X_edge_list = urdfCreator.X_edge
    X_edge_dic = urdfCreator.convert_edgelist_to_edgedic(X_edge_list)

    
    #create TreeGenerator
    tg = sca.TreeGenerator(path=path,max_steps=10000, att_pts_max=320, da=17, dt=0.1, step_width=0.5, offset=[-0.5, -0.5, 0.375], scaling=1, max_tree_points=TREE_POINTS, tip_radius=0.008, tree_id=0, pipe_model_exponent=3, z_strech=0.5, y_strech=0.5, x_strech=0.5, step_width_scaling=0.65, env_num=0, gui_on=1)
    tg.tree_points = X_node_pos
    tg.edges = X_edge_dic
    tg.edge_list = []


    #plot
    urdfCreator.plot_tree_from_real(tg.tree_points, urdfCreator.X_edge)

    
    #generate URDF
    tg.calculate_branch_thickness()
    name_dict, edge_def, urdf_path = tg.generate_urdf()
    yaml_path, stiffness_list, damping_list = tg.generate_yaml()

    print(f" name_dict: {name_dict} , \n edge_def: {edge_def} , \n urdf_path: {urdf_path} ")
    print(f" yaml_path: {yaml_path} , \n stiffness_list: {stiffness_list} , \n damping_list: {damping_list} ")
    print(f" len(name_dict): {len(name_dict['joints'])}")


    print(f" ================ ending sample script ================  ")


if __name__ == '__main__':
    main()
