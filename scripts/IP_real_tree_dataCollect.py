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
This script is to emulate the recorded  interaction in real world to the sim tree 
input: URDF tree of real tree
input: X,F,edge_def, Y
output: visualization of the difference between real and sim
##############################################################
''' 


path = '/home/mark/data/IsaacGym/dataset_mark/real_URDF/'
TREE_POINTS = 10

class Real2SimEvaluator(object):
    def __init__(self, path = path):
        self.path = path

        #load pos and edge npy files
        X_path = path + 'X_total.npy'
        edge_path = path + 'X_edge_def1.npy'
        Y_path = path + 'Y_total.npy'
        F_path = path + 'F_vector_final.npy'

        self.X = np.load(X_path, allow_pickle=True)
        self.Y = np.load(Y_path, allow_pickle=True)
        self.F = np.load(F_path, allow_pickle=True)
        self.edge_def = np.load(edge_path, allow_pickle=True)

        self.NUM_PUSHES = self.X.shape[0]
        self.NUM_NODES = self.X.shape[1]
        

    def plot_real_tree_measurements(self, X,F,Y, edge_def):
        '''
        plot the recorded marker displacements in world
        '''

        print(f" ==================== FXY {F.shape}, {X.shape, Y.shape} ==================== ")

        
        data1 = np.zeros((4, self.NUM_PUSHES  * self.NUM_NODES)) #4 bc xyz,id

        nodes_position_list = []
        count = 0

        for i in range (self.NUM_NODES):

            for pushes in Y:
                
                node = i
                # pushes = nodes x pose [N,7]
                data1[0,count] = pushes[node][0]
                data1[1,count] = pushes[node][1]
                data1[2,count] = pushes[node][2]
                data1[3,count] = node
                # print(f"x,y,z, {data1[0,count], data1[1,count], data1[2,count]} ")
                count = count + 1

        x = data1[0]     
        y = data1[1] 
        z = data1[2] 
        c = data1[3]

        # print(f"size of xyz is {x.shape, y.shape, z.shape}")

        # plotting
        fig = plt.figure()
        
        # syntax for 3-D projection
        ax = plt.axes(projection ='3d')
    

        scatter = ax.scatter(x, y, z, c = c, s=2)
        # produce a legend with the unique colors from the scatter
        legend1 = ax.legend(*scatter.legend_elements(),
                        loc="lower left", title="branch")

        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        # plt.zlabel("z (m)")

        ax.set_title('Tree Displacement in 3D  ')

        #= init pt plot ===
        print(f" size of X {X.shape} and {X[0].shape}")


        #======draw red circles of nodes ============================================================


        initx_array = np.zeros((3,10))

        valid_push_idx = 700

        for n in range(10):
            initx_array[0,n] = X[valid_push_idx][n][0]
            initx_array[1,n] = X[valid_push_idx][n][1]
            initx_array[2,n] = X[valid_push_idx][n][2]
        
        scatter2 = ax.scatter(initx_array[0], initx_array[1],initx_array[2], c='r', s = 50)
        print(f" size {initx_array.shape} {initx_array}")

        # plt.show()

        #======draw lines between tree============================================================

        xtreelist = []
        ytreelist = []
        ztreelist = []

        line_3D_list = []

        for idx,edge in enumerate(edge_def):
            edge_a = edge[0]
            edge_b = edge[1]
            print(f"idx {idx} with {edge_a,edge_b}")

            line_3D_list.append([ initx_array[:,edge_a] , initx_array[:,edge_b]])


        x0_lc = Line3DCollection(line_3D_list, colors=[1,0,0,1], linewidths=1)

        ax.add_collection(x0_lc)

        plt.show()



def main():
    print(f" ================ starting sample script ================  ")

    Real2Sim = Real2SimEvaluator(path)
    Real2Sim.plot_real_tree_measurements(Real2Sim.X,Real2Sim.F,Real2Sim.Y, Real2Sim.edge_def)

    #load URDF of real tree

    #load F applied info

    #create IsaacGym loader with tree, F applied

    #collect data for Y in sim 

    #compare Y in sim and Y in real



    print(f" ================ ending sample script ================  ")


if __name__ == '__main__':
    main()
