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

#custom util lib for tree IG
import SCA_tree_gen as sca
import isaacgym_loader as ig_loader


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
sim_path = '/home/mark/data/IsaacGym/dataset_mark/real_URDF/real2sim_data/'
save_path = '/home/mark/data/IsaacGym/dataset_mark/real_URDF/real2sim_data/K_search_01/'
K_search_path = '/home/mark/data/IsaacGym/dataset_mark/real_URDF/real2sim_data/K_search/'

TREE_POINTS = 10

class Real2SimEvaluator(object):
    def __init__(self, path = path, sim_path = sim_path, save_path = save_path):
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

        #remove None data
        # self.X_clean = self.preprocessX(self.X)

        self.NUM_PUSHES = self.X.shape[0]
        self.NUM_NODES = self.X.shape[1]

        #transform X,Y data to the origin
        self.X_origin = self.offset_origin_dataX(self.X)
        self.Y_origin = self.offset_origin_data(self.Y)


        #load sim pos data
        # X_sim_path = sim_path + '[14]X_vertex_init_pos_treeK_env01.npy'
        # F_sim_path = sim_path + '[14]X_force_applied_treeK_env01.npy'
        # Y_sim_path = sim_path + '[14]Y_vertex_final_pos_treeK_env01.npy'
        # self.X_sim = np.load(X_sim_path, allow_pickle=True)
        # self.F_sim = np.load(F_sim_path, allow_pickle=True)
        # self.Y_sim = np.load(Y_sim_path, allow_pickle=True)

        # self.NUM_PUSHES_SIM = self.X_sim.shape[0]
        # self.NUM_NODES_SIM = self.X_sim.shape[2]

        #load yaml file with config data about IG param and tree param
        yaml_path = os.path.join(path, "[10]tree0.yaml")
        with open(yaml_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
        self.NUM_ENV = cfg['scene']['n_envs']

        # print(f"NUM_ENV {self.NUM_ENV} ")


    # def preprocessX(self, data):
    #     '''
    #     remove None data
    #     '''
    #     X_clean = []
    #     for i in range(data.shape[0]):
    #         X_clean.append([])
    #         for j in range(data.shape[1]):
    #             if data[i,j] is not None:
    #                 X_clean[i].append(data[i,j])

    #     return X_clean

    def offset_origin_dataX(self, data):
        '''
        offset the data to the origin using the first initial position
        '''

        #get initial position
        print(f" data shape {data.shape}")
        init_pos = data[0,0][0:3]
        print(f" init_pos {init_pos} ")

        #offset the rest of the data from initial position
        for i in range(data.shape[0]): #num pushes

            # print(f" i: {i},  data[i,:] {data[i,:]}")
            if (None in data[i,:] ):
                #pad this with zeros
                data[i,:] = [0] * data.shape[1]
                continue

            for j in range(data.shape[1]): #num nodes
                # print(f" data[i,j] {data[i,j][0:3]} \n init_pos {init_pos}")
                data[i,j] = np.array(data[i,j][0:3]) - np.array(init_pos)
        

        return data

    def offset_origin_data(self, data):
        '''
        offset the data to the origin using the first initial position
        '''

        #get initial position
        print(f" data shape {data.shape}")
        init_pos = data[0,0,0:3]
        print(f" init_pos {init_pos} ")

        data_clean = np.zeros((data.shape[0], data.shape[1], 3))

        #offset the rest of the data from initial position
        for i in range(data.shape[0]): #num pushes
            for j in range(data.shape[1]): #num nodes
                # print(f" before offset {data[i,j,0:3]}")
                data_clean[i,j,0:3] = np.array(data[i,j,0:3]) - np.array(init_pos)
                # print(f" after offset {data_clean[i,j,0:3]}")
        

        return data_clean


       


    def load_data_from_path(self,path_to_data):
        X = []
        F = []
        Y = []

        X_path = path_to_data + '[14]X_vertex_init_pos_treeK_env0.npy'
        F_path = path_to_data + '[14]X_force_applied_treeK_env0.npy'
        Y_path = path_to_data + '[14]Y_vertex_final_pos_treeK_env0.npy'
        edge_path = path_to_data + 'X_edge_def1.npy'

        X = np.load(X_path, allow_pickle=True)
        F = np.load(F_path, allow_pickle=True)
        Y = np.load(Y_path, allow_pickle=True)
        edge_def = np.load(edge_path, allow_pickle=True)

        self.NUM_PUSHES_SIM = X.shape[0]
        self.NUM_NODES_SIM = X.shape[2]

        return X,F,Y, edge_def

    def plot_sim_tree_measurements(self, X,F,Y, edge_def):
        '''
        plot the IG  displacements in world
        '''

        print(f" ==================== FXY {F.shape}, {X.shape, Y.shape} ==================== ")

        
        data1 = np.zeros((4, self.NUM_PUSHES_SIM * self.NUM_NODES_SIM)) #4 bc xyz,id
        # print(f"size of data1 is {data1.shape} b/c num push, num node: {self.NUM_PUSHES_SIM, self.NUM_NODES_SIM} ")

        nodes_position_list = []
        count = 0

        
        for pushes in Y:

            for i in range (self.NUM_NODES_SIM ):

            
                
                node = i
                # pushes = nodes x pose [N,7]
                data1[0,count] = pushes[0][node]
                data1[1,count] = pushes[1][node]
                data1[2,count] = pushes[2][node]
                data1[3,count] = node
                # print(f"x,y,z, {data1[0,count], data1[1,count], data1[2,count]} ")
                count = count + 1

            

            

        x = data1[0,:]      
        y = data1[1,:] 
        z = data1[2,:] 
        c = data1[3,:]

        
        


        print(f"size of xyz is {x.shape, y.shape, z.shape}")

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

        ax.set_title('Sim Tree Displacement')

        for i in range (self.NUM_NODES_SIM):
            ax.text(x[i], y[i], z[i], c[i].astype(int), color='black')


        #= init pt plot ===
        print(f" size of Y {Y.shape} and {Y[0].shape}")

        # ======draw red circles of nodes ============================================================


        initx_array = np.zeros((3,self.NUM_NODES_SIM))

        valid_push_idx = 0

        for n in range(self.NUM_NODES_SIM):
            initx_array[0,n] = X[valid_push_idx][0][n]
            initx_array[1,n] = X[valid_push_idx][1][n]
            initx_array[2,n] = X[valid_push_idx][2][n]
        
        scatter2 = ax.scatter(initx_array[0], initx_array[1],initx_array[2], c='r', s = 50)
        # print(f" size {initx_array.shape} {initx_array}")

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


        x = data1[0,:]      
        y = data1[1,:] 
        z = data1[2,:] 
        c = data1[3,:]

        #down sample
        x = data1[0,::10]      
        y = data1[1,::10] 
        z = data1[2,::10] 
        c = data1[3,::10]


        # print(f"size of xyz is {x.shape, y.shape, z.shape}")

        # plotting
        fig = plt.figure()
        
        # syntax for 3-D projection
        ax = plt.axes(projection ='3d')
    

        scatter = ax.scatter(x, y, z, c = c, s=1)
        # produce a legend with the unique colors from the scatter
        legend1 = ax.legend(*scatter.legend_elements(),
                        loc="lower left", title="branch")

        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        # plt.zlabel("z (m)")

        ax.set_title('Real Tree Displacement ')
        
        
            


        #======draw red circles of nodes ============================================================


        initx_array = np.zeros((3,10))

        valid_push_idx = 700

        for n in range(10):
            initx_array[0,n] = X[valid_push_idx][n][0]
            initx_array[1,n] = X[valid_push_idx][n][1]
            initx_array[2,n] = X[valid_push_idx][n][2]

            # add text label
            ax.text(initx_array[0,n], initx_array[1,n], initx_array[2,n], n  , color='black')

        
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

    def convert_onehot_to_array(self, F):
        '''
        convert onehot to list of contact idx and contact force
        '''
        contact_idx_list = []
        contact_force_list = []

        for idx, push in enumerate(F):
            # print(f"idx {idx} with {push}")
            # print(f" push.shape {push.shape}")

            #find column index of one hot vector 
            contact_row, contact_col = np.unravel_index(np.argmax(push), push.shape)
            # print(f"contact idx, col {contact_row, contact_col}")
            contact_idx = contact_col
            contact_idx_list.append(contact_idx)
            contact_force_list.append(push[:,contact_idx])

        # return contact_idx_list, contact_force_list

        F_idx_vec_array = np.zeros((len(contact_idx_list), 4)) #idx, Fx, Fy, Fz
        for idx, contact_idx in enumerate(contact_idx_list):
            F_idx_vec_array[idx,0] = (contact_idx).astype(int)
            F_idx_vec_array[idx,1] = contact_force_list[idx][0]
            F_idx_vec_array[idx,2] = contact_force_list[idx][1]
            F_idx_vec_array[idx,3] = contact_force_list[idx][2]
        
        
        return F_idx_vec_array

        

def main():
    print(f" ================ starting sample script ================  ")

    Real2Sim = Real2SimEvaluator(path,sim_path,save_path)
    Real2Sim.plot_real_tree_measurements(Real2Sim.X_origin,Real2Sim.F,Real2Sim.Y_origin, Real2Sim.edge_def)
    sys.exit()

    # ================ data collect =================

    #load F applied info
    # print(f" Real2Sim.F shape {Real2Sim.F.shape} ")
    F_push_array = Real2Sim.convert_onehot_to_array(Real2Sim.F)
    print(f" F_push_array before downsample  {F_push_array.shape}")

    #down_sampled F for quicker data collection
    F_push_array = F_push_array[::10,:]
    print(f" F_push_array after downsample {F_push_array.shape}")


    #create 1D search space for K = C * f(radius). 
    stiffness_increment  = np.linspace(0.2, 4.0, Real2Sim.NUM_ENV) 

    #create IsaacGym loader with tree, F applied
    ig = ig_loader.IG_loader(path, save_path, stiffness_list= None, stiffness_increment = stiffness_increment, F_push_array = F_push_array)
    
    #collect data for Y in sim 
    # ig.run_policy_do_nothing()
    ig.run_policy()

    # ================ analysis =================

    #plot Y sim
    # Real2Sim.plot_sim_tree_measurements(Real2Sim.X_sim,Real2Sim.F_sim,Real2Sim.Y_sim, Real2Sim.edge_def)

    #plot Y sim in K search dataset
    # X_search, F_search, Y_search, edge_def_search = Real2Sim.load_data_from_path(K_search_path)
    # Real2Sim.plot_sim_tree_measurements(X_search, F_search, Y_search, edge_def_search)
 





    print(f" ================ ending sample script ================  ")


if __name__ == '__main__':
    main()
