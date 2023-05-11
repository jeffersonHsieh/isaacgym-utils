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
import IP_real_tree_dataCollect as IP_dataCollect
import torch

''' 
##############################################################
This script is for Interactive Perception 2023 paper.
input: 
output: 
##############################################################
''' 


path = '/home/mark/data/IsaacGym/dataset_mark/real_URDF/'
sim_path = '/home/mark/data/IsaacGym/dataset_mark/real_URDF/real2sim_data/'
save_path = '/home/mark/data/IsaacGym/dataset_mark/real_URDF/real2sim_data'
K_search_path = '/home/mark/data/IsaacGym/dataset_mark/real_URDF/real2sim_data/K_search_01/'


def load_data_with_varying_env(path,n_env):
    '''
    load data from path with varying env
    input: path, n_env
    output: X, F, Y, edge_def
    '''
    #load data
    '[14]X_vertex_init_pos_treeK_env30.npy'

    X = np.load(path + f'[14]X_vertex_init_pos_treeK_env{n_env}.npy')
    F = np.load(path + f'[14]X_force_applied_treeK_env{n_env}.npy')
    Y = np.load(path + f'[14]Y_vertex_final_pos_treeK_env{n_env}.npy')
    edge_def = np.load(path + f'X_edge_def1.npy')

    return X, F, Y, edge_def


def getMSE(realY, simY):
    '''
    get MSE of realY and simY for all nodes
    input: realY, simY (dim: NUM_PUSH x 7 x NUM_NODES)
    output: MSE (dim: scalar)
    '''
    MSE_avg = 0
    MSE_push_list = []

    NUM_PUSH = realY.shape[0]

    # print(f" shape of realY = {realY.shape}, simY = {simY.shape}")

    for n_push in range (NUM_PUSH):
        #get pos from pose of each push
        pose_realY = realY[n_push]
        pose_simY = simY[n_push]

        pos_realY = pose_realY[0:3,:] #xyz
        pos_simY = pose_simY[0:3,:]

        #get MSE of pos
        diff = pos_realY - pos_simY
        # print(f" pos_realY = {pos_realY}, \n pos_simY = {pos_simY}")
        MSE = np.sum( np.linalg.norm(pos_realY - pos_simY, axis=0) ) / (realY.shape[2])
        print(f" norm = {MSE}")
        MSE_push_list.append(MSE)
        

    
    MSE_avg = np.sum(MSE_push_list) / NUM_PUSH
    print(f" norm avg = {MSE_avg}")

    return MSE_avg


def main():
    print(f" ================ starting sample script ================  ")

 

    Real2Sim = IP_dataCollect.Real2SimEvaluator(path,sim_path,save_path)
    # Real2Sim.plot_real_tree_measurements(Real2Sim.X_origin,Real2Sim.F,Real2Sim.Y_origin, Real2Sim.edge_def)
    
    #plot Y sim in K search dataset

    X_search, F_search, Y_search, edge_def_search = Real2Sim.load_data_from_path(K_search_path)
    # Real2Sim.plot_sim_tree_measurements(X_search, F_search, Y_search, edge_def_search)

    print(f" shape Y  = {Y_search.shape}")

    # sys.exit()

    
    



    # =================================================
    MSE_K_list = []

    #get MSE of sim tree vs real tree for each env loaded npy
    for n_env  in range (Real2Sim.NUM_ENV):

        #load realY, simY[n_env]
        _, _, Y_search, _ = load_data_with_varying_env(K_search_path, n_env)

        # ============== get MSE ==============
        realY_down = Real2Sim.Y_origin[::10,:,:]
        # print(f"shape of realY_down before flip = {realY_down.shape}")

        before_flip = realY_down[0,:,0:3]
        # print(f" realY {before_flip.shape}, \n {before_flip}")

        #flip 2nd and 3rd dim of realY_down
        realY_down = np.swapaxes(realY_down, 1, 2)
        # print(f" shape of realY_down after flip = {realY_down.shape}")
        after_flip = realY_down[0,0:3,:]
        # print(f" realY {after_flip.shape}, \n {after_flip}")
        
        # print(f" Y_search shape = {Y_search.shape}")
        Y_sample = Y_search[0,0:3,:]
        # print(f" Y_sample {Y_sample}")




        MSE_env = getMSE(realY_down, Y_search)

        #append to MSElist
        MSE_K_list.append(MSE_env)

    
    #create 1D search space for K = C * f(radius). 
    C_search  = np.linspace(0.2, 4.0, Real2Sim.NUM_ENV)  

    fig = plt.figure()
    ax = plt.axes()

    plt.title('MSE vs C*')
    plt.xlabel('C*')
    plt.ylabel('MSE (m)')

    
    ax.plot(C_search, MSE_K_list)
    # plt.show()

    # =================================================

    #find argmin of MSE_K_list
    C_min_idx  = np.argmin(np.array(MSE_K_list))
    C_scale = C_search[C_min_idx]
    MSE_min_val = MSE_K_list[C_min_idx]
    print(f" C_min_idx = {C_min_idx}, C_scale = {C_scale}")
    print(f" C* = {C_scale} resulting in MSE = {MSE_min_val}")

    #add to plot C_min and MSE_min
    ax.plot(C_scale, MSE_min_val, 'ro')
    plt.show()

    # =================================================





    print(f" ================ ending sample script ================  ")


if __name__ == '__main__':
    main()
