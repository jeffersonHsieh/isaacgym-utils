import os
import numpy as np


def load_npy(data_dir, tree_num):
    # Load npy files from dataset_dir. A shortcut to 'sample_1_push' shared folder has been added to 'My Drive' 
    prefix = "[10]"
    X_stiffness_damping = np.load(os.path.join(data_dir, prefix + 'X_coeff_stiff_damp_tree%s.npy'%tree_num))
    X_edges = np.load(os.path.join(data_dir, prefix + 'X_edge_def_tree%s.npy'%tree_num))
    X_force = np.load(os.path.join(data_dir, prefix + 'X_force_applied_tree%s.npy'%tree_num))
    X_pos = np.load(os.path.join(data_dir, prefix + 'X_vertex_init_pose_tree%s.npy'%tree_num))
    Y_pos = np.load(os.path.join(data_dir, prefix + 'Y_vertex_final_pos_tree%s.npy'%tree_num))

    # Truncate node orientations and tranpose to shape (num_graphs, num_nodes, 3)
    X_pos = X_pos[:, :7, :].transpose((0,2,1))
    Y_pos = Y_pos[:, :7, :].transpose((0,2,1))
    X_force = X_force.transpose((0,2,1))
    return X_edges, X_force, X_pos, Y_pos

TREE_NUM = 1
d = "/home/mark/github/isaacgym-utils/data/20Ntest_per_tree/"

dataset = []
for tree in range(0, TREE_NUM):
    X_force_list = []
    X_pos_list = []
    Y_pos_list = []
    X_edges, X_force, X_pos, Y_pos = load_npy(d, tree)
    for index in range(0,len(X_pos)):
        active_push_X = X_pos[index]
        active_push_Y = Y_pos[index]
        duplication_indices = []
        for comp_index in range(0,len(X_pos)):
            if comp_index != index:
                if np.all(np.equal(X_pos[index], X_pos[comp_index])) and np.all(np.equal(Y_pos[index], Y_pos[comp_index])):
                    duplication_indices.append(comp_index)
        if len(duplication_indices) != 0:
            print(duplication_indices)


