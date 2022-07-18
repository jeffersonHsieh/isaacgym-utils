import numpy as np
import os

def load_npy(data_dir, tree_num, orientational):
    # Load npy files from dataset_dir. A shortcut to 'sample_1_push' shared folder has been added to 'My Drive' 
    if orientational:
        X_stiffness_damping = np.load(os.path.join(data_dir, 'X_coeff_stiff_damp_tree%s.npy'%tree_num))
        X_edges = np.load(os.path.join(data_dir, 'X_edge_def_tree%s_ori.npy'%tree_num))
        X_force = np.load(os.path.join(data_dir, 'X_force_applied_tree%s_ori.npy'%tree_num))
        X_pos = np.load(os.path.join(data_dir, 'X_vertex_init_tree%s_ori.npy'%tree_num))
        Y_pos = np.load(os.path.join(data_dir, 'Y_vertex_final_tree%s_ori.npy'%tree_num))
    else:
        X_stiffness_damping = np.load(os.path.join(data_dir, 'X_coeff_stiff_damp_tree%s.npy'%tree_num))
        X_edges = np.load(os.path.join(data_dir, 'X_edge_def_tree%s.npy'%tree_num))
        X_force = np.load(os.path.join(data_dir, 'X_force_applied_tree%s.npy'%tree_num))
        X_pos = np.load(os.path.join(data_dir, 'X_vertex_init_pose_tree%s.npy'%tree_num))
        Y_pos = np.load(os.path.join(data_dir, 'Y_vertex_final_pos_tree%s.npy'%tree_num))

        # Truncate node orientations and tranpose to shape (num_graphs, num_nodes, 3)
        X_pos = X_pos[:, :7, :].transpose((0,2,1))
        Y_pos = Y_pos[:, :7, :].transpose((0,2,1))
        X_force = X_force.transpose((0,2,1))

    return X_edges, X_force, X_pos, Y_pos

TREE_NUM = 10
d = "/mnt/hdd/jan-malte/test_set_by_tree/"
X_force_list = []
X_pos_list = []
Y_pos_list = []
for tree in range(0, TREE_NUM):
    X_edges, X_force, X_pos, Y_pos = load_npy(d, tree, False)
    X_force_list.append(X_force)
    X_pos_list.append(X_pos)
    Y_pos_list.append(Y_pos)
    #if tree == 0:
    #    X_force_arr = X_force
    #    X_pos_arr = X_pos
    #    Y_pos_arr = Y_pos
    #else:
    #    print(np.shape(X_force_arr))
    #    print(np.shape(X_force))
    #    X_force_arr = np.concatenate((X_force_arr, X_force))
    #    X_pos_arr = np.concatenate((X_pos_arr, X_pos))
    #    Y_pos_arr = np.concatenate((Y_pos_arr, Y_pos))

#bad_idx=[36900, 1941, 71490, 77098, 94563, 96522]
print(len(X_pos_list))
total_pushes = 0
bad_tree_num = 0
for tree_idx in range(0, len(X_pos_list)):
    X_pos_arr = X_pos_list[tree_idx]
    Y_pos_arr = Y_pos_list[tree_idx]
    X_force_arr = X_force_list[tree_idx]
    total_pushes += len(X_pos_arr)
    print(np.shape(X_pos_arr))
    for idx in range(0, len(X_pos_arr)):
        avg_diff = np.sum(Y_pos_arr[idx,:,:2] - X_pos_arr[idx,:,:2], axis=0)
        force = np.sum(X_force_arr[idx,:,:2], axis=0)
        if not ((avg_diff > 0) == (force > 0)).all():
            if bad_tree_num < 20:
                print("Tree: %s"%idx)
                print("##### FORCE #####")
                print(force)
                #print("----- INIT POS -----")
                #print(X_pos_arr[idx,:,:3])
                #print("----- FINAL POS -----")
                #print(Y_pos_arr[idx,:,:3])
                #print("----- DIFF -----")
                #print(Y_pos_arr[idx,:,:3] - X_pos_arr[idx,:,:3])
                print("----- FLAT DIFF -----")
                print(avg_diff)
            bad_tree_num += 1

print(total_pushes)
print(bad_tree_num)