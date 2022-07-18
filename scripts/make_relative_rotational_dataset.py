import numpy as np
from scipy.spatial.transform import Rotation
import os

def get_parent(node, edges):
    for parent, child in edges:
        if child == node:
            return parent
    return None

def calculate_orientational_data(X_edges, X_pos, Y_pos, X_force):
    #remove useless quat data
    X_pos = X_pos[:,:,:3]
    Y_pos = Y_pos[:,:,:3]
    shape = np.shape(X_pos)
    init_quat_data = np.zeros((shape[0], shape[1], 4))
    final_quat_data = np.zeros((shape[0], shape[1], 4)) 
    for push_idx in range(0,shape[0]):
        init_inv_rot_dict = {} #dictionary to store inverse rotations => used to calculate relative rotation
        final_inv_rot_dict = {} #dictionary to store inverse rotations => used to calculate relative rotation
        for parent, child in X_edges:
            rot_init = calculate_rot(X_pos[push_idx][parent], X_pos[push_idx][child])
            rot_final = calculate_rot(Y_pos[push_idx,parent], Y_pos[push_idx,child])
            if parent == 0: #ASSUMPTION: Root node is always labelled 0 in the dataset
                init_inv_rot = rot_init.inv()
                final_inv_rot = rot_final.inv()
            else:
                parents_parent = get_parent(parent, X_edges)
                rot_init = rot_init * init_inv_rot_dict[parents_parent] # rot = rot_self*rot_parent => rot_self = rot*rot_parent^(-1)
                rot_final = rot_final * final_inv_rot_dict[parents_parent]
                init_inv_rot = init_inv_rot_dict[parents_parent] * rot_init.inv()
                final_inv_rot =  final_inv_rot_dict[parents_parent] * rot_final.inv()
                
            quat_init = rot_init.as_quat()
            quat_final = rot_final.as_quat()

            init_quat_data[push_idx][parent] = quat_init
            final_quat_data[push_idx][parent] = quat_final
            
            if parent not in init_inv_rot_dict.keys():
                init_inv_rot_dict[parent] = init_inv_rot
                final_inv_rot_dict[parent] = final_inv_rot

    X_pos = np.append(X_pos, init_quat_data, axis=2)
    Y_pos = np.append(Y_pos, final_quat_data, axis=2)
    X_edges, X_pos, Y_pos, X_force = prune_tip_links(X_edges, X_pos, Y_pos, X_force)
    return X_edges, X_pos, Y_pos, X_force

def calculate_rot(parent_pos, child_pos): #Function produces correct results -> Identical with ground truth
        Z_tmp = child_pos - parent_pos
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
        return rot

def add_length_parameter(X_edges, X_pos):
    length_list = [0]*np.shape(X_pos)[1]
    tree_rep = X_pos[0]
    for parent, child in X_edges:
        length = np.linalg.norm(tree_rep[child][:3] - tree_rep[parent][:3])
        length_list[parent] = length
    length_arr = np.tile(np.array(length_list), (np.shape(X_pos)[0],1))
    length_arr = np.reshape(length_arr, (np.shape(X_pos)[0], np.shape(X_pos)[1], 1))
    X_pos = np.append(X_pos, length_arr, axis=2)
    return X_pos

def remove_tip(tip_index, edges, init_positions, final_positions, X_force):
    init_positions = np.delete(init_positions, tip_index, axis=1)
    final_positions = np.delete(final_positions, tip_index, axis=1)
    illegal_pushes = []
    for idx, force in enumerate(X_force): # If a push occurs on a tip that needs to be removed, delete the entire push
        if np.linalg.norm(force[tip_index]) != 0:
            illegal_pushes.append(idx)
    X_force = np.delete(X_force, tip_index, axis=1)
    while len(illegal_pushes) > 0:
        push_idx = illegal_pushes.pop() # remove currently highest index illegal push and delete it
        init_positions = np.delete(init_positions, push_idx, axis=0)
        final_positions = np.delete(final_positions, push_idx, axis=0)
        X_force = np.delete(X_force, push_idx, axis=0)
    new_edges = []
    for parent, child in edges:
        if child != tip_index:
            new_edges.append((parent, child))
    return new_edges, init_positions, final_positions, X_force

def prune_tip_links(edges, init_positions, final_positions, X_force):

    init_positions = add_length_parameter(edges, init_positions)
    tree_representative = init_positions[0]
    tips = []
    for i, node in enumerate(tree_representative):
        if len(find_children(edges, i)) == 0:
            tips.append(i)
    while len(tips) > 0:
        tip_index = tips.pop()
        edges, init_positions, final_positions, X_force = remove_tip(tip_index, edges, init_positions, final_positions, X_force)
        tips = adjust_indexing_single(tips, tip_index)
        edges = adjust_indexing(edges, tip_index)
    edges = np.array(edges)
    return edges, init_positions, final_positions, X_force

def find_children(edges, node):
    children = []
    for parent, child in edges:
        if parent == node:
            children.append(child)
    return children

def adjust_indexing_single(in_list, deleted_index):
    new_list = []
    for i in in_list:
        if i > deleted_index:
            i = i-1
        new_list.append(i)
    return new_list

def adjust_indexing(tuple_list, deleted_index):
    new_tuple_list = []
    for i, j in tuple_list:
        if i > deleted_index:
            i = i-1
        if j > deleted_index:
            j = j-1
        new_tuple_list.append((i,j))
    return new_tuple_list

GET_PATH = "/mnt/hdd/jan-malte/8Nodes_new_by_tree/"
PUT_PATH = "/mnt/hdd/jan-malte/8Nodes_new_by_tree/"
TREE_NUM = 12

for tree in range(0, TREE_NUM):
    X_edges = np.load(os.path.join(GET_PATH, 'X_edge_def_tree%s.npy'%tree))
    force_applied_array = np.load(os.path.join(GET_PATH, 'X_force_applied_tree%s.npy'%tree))
    x_vert_array = np.load(os.path.join(GET_PATH, 'X_vertex_init_pose_tree%s.npy'%tree))
    y_vert_array = np.load(os.path.join(GET_PATH, 'Y_vertex_final_pos_tree%s.npy'%tree))

    print("######################################")

    print(np.shape(x_vert_array))
    print(np.shape(force_applied_array))
    print(np.shape(y_vert_array))
    print(np.shape(X_edges))

    x_vert_array = x_vert_array[:, :7, :].transpose((0,2,1))
    y_vert_array = y_vert_array[:, :7, :].transpose((0,2,1))
    force_applied_array = force_applied_array.transpose((0,2,1))

    print("--------------------------------------")
    print(np.shape(x_vert_array))
    print(np.shape(force_applied_array))
    print(np.shape(y_vert_array))
    print(np.shape(X_edges))

    X_edges, x_vert_array, y_vert_array, force_applied_array = calculate_orientational_data(X_edges, x_vert_array, y_vert_array, force_applied_array)

    print("--------------------------------------")
    print(np.shape(x_vert_array))
    print(np.shape(force_applied_array))
    print(np.shape(y_vert_array))
    print(np.shape(X_edges))

    print("######################################")

    np.save(PUT_PATH + 'X_vertex_init_tree%s_ori'%tree, x_vert_array)
    np.save(PUT_PATH + 'X_force_applied_tree%s_ori'%tree, force_applied_array )
    np.save(PUT_PATH + 'Y_vertex_final_tree%s_ori'%tree, y_vert_array)
    np.save(PUT_PATH + 'X_edge_def_tree%s_ori'%tree, X_edges)