import numpy as np
GET_PATH = "/mnt/hdd/jan-malte/15Nodes_Large_by_tree/"
PUT_PATH = "/mnt/hdd/jan-malte/15Nodes_Large_by_tree/"
TREE_NUM = 36

for tree in range(0, TREE_NUM):
    x_vert_array = np.load(GET_PATH + 'X_vertex_init_pose_tree%s.npy'%(tree))
    force_applied_array = np.load(GET_PATH + 'X_force_applied_tree%s.npy'%(tree))
    y_vert_array = np.load(GET_PATH + 'Y_vertex_final_pos_tree%s.npy'%(tree))

    print(np.shape(x_vert_array))
    print(np.shape(force_applied_array))
    print(np.shape(y_vert_array))

    to_delete_idx = []
    for idx in range(0, len(force_applied_array)):
        force = force_applied_array[idx]
        if np.sum(np.abs(force)) == 0:
            to_delete_idx.append(idx)

    print("to delete: %s"%to_delete_idx)

    x_vert_array = np.delete(x_vert_array, to_delete_idx, axis=0)
    force_applied_array = np.delete(force_applied_array, to_delete_idx, axis=0)
    y_vert_array = np.delete(y_vert_array, to_delete_idx, axis=0)

    print(np.shape(x_vert_array))
    print(np.shape(force_applied_array))
    print(np.shape(y_vert_array))

    np.save(PUT_PATH + 'X_vertex_init_tree%s_clean'%tree, x_vert_array)
    np.save(PUT_PATH + 'X_force_applied_tree%s_clean'%tree, force_applied_array )
    np.save(PUT_PATH + 'Y_vertex_final_tree%s_clean'%tree, y_vert_array)