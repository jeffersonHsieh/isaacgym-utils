import numpy as np

TREE_NUM = 234
TREE_PTS = 10
ENV_NUM = 100
PER_TREE = True
GET_PATH = "./dataset/"
PUT_PATH = "./dataset_by_tree/"
TREE_START = 0

def combine(tree_start=TREE_START, tree_num=TREE_NUM, env_num=ENV_NUM, get_path=GET_PATH, put_path=PUT_PATH, per_tree=PER_TREE, tree_pts=TREE_PTS):
    y_vert_arrays = []
    x_vert_arrays = []
    force_applied_arrays = []
    coeff_arrays = []
    edge_def_arrays = []

    print("combine")
    for tree in range(tree_start, tree_num):
        if per_tree:
            y_vert_arrays = []
            x_vert_arrays = []
            force_applied_arrays = []
            coeff_arrays = []
            edge_def_arrays = []

        prefix = "[%s]"%tree_pts
        try:
            print(get_path + prefix + 'X_coeff_stiff_damp_tree%s.npy'%(tree))
            coeff_arrays.append(np.load(get_path + prefix + 'X_coeff_stiff_damp_tree%s.npy'%(tree)))
        except:
            coeff_arrays.append(np.load(get_path + 'X_coeff_stiff_damp_tree%s.npy'%(tree)))
            prefix = ""

        edge_def_arrays.append(np.load(get_path + prefix + 'X_edge_def_tree%s.npy'%(tree)))
        for env in range(0, env_num):
            x_vert_arrays.append(np.load(get_path + prefix + 'X_vertex_init_pos_tree%s_env%s.npy'%(tree, env)))
            force_applied_arrays.append(np.load(get_path + prefix + 'X_force_applied_tree%s_env%s.npy'%(tree, env)))
            y_vert_arrays.append(np.load(get_path + prefix + 'Y_vertex_final_pos_tree%s_env%s.npy'%(tree, env)))

        if per_tree:
            x_vert_save = x_vert_arrays[0]
            y_vert_save = y_vert_arrays[0]
            force_applied_save = force_applied_arrays[0]
            coeff_save = coeff_arrays[0]
            edge_def_save = edge_def_arrays[0]

            for idx in range(1,env_num):
                x_vert_save = np.vstack((x_vert_save, x_vert_arrays[idx]))
                y_vert_save = np.vstack((y_vert_save, y_vert_arrays[idx]))
                force_applied_save = np.vstack((force_applied_save, force_applied_arrays[idx]))
                #coeff_save = np.vstack((coeff_save, coeff_arrays[idx]))
                #edge_def_save = np.vstack((edge_def_save, edge_def_arrays[idx]))

        #print(np.shape(x_vert_save))
        #print(np.shape(y_vert_save))
        #print(np.shape(force_applied_save))
        if per_tree:
            print(np.shape(x_vert_save))
            print(np.shape(y_vert_save))
            print(np.shape(force_applied_save))
            np.save(put_path + '[%s]X_vertex_init_pose_tree%s'%(tree_pts, tree), x_vert_save)
            np.save(put_path + '[%s]X_coeff_stiff_damp_tree%s'%(tree_pts, tree), coeff_save )
            np.save(put_path + '[%s]X_edge_def_tree%s'%(tree_pts, tree), edge_def_save )
            np.save(put_path + '[%s]X_force_applied_tree%s'%(tree_pts, tree), force_applied_save )
            np.save(put_path + '[%s]Y_vertex_final_pos_tree%s'%(tree_pts, tree), y_vert_save)

    if not per_tree:
        x_vert_save = x_vert_arrays[0]
        y_vert_save = y_vert_arrays[0]
        force_applied_save = force_applied_arrays[0]
        coeff_save = coeff_arrays[0]
        edge_def_save = edge_def_arrays[0]

        for idx in range(1,len(x_vert_save)):
            x_vert_save = np.vstack((x_vert_save, x_vert_arrays[idx]))
            y_vert_save = np.vstack((y_vert_save, y_vert_arrays[idx]))
            force_applied_save = np.vstack((force_applied_save, force_applied_arrays[idx]))

            print(np.shape(x_vert_save))
            print(np.shape(y_vert_save))
            print(np.shape(force_applied_save))

            np.save(put_path + 'X_vertex_init_pose', x_vert_save)
            np.save(put_path + 'X_coeff_stiff_damp', coeff_save )
            np.save(put_path + 'X_edge_def', edge_def_save )
            np.save(put_path + 'X_force_applied', force_applied_save )
            np.save(put_path + 'Y_vertex_final_pos', y_vert_save)

# combine()