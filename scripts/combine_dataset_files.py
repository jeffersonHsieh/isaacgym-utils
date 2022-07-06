import numpy as np

TREE_NUM = 37
ENV_NUM = 100
PER_TREE = True
GET_PATH = "/mnt/hdd/jan-malte/15Nodes_Large/"
PUT_PATH = "/mnt/hdd/jan-malte/15Nodes_Large_by_tree/"
TREE_START = 0

y_vert_arrays = []
x_vert_arrays = []
force_applied_arrays = []
coeff_arrays = []
edge_def_arrays = []


for tree in range(TREE_START, TREE_NUM):
    if PER_TREE:
        y_vert_arrays = []
        x_vert_arrays = []
        force_applied_arrays = []
        coeff_arrays = []
        edge_def_arrays = []

    coeff_arrays.append(np.load(GET_PATH + 'X_coeff_stiff_damp_tree%s.npy'%(tree)))
    edge_def_arrays.append(np.load(GET_PATH + 'X_edge_def_tree%s.npy'%(tree)))
    for env in range(0, ENV_NUM):
        #print(np.shape(np.load('X_vertex_init_pose_tree%s_env%s.npy'%(tree, env))))
        #print(np.shape(np.load('X_force_applied_tree%s_env%s.npy'%(tree, env))))
        #print(np.shape(np.load('Y_vertex_final_pos_tree%s_env%s.npy'%(tree, env))))
        x_vert_arrays.append(np.load(GET_PATH + 'X_vertex_init_pose_tree%s_env%s.npy'%(tree, env)))
        force_applied_arrays.append(np.load(GET_PATH + 'X_force_applied_tree%s_env%s.npy'%(tree, env)))
        y_vert_arrays.append(np.load(GET_PATH + 'Y_vertex_final_pos_tree%s_env%s.npy'%(tree, env)))


    #print(np.shape(x_vert_arrays[-1]))
    #print(np.shape(y_vert_arrays[-1]))
    #print(np.shape(force_applied_arrays[-1]))

    if PER_TREE:
        x_vert_save = x_vert_arrays[0]
        y_vert_save = y_vert_arrays[0]
        force_applied_save = force_applied_arrays[0]
        coeff_save = coeff_arrays[0]
        edge_def_save = edge_def_arrays[0]

        for idx in range(1,ENV_NUM):
            x_vert_save = np.vstack((x_vert_save, x_vert_arrays[idx]))
            y_vert_save = np.vstack((y_vert_save, y_vert_arrays[idx]))
            force_applied_save = np.vstack((force_applied_save, force_applied_arrays[idx]))
            #coeff_save = np.vstack((coeff_save, coeff_arrays[idx]))
            #edge_def_save = np.vstack((edge_def_save, edge_def_arrays[idx]))

    #print(np.shape(x_vert_save))
    #print(np.shape(y_vert_save))
    #print(np.shape(force_applied_save))
    if PER_TREE:
        print(np.shape(x_vert_save))
        print(np.shape(y_vert_save))
        print(np.shape(force_applied_save))
        np.save(PUT_PATH + 'X_vertex_init_pose_tree%s'%(tree), x_vert_save)
        np.save(PUT_PATH + 'X_coeff_stiff_damp_tree%s'%(tree), coeff_save )
        np.save(PUT_PATH + 'X_edge_def_tree%s'%(tree), edge_def_save )
        np.save(PUT_PATH + 'X_force_applied_tree%s'%(tree), force_applied_save )
        np.save(PUT_PATH + 'Y_vertex_final_pos_tree%s'%(tree), y_vert_save)

if not PER_TREE:
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

        np.save(PUT_PATH + 'X_vertex_init_pose', x_vert_save)
        np.save(PUT_PATH + 'X_coeff_stiff_damp', coeff_save )
        np.save(PUT_PATH + 'X_edge_def', edge_def_save )
        np.save(PUT_PATH + 'X_force_applied', force_applied_save )
        np.save(PUT_PATH + 'Y_vertex_final_pos', y_vert_save)

        