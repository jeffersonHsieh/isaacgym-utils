import numpy as np

TREE_NUM = 10
ENV_NUM = 1
PATH = "/mnt/hdd/jan-malte/12Nodes_by_tree/"
PER_TREE = True
CLEAN = "" #"_clean"
displacements = []
max_displacements = []
min_displacements = []

for tree in range(0, TREE_NUM):
    y_vert_arrays = []
    x_vert_arrays = []
    force_applied_arrays = []
    if not PER_TREE:
        for env in range(0, ENV_NUM):
            x_vert_arrays.append(np.load(PATH + 'X_vertex_init_pose_tree%s_env%s.npy'%(tree, env)))
            force_applied_arrays.append(np.load(PATH + 'X_force_applied_tree%s_env%s.npy'%(tree, env)))
            y_vert_arrays.append(np.load(PATH + 'Y_vertex_final_pos_tree%s_env%s.npy'%(tree, env)))

        init_vertices = x_vert_arrays[0]
        final_vertices = y_vert_arrays[0]
        force_applied = force_applied_arrays[0]

        for idx in range(1,ENV_NUM):
            init_vertices = np.vstack((init_vertices, x_vert_arrays[idx]))
            final_vertices = np.vstack((final_vertices, y_vert_arrays[idx]))
            force_applied = np.vstack((force_applied, force_applied_arrays[idx]))
    else:
        init_vertices = np.load(PATH + 'X_vertex_init_pose_tree%s'%(tree) + CLEAN +'.npy')
        final_vertices = np.load(PATH + 'Y_vertex_final_pos_tree%s'%(tree) + CLEAN +'.npy')
        force_applied = np.load(PATH + 'X_force_applied_tree%s'%(tree) + CLEAN +'.npy')

    
    init_vertex_variation = 0
    first_init_vertex = init_vertices[0]
    for init_vertex in init_vertices[1:]:
        init_vertex_variation += np.linalg.norm(first_init_vertex[0:3,:] - init_vertex[0:3,:])
    init_vertex_variation = init_vertex_variation/np.shape(init_vertices)[0]
    print("average variation of initial_vertex (relative to first): %s"%init_vertex_variation)

    print("shape of init_vertices for tree%s: %s"%(tree, np.shape(init_vertices)))
    print("shape of final_vertices for tree%s: %s"%(tree, np.shape(final_vertices)))
    print("shape of force_applied for tree%s: %s"%(tree, np.shape(force_applied)))

    displacement = 0
    max_displacement = max(np.sum(np.linalg.norm((init_vertices[:,0:3] - final_vertices[:,0:3]), axis=1), axis=1))
    min_displacement = min(np.sum(np.linalg.norm((init_vertices[:,0:3] - final_vertices[:,0:3]), axis=1), axis=1))
    print("minimum displacement: %s"%min_displacement)
    print("maximum displacement: %s"%max_displacement)
    max_displacements.append(max_displacement)
    min_displacements.append(min_displacement)
    print(np.shape(init_vertices[:,0:3] - final_vertices[:,0:3]))
    displacement = np.sum(np.linalg.norm(init_vertices[:,0:3] - final_vertices[:,0:3], axis=1))
    displacement = displacement / np.shape(init_vertices)[0]
    displacements.append(displacement)
    print("tree %s avg displacement: "%tree + str(displacement))

displacement_min = min(displacements)
displacement_max = max(displacements)
print("avg displacement between: %s and %s"%(displacement_min, displacement_max))
print("smallest displacement overall: %s"%min(min_displacements))
print("largest displacement overall: %s"%max(max_displacements))

#Values for Interesting single tree deformations:
# upper bound: 4-6
# lower bound: 0
# findings: - avg displacement usually falls between 1 and 2
#	    - initial positions do vary, but the differences should be small enough
#	    - sometimes displacement spikes heavily. Might lead to bad datapoints. Think about pruning datapoints with displacement that is too high. 
# THESE VALUES MUST ALWAYS BE CALCULATED PER TREE!