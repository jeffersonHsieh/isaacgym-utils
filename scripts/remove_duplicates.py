import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import Axes3D

def remove_duplicate_nodes(edges, init_positions, final_positions, X_force):
    tree_representative = init_positions[0]
    tree_representative = np.around(tree_representative, decimals=4)
    duplicates = [(0,1)] #treat 0 and 1 as duplicates, as 0 represents the base_link aka the floor, which should behave like the root
    for i, node in enumerate(tree_representative):
        for j, nodec in enumerate(tree_representative):
            if (node[:3] == nodec[:3]).all() and i != j and has_same_parent(i,j,edges):
                if i < j:
                    duplicates.append((i,j))
                else:
                    duplicates.append((j,i))
    duplicates = list(set(duplicates))
    while len(duplicates) > 0:
        original, duplicate = duplicates.pop()
        edges, init_positions, final_positions, duplicates, X_force = remove_duplicate(original, duplicate, edges, init_positions, final_positions, duplicates, X_force)
        duplicates = list(set(duplicates))
        duplicates = adjust_indexing(duplicates, duplicate)
        edges = adjust_indexing(edges, duplicate)
    edges = np.array(edges)
    return edges, init_positions[:,:,:3], final_positions[:,:,:3], X_force

def has_same_parent(i,j,edges):
    for parent, child in edges:
        if child == i:
            parent_i = parent
        if child == j:
            parent_j = parent
    return parent_i == parent_j

def remove_duplicate(original, duplicate, edge_def, init_positions, final_positions, duplicates, forces):
    init_positions = np.delete(init_positions, duplicate, axis=1)
    final_positions = np.delete(final_positions, duplicate, axis=1)

    new_edge_def = []
    new_duplicates = []
    for orig, dup in duplicates:
        if orig == duplicate:
            new_duplicates.append((original,dup))
        elif duplicate != dup and duplicate != orig:
            new_duplicates.append((orig,dup))

    for parent, child in edge_def:
        if duplicate == parent:
            new_edge_def.append((original,child))
        elif duplicate != parent and duplicate != child:
            new_edge_def.append((parent,child))

    for idx, force in enumerate(forces):
        if np.linalg.norm(force[duplicate]) != 0:
            forces[idx][original] += forces[idx][duplicate]
    
    forces = np.delete(forces, duplicate, axis=1)

    return new_edge_def, init_positions, final_positions, new_duplicates, forces

def adjust_indexing(tuple_list, deleted_index):
    new_tuple_list = []
    for i, j in tuple_list:
        if i > deleted_index:
            i = i-1
        if j > deleted_index:
            j = j-1
        new_tuple_list.append((i,j))
    return new_tuple_list

def load_npy(data_dir, sim=True):
    if sim:
        # Load npy files from dataset_dir. A shortcut to 'sample_1_push' shared folder has been added to 'My Drive' 
        #X_stiffness_damping = np.load(os.path.join(data_dir, 'X_coeff_stiff_damp.npy'))
        X_edges = np.load(os.path.join(data_dir, 'X_edge_def.npy'))
        X_force = np.load(os.path.join(data_dir, 'final_F.npy'))
        X_pos = np.load(os.path.join(data_dir, 'final_X.npy'))
        Y_pos = np.load(os.path.join(data_dir, 'final_Y.npy'))
        # Truncate node orientations and tranpose to shape (num_graphs, num_nodes, n_features)
        X_pos = X_pos[:, :7, :].transpose((0,2,1))
        Y_pos = Y_pos[:, :7, :].transpose((0,2,1))
        X_force = X_force.transpose((0,2,1))
    else:
        X_edges = np.load(os.path.join(data_dir, 'X_edge_def.npy'))
        X_force = np.load(os.path.join(data_dir, 'final_F.npy'))
        X_pos = np.load(os.path.join(data_dir, 'final_X.npy'), allow_pickle=True)
        Y_pos = np.load(os.path.join(data_dir, 'final_Y.npy'), allow_pickle=True)

        invalid_graphs = []
        for i, graph in enumerate(X_pos):
            for j, node in enumerate(graph):
                if node is None:
                    invalid_graphs.append(i)

        X_force = np.delete(X_force, invalid_graphs, axis=0)
        X_pos = np.delete(X_pos, invalid_graphs, axis=0)
        Y_pos = np.delete(Y_pos, invalid_graphs, axis=0)
        
        X_pos_list = []
        for graph in X_pos:
            for node in graph:
                for feature in node:
                    X_pos_list.append(feature)
        X_pos = np.array(X_pos_list)
        X_pos = X_pos.reshape(Y_pos.shape[0],Y_pos.shape[1],Y_pos.shape[2]) 
        X_force = X_force.transpose((0,2,1))
    return X_edges, X_force, X_pos, Y_pos

def visualize_graph(X, Y, X_0, edge_index, force_node, force):
    #force = force.detach().cpu().numpy()
    
    force_vector = force[force_node]/np.linalg.norm(force[force_node])/2
    force_A = X_0[force_node]
    force_B = X_0[force_node] + force_vector*2
    
    
    x_0 = []
    x_edges = []
    y_edges = []
    for edge in edge_index.T:
        x_0.append([X_0[edge[0]], X_0[edge[1]]])
        x_edges.append([X[edge[0]], X[edge[1]]])
        y_edges.append([Y[edge[0]], Y[edge[1]]])
    x_0 = np.array(x_0)
    x_edges = np.array(x_edges)
    y_edges = np.array(y_edges)

    
    ax = plt.figure().add_subplot(projection='3d')
    fn = X_0[force_node]
    ax.scatter(fn[0], fn[1], fn[2], c='m', s=50)
    x0_lc = Line3DCollection(x_0, colors=[0,0,1,1], linewidths=1)
    x_lc = Line3DCollection(x_edges, colors=[1,0,0,1], linewidths=5)
    y_lc = Line3DCollection(y_edges, colors=[0,1,0,1], linewidths=5)
    ax.add_collection3d(x0_lc)
    ax.add_collection3d(x_lc)
    ax.add_collection3d(y_lc)
    
    arrow_prop_dict = dict(mutation_scale=30, arrowstyle='-|>', color='m', shrinkA=0, shrinkB=0)
    a = Arrow3D([force_A[0], force_B[0]], 
                [force_A[1], force_B[1]], 
                [force_A[2], force_B[2]], **arrow_prop_dict)
    ax.add_artist(a)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])
    ax.set_zlim([0, 2.0])
    
    custom_lines = [Line2D([0], [0], color=[0,0,1,1], lw=2),
                    Line2D([0], [0], color=[1,0,0,1], lw=4),
                    Line2D([0], [0], color=[0,1,0,1], lw=4)]

    ax.legend(custom_lines, ['Input', 'Predicted', 'GT'])
    
    
    ax = set_axes_equal(ax)
    plt.tight_layout()
    plt.show() 


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    return ax

d = "/home/mark/github/tree_is_all_you_need/data/10Nodes_by_tree/trial0/"

X_force_list = []
X_pos_list = []
Y_pos_list = []
X_edges, X_force, X_pos, Y_pos = load_npy(d)
print(np.shape(Y_pos))
print(np.shape(X_pos))
print(np.shape(X_force))
print(X_edges)
print(Y_pos[0])
visualize_graph(X_pos[1][:,:3], Y_pos[1][:,:3], X_pos[1][:,:3], X_edges, 0, [0,0,0])
X_edges, X_pos, Y_pos, X_force = remove_duplicate_nodes(X_edges, X_pos, Y_pos, X_force)
print(X_edges)
print(Y_pos[0])
print(np.shape(Y_pos))
print(np.shape(X_pos))
print(np.shape(X_force))
visualize_graph(X_pos[1], Y_pos[1], X_pos[1], X_edges, 0, [0,0,0])

