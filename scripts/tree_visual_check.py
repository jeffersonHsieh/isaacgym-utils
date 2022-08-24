import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import os
import numpy as np
import torch
from torch_geometric.data import Data
import random

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

def make_dataset(X_edges, X_force, X_pos, Y_pos): # CALLED PER TREE
    num_graphs = len(X_pos)
    dataset = []
    for i in range(num_graphs): 
        # Combine all node features: [position, force, stiffness] with shape (num_nodes, xyz(3)+force(3)+stiffness_damping(4)) 
        # stiffness damping is (4) because of bending stiffness/damping and torsional stiffness/damping
        root_feature = np.zeros((len(X_pos[i]), 1))
        #root_feature[0, 0] = 1.0
        #X_data = np.concatenate((X_pos[i], X_force[i], root_feature), axis=1) # TODO: Add stiffness damping features later
        X_data = np.concatenate((X_pos[i], X_force[i]), axis=1) # TODO: Add stiffness damping features later

        edge_index = torch.tensor(X_edges.T, dtype=torch.long)
        x = torch.tensor(X_data, dtype=torch.float)
        y = torch.tensor(Y_pos[i], dtype=torch.float)
        force_node = np.argwhere(np.sum(np.abs(X_force[i]), axis=1))[0,0]
        graph_instance = Data(x=x, edge_index=edge_index, y=y, force_node=force_node)
        dataset.append(graph_instance)
    return dataset

def load_npy(data_dir, tree_num):
    # Load npy files from dataset_dir. A shortcut to 'sample_1_push' shared folder has been added to 'My Drive' 
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

def visualize_graph(X, Y, X_0, edge_index, force_node, force, name):
    force = force.detach().cpu().numpy()
    
    force_vector = force[force_node]/np.linalg.norm(force[force_node])/2
    force_A = X_0.detach().cpu().numpy()[force_node]
    force_B = X_0.detach().cpu().numpy()[force_node] + force_vector*2
    
    
    x_0 = []
    x_edges = []
    y_edges = []
    for edge in edge_index.T:
        x_0.append([X_0[edge[0]].detach().cpu().numpy(), X_0[edge[1]].detach().cpu().numpy()])
        x_edges.append([X[edge[0]].detach().cpu().numpy(), X[edge[1]].detach().cpu().numpy()])
        y_edges.append([Y[edge[0]].detach().cpu().numpy(), Y[edge[1]].detach().cpu().numpy()])
    x_0 = np.array(x_0)
    x_edges = np.array(x_edges)
    y_edges = np.array(y_edges)

    
    ax = plt.figure().add_subplot(projection='3d')
    fn = X_0[force_node].detach().cpu().numpy()
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
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([0, 3])
    
    custom_lines = [Line2D([0], [0], color=[0,0,1,1], lw=2),
                    Line2D([0], [0], color=[1,0,0,1], lw=4),
                    Line2D([0], [0], color=[0,1,0,1], lw=4)]

    ax.legend(custom_lines, ['Input', 'Predicted', 'GT'])
    
    
    ax = set_axes_equal(ax)
    plt.tight_layout()
    plt.savefig(name)
    plt.show()

TREE_NUM = 10
d = "/mnt/hdd/jan-malte/test_set_by_tree/"

dataset = []
for tree in range(0, TREE_NUM):
    X_force_list = []
    X_pos_list = []
    Y_pos_list = []
    X_edges, X_force, X_pos, Y_pos = load_npy(d, tree)
    X_force_list.append(X_force)
    X_pos_list.append(X_pos)
    Y_pos_list.append(Y_pos)
    X_force_arr = np.concatenate(X_force_list)
    X_pos_arr = np.concatenate(X_pos_list)
    Y_pos_arr = np.concatenate(Y_pos_list)
    dataset_tree = make_dataset(X_edges, X_force_arr, X_pos_arr, Y_pos_arr)
    dataset = dataset + dataset_tree

results_path = "/mnt/hdd/jan-malte/test_set_by_tree/inspection"
os.mkdir(results_path)
results_path = results_path+"/"

for _ in range(30):
    i = random.randint(0, len(dataset))
    print("shown tree index: %s"%i)
    X = dataset[i].x[:,:3]
    Y = dataset[i].y[:,:3]
    force_node = dataset[i].force_node
    print_edges = dataset[i].edge_index
    force = dataset[i].x[:,-3:]
    visualize_graph(X, Y, X, print_edges, force_node, force, results_path+"tree_push%s"%i)

fig = plt.figure()
ax = fig.add_subplot(111)