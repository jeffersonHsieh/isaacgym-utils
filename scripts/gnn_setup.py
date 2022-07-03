import copy
import os
import random
from queue import LifoQueue, Queue
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, ResGatedGraphConv, TransformerConv
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

DATASET_DIR = "/mnt/hdd/jan-malte/12Nodes_by_tree/"
TREE_NUM = 10

seed = 0
np.random.seed(seed)
random.seed(seed)

######## NETWORK DEF START ########

class GCNResidualBlock(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gcn_conv = GCNConv(hidden_size, hidden_size) # (hidden, num_out_features_per_node)
        self.linear = torch.nn.Linear(hidden_size, hidden_size)
  
    def forward(self, x, edge_index):
        x_block = self.gcn_conv(x, edge_index)
        x_block = F.relu(x_block)
        x_block = self.linear(x_block)
        x_block = F.relu(x_block)
        return x_block+x

class FGCNResidualBlock(torch.nn.Module):
    def __init__(self, hidden_size, p):
        self.block = GCNResidualBlock(hidden_size)
        self.bn = torch.nn.BatchNorm1d(hidden_size)
        self.do = torch.nn.Dropout(p)
    
    def forward(self, x, edge_index):
        x = self.block(x, edge_index)
        x = self.bn(x)
        x = self.do(x)
        return x

class FGCN(torch.nn.Module):
    def __init__(self, n_graph_nodes):
        super().__init__()
        hidden_size = 1280 #might need to be adjusted?
        p = 0.4
        self.stem = torch.nn.Sequential(
            torch.nn.Linear(10, hidden_size),
            torch.nn.ReLU(),
        )

        for idx in range(0,n_graph_nodes):
            self.add_module(FGCNResidualBlock(hidden_size,p), "block%s"%idx)

        self.out = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 7)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.stem(x)

        for block in self.children():
            x = block(x, edge_index)

        x = self.out(x)        
        return x

# Residual connection
# Needs to be adjusted to allow for larger number (>5) of Tree Nodes 
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 1280 #might need to be adjusted?
        p = 0.4
        self.stem = torch.nn.Sequential(
            torch.nn.Linear(10, hidden_size),
            torch.nn.ReLU(),
        )

        self.block1 = GCNResidualBlock(hidden_size)
        self.bn1 = torch.nn.BatchNorm1d(hidden_size)
        self.do1 = torch.nn.Dropout(p)
        
        self.block2 = GCNResidualBlock(hidden_size)
        self.bn2 = torch.nn.BatchNorm1d(hidden_size)
        self.do2 = torch.nn.Dropout(p)
        
        self.block3 = GCNResidualBlock(hidden_size)
        self.bn3 = torch.nn.BatchNorm1d(hidden_size)
        self.do3 = torch.nn.Dropout(p)

        self.block4 = GCNResidualBlock(hidden_size)
        self.bn4 = torch.nn.BatchNorm1d(hidden_size)
        self.do4 = torch.nn.Dropout(p)

        self.block5 = GCNResidualBlock(hidden_size)
        self.bn5 = torch.nn.BatchNorm1d(hidden_size)
        self.do5 = torch.nn.Dropout(p)

        self.out = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 7)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.stem(x)
        x = self.block1(x, edge_index)
        x = self.bn1(x)
        x = self.do1(x)
        
        x = self.block2(x, edge_index)
        x = self.bn2(x)
        x = self.do2(x)
        
        x = self.block3(x, edge_index)
        x = self.bn3(x)
        x = self.do3(x)
        
        x = self.block4(x, edge_index)
        x = self.bn4(x)
        x = self.do4(x)
        
        x = self.block5(x, edge_index)
        x = self.bn5(x)
        x = self.do5(x)
                
        x = self.out(x)        
        return x

######## NETWORK DEF END ########         
##### FUNCTION DEF START #####

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

def train(model, optimizer, criterion, train_loader, epoch, device):
    model.train()
    running_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch.to(device)
        out = model(batch)
        loss = criterion(out[:,:7], batch.y[:,:7])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss/len(train_loader)
    if epoch%10==0:
        print('[EPOCH {}] Train loss: {}'.format(epoch, train_loss))
    return train_loss

def validate(model, criterion, val_loader, epoch, device):
    model.eval()
    running_l2_norm = 0
    num_graphs = 0
    for batch in val_loader:
        batch.to(device)
        out = model(batch)
        running_l2_norm += torch.sum(torch.norm(out[:,:3]-batch.y[:,:3], dim=1)).item()
        num_graphs+=out.size()[0]
        #loss = criterion(out[:,:3], batch.y[:,:3])
        #running_loss += loss.item()
    val_loss = running_l2_norm/num_graphs
    if epoch%10==0:
        print('[EPOCH {}] Validation loss: {}'.format(epoch, val_loss))
    return val_loss

def test(model, test_loader, device):
    model.eval()
    running_l2_norm = 0
    num_graphs = 0
    for batch in test_loader:
        batch.to(device)
        out = model(batch)
        running_l2_norm += torch.sum(torch.norm(out[:,:3]-batch.y[:,:3], dim=1))
        num_graphs+=out.size()[0]
        visualize_graph(out[:,:3], 
                        batch.y[:,:3], 
                        batch.x[:,:3], 
                        batch.edge_index, batch.force_node[0], 
                        batch.x[:,-3:])
    l2_norm = running_l2_norm/num_graphs
    print('Average node distance error: {}'.format(l2_norm))

def visualize_graph(X, Y, X_0, edge_index, force_node, force):
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

    ax.legend(custom_lines, ['Input', 'GT', 'Predicted'])
    
    
    ax = set_axes_equal(ax)
    plt.show()

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

def get_topological_order(neighbor_dict, root=0):
    """
    Find the topological order of the tree.
    :param neighbor_dict: dict (key:int, val: set); the dictionary saving neighbor nodes.
    :param root: int; the root node index.
    :return topological_order: list of int; the topological order of the tree (start from root).
    """
    topological_order = []
    queue = LifoQueue()
    queue.put(root)
    expanded = [False] * len(neighbor_dict)
    while not queue.empty():
        src = queue.get()
        topological_order.append(src)
        expanded[src] = True
        for tgt in neighbor_dict[src]:
            if not expanded[tgt]:
                queue.put(tgt)
    return topological_order

def get_parents(neighbor_dict, root = 0):
    """
    Find the parents of each node in the tree.
    :param neighbor_dict: dict (key:int, val: set); the dictionary saving neighbor nodes.
    :param root: int; the root node index.
    :return parents: list of int; the parent indices of each node in the tree.
    """
    parents = [None] * len(neighbor_dict)
    parents[root] = -1
    queue = Queue()
    queue.put(root)
    while not queue.empty():
        src = queue.get()
        for tgt in neighbor_dict[src]:
            if parents[tgt] is None:
                parents[tgt] = src
                queue.put(tgt)
    return parents

def get_trunk(parents, leaf, root=0):
    """
    Get the trunk of the tree from leaf to root.
    :param parents: list of int; the parent indices of each node in the tree.
    :param leaf: int; the leaf node.
    :param root: int; the root node.
    :return trunk: set of tuple of int; the set of trunk edges.
    """
    trunk = set([])
    tgt = leaf
    while tgt != root:
        trunk.add((tgt, parents[tgt]))
        tgt = parents[tgt]
    return trunk

def make_directed_and_prune_augment(X_edges, X_force, X_pos, Y_pos, make_directed=True, prune_augmented=True):
    """
    Make the dataset edge connections directed and augment the dataset by random pruning.
    Note that this function assumes the input coming from graphs in same topology and same node ordering.
    :param X_edges: np.ndarray (n_edges, 2); the edge connection of the graph.
    :param X_force: np.ndarray (n_graphs, n_nodes, 3); the force applied on the graph.
    :param X_pos: np.ndarray (n_graphs, n_nodes, 3); the initial pose of the graph.
    :param Y_pos: np.ndarray (n_graphs, n_nodes, 3); the end pose of the graph.
    :param make_directed: bool; whether augment data by making it directed. If this is set to False, the
        graph is constructed as a undirected graph.
    :param prune_augmented: bool; whether augment the data by random pruning.
    :return:
        new_X_edges: np.ndarray (n_graphs, n_edges, 2); the edge connection of the augmented graph.
        new_X_force: np.ndarray (n_graphs, n_nodes, 3); the force applied on the graph.
        new_X_pos: np.ndarray (n_graphs, n_nodes, 3); the initial pose of the graph.
        new_Y_pos: np.ndarray (n_graphs, n_nodes, 3); the end pose of the graph.
    """
    num_graphs = len(X_force)
    # Construct a neighboring dictionary
    neighbor_dict = {}
    for src, tgt in X_edges:
        if src not in neighbor_dict:
            neighbor_dict[src] = set([tgt])
        else:
            neighbor_dict[src].add(tgt)
        if tgt not in neighbor_dict:
            neighbor_dict[tgt] = set([src])
        else:
            neighbor_dict[tgt].add(src)

    # Topologically sort the indices
    topological_order = get_topological_order(neighbor_dict)
    # Find the parents to the nodes
    parents = get_parents(neighbor_dict)

    # Data augmentation
    new_X_edges = []
    new_X_force = []
    new_X_pos = []
    new_Y_pos = []
    for i in range(num_graphs):
        # Find the node that force is applied on
        force_index = np.argwhere(np.sum(np.abs(X_force[i]), axis=1))[0,0]
        # Only keep the edges from force_index to the root
        trunk = get_trunk(parents, force_index)
        # Find leaf of the trunk
        trunk_nodes = set([])
        for edge in trunk:
            trunk_nodes.add(edge[0])
            trunk_nodes.add(edge[1])
        leaf_nodes = set([])
        for node in trunk_nodes:
            for child in neighbor_dict[node]:
                if child not in trunk_nodes:
                    leaf_nodes.add(child)
        # Add the directed/undirected graph without pruning
        if make_directed:
            edges = []
            for tgt in range(1, len(neighbor_dict)):
                src = parents[tgt]
                if src == 0:
                    edges.append([src, tgt])
                elif (tgt, src) in trunk:
                    edges.append([tgt, src])
                    edges.append([src, tgt])
                else:
                    edges.append([src, tgt])
            new_X_edges.append(np.array(edges))
        else:
            edges = []
            for tgt in range(1, len(neighbor_dict)):
                src = parents[tgt]
                edges.append([tgt, src])
                edges.append([src, tgt])
            new_X_edges.append(np.array(edges))
        new_X_force.append(X_force[i])
        new_X_pos.append(X_pos[i])
        new_Y_pos.append(Y_pos[i])
        # Add the graph with pruning
        if prune_augmented:
            for edge_size in range(len(trunk), len(neighbor_dict) - 1):
                # Get new tree edges
                add_size = edge_size - len(trunk)
                nodes = copy.copy(trunk_nodes)
                node_candidates = copy.copy(leaf_nodes)
                for _ in range(add_size):
                    new_node = random.sample(node_candidates, 1)[0]
                    nodes.add(new_node)
                    node_candidates.remove(new_node)
                    for child in neighbor_dict[new_node]:
                        if child not in nodes:
                            node_candidates.add(child)
                # Re-indexing while keeping root
                reindex_mapping = list(nodes)
                random.shuffle(reindex_mapping)
                root_index = reindex_mapping.index(0)
                reindex_mapping[root_index] = reindex_mapping[0]
                reindex_mapping[0] = 0
                inverse_mapping = [-1] * len(neighbor_dict)
                for new_idx, old_idx in enumerate(reindex_mapping):
                    inverse_mapping[old_idx] = new_idx
                # Add edges to the dataset
                if make_directed:
                    edges = []
                    for tgt in nodes:
                        src = parents[tgt]
                        new_src = inverse_mapping[src]
                        new_tgt = inverse_mapping[tgt]
                        if src == -1:
                            pass
                        elif src == 0:
                            edges.append([new_src, new_tgt])
                        elif (tgt, src) in trunk:
                            edges.append([new_tgt, new_src])
                            edges.append([new_src, new_tgt])
                        else:
                            edges.append([new_src, new_tgt])
                    new_X_edges.append(np.array(edges))
                else:
                    edges = []
                    for tgt in nodes:
                        src = parents[tgt]
                        new_src = inverse_mapping[src]
                        new_tgt = inverse_mapping[tgt]
                        if src == -1:
                            pass
                        else:
                            edges.append([new_tgt, new_src])
                            edges.append([new_src, new_tgt])
                    new_X_edges.append(np.array(edges))
                # Add to the dataset
                new_X_force.append(X_force[i][reindex_mapping])
                new_X_pos.append(X_pos[i][reindex_mapping])
                new_Y_pos.append(Y_pos[i][reindex_mapping])
    return new_X_edges, new_X_force, new_X_pos, new_Y_pos

def rotate_augment(X_edges, X_force, X_pos, Y_pos, rotate_augment_factor=5, stddev_x_angle=0.2, stddev_y_angle=0.2):
    """
    Augment the graph by random rotation.
    :param X_edges: np.ndarray (n_graphs, n_edges, 2); the edge connection of the graph.
    :param X_force: np.ndarray (n_graphs, n_nodes, 3); the force applied on the graph.
    :param X_pos: np.ndarray (n_graphs, n_nodes, 3); the initial pose of the graph.
    :param Y_pos: np.ndarray (n_graphs, n_nodes, 3); the end pose of the graph.
    :param rotate_augment_factor: int; number of random rotation per graph.
    :param stddev_x_angle: float; the stddev of random rotation in x direction.
    :param stddev_y_angle: float; the stddev of random rotation in y direction.
    :return:
        new_X_edges: np.ndarray (n_graphs, n_edges, 2); the edge connection of the augmented graph.
        new_X_force: np.ndarray (n_graphs, n_nodes, 3); the force applied on the graph.
        new_X_pos: np.ndarray (n_graphs, n_nodes, 3); the initial pose of the graph.
        new_Y_pos: np.ndarray (n_graphs, n_nodes, 3); the end pose of the graph.
    """
    num_graphs = len(X_force)
    # Augment the data by rotation
    new_X_edges = []
    new_X_force = []
    new_X_pos = []
    new_Y_pos = []
    new_X_quat = []
    new_Y_quat = []
    for i in range(num_graphs):
        X_edge = X_edges[i]
        for _ in range(rotate_augment_factor):
            theta_x = np.random.normal(0., stddev_x_angle)
            theta_y = np.random.normal(0., stddev_y_angle)
            theta_z = np.random.uniform(0., 2. * np.pi)
            R = Rotation.from_euler('zyx', [theta_z, theta_y, theta_x]).as_matrix()
            
            X_R = Rotation.from_quat(X_pos[i][:,3:]).as_matrix()
            Y_R = Rotation.from_quat(X_pos[i][:,3:]).as_matrix()
            
            X_R = R@X_R
            Y_R = R@Y_R
            
            X_q = Rotation.from_matrix(X_R).as_quat()
            Y_q = Rotation.from_matrix(Y_R).as_quat()
            
            X_pos_quat = np.concatenate((np.dot(R, X_pos[i][:,:3].T).T, X_q), axis=1)
            Y_pos_quat = np.concatenate((np.dot(R, Y_pos[i][:,:3].T).T, Y_q), axis=1)
            
            new_X_edges.append(X_edge)
            new_X_force.append(np.dot(R, X_force[i].T).T)
            
            
            new_X_pos.append(X_pos_quat)
            new_Y_pos.append(Y_pos_quat)
            
    return new_X_edges, new_X_force, new_X_pos, new_Y_pos

def load_npy(data_dir, tree_num):
    # Load npy files from dataset_dir. A shortcut to 'sample_1_push' shared folder has been added to 'My Drive' 
    X_stiffness_damping = np.load(os.path.join(data_dir, 'X_coeff_stiff_damp_tree%s.npy'%tree_num))
    X_edges = np.load(os.path.join(data_dir, 'X_edge_def_tree%s.npy'%tree_num))
    X_force = np.load(os.path.join(data_dir, 'X_force_applied_tree%s_clean.npy'%tree_num))
    X_pos = np.load(os.path.join(data_dir, 'X_vertex_init_tree%s_clean.npy'%tree_num))
    Y_pos = np.load(os.path.join(data_dir, 'Y_vertex_final_tree%s_clean.npy'%tree_num))

    # Truncate node orientations and tranpose to shape (num_graphs, num_nodes, 3)
    X_pos = X_pos[:, :7, :].transpose((0,2,1))
    Y_pos = Y_pos[:, :7, :].transpose((0,2,1))
    X_force = X_force.transpose((0,2,1))

    return X_edges, X_force, X_pos, Y_pos

def make_dataset(X_edges, X_force, X_pos, Y_pos, 
                 make_directed=True, prune_augmented=False, rotate_augmented=False):
    num_graphs = len(X_pos)
    X_edges, X_force, X_pos, Y_pos = make_directed_and_prune_augment(X_edges, X_force, X_pos, Y_pos,
                                                                     make_directed=make_directed, 
                                                                     prune_augmented=prune_augmented)
    if rotate_augmented:
        X_edges, X_force, X_pos, Y_pos = rotate_augment(X_edges, X_force, X_pos, Y_pos)

    num_graphs = len(X_pos)
    dataset = []
    for i in range(num_graphs): 
        # Combine all node features: [position, force, stiffness] with shape (num_nodes, xyz(3)+force(3)+stiffness_damping(4)) 
        # stiffness damping is (4) because of bending stiffness/damping and torsional stiffness/damping
        root_feature = np.zeros((len(X_pos[i]), 1))
        #root_feature[0, 0] = 1.0
        #X_data = np.concatenate((X_pos[i], X_force[i], root_feature), axis=1) # TODO: Add stiffness damping features later
        X_data = np.concatenate((X_pos[i], X_force[i]), axis=1) # TODO: Add stiffness damping features later

        edge_index = torch.tensor(X_edges[i].T, dtype=torch.long)
        x = torch.tensor(X_data, dtype=torch.float)
        y = torch.tensor(Y_pos[i], dtype=torch.float)
        force_node = np.argwhere(np.sum(np.abs(X_force[i]), axis=1))[0,0]
        graph_instance = Data(x=x, edge_index=edge_index, y=y, force_node=force_node)
        dataset.append(graph_instance)
    return dataset

def shuffle_in_unison(a,b,c):
    assert len(a)==len(b)==len(c)
    order = np.arange(len(a))
    np.random.shuffle(order)
    return a[order],b[order],c[order]

##### FUNCTION DEF END #####

# Loading Datase

d = DATASET_DIR # Assumes all trees in one dataset (no Iteration needed)

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
    dataset_tree = make_dataset(X_edges, X_force_arr, X_pos_arr, Y_pos_arr, 
                                make_directed=True, prune_augmented=False, rotate_augmented=False)
    dataset = dataset + dataset_tree

print(np.shape(dataset))
# Shuffle Dataset
#X_force_arr, X_pos_arr, Y_pos_arr = shuffle_in_unison(X_force_arr, X_pos_arr, Y_pos_arr)
random.shuffle(dataset)

# setup validation/test split
train_val_split = int(len(dataset)*0.9)

train_dataset = dataset[:train_val_split]
val_dataset = dataset[train_val_split:]
#X_force_train = X_force_arr[:train_val_split] 
#X_pos_train = X_pos_arr[:train_val_split] 
#Y_pos_train = Y_pos_arr[:train_val_split] 

#X_force_val = X_force_arr[train_val_split:] 
#X_pos_val = X_pos_arr[train_val_split:] 
#Y_pos_val = Y_pos_arr[train_val_split:] 

# generate validation/test datasets
#train_dataset = make_dataset(X_edges, X_force_train, X_pos_train, Y_pos_train, 
#                 make_directed=True, prune_augmented=False, rotate_augmented=False)
#val_dataset = make_dataset(X_edges, X_force_val, X_pos_val, Y_pos_val, 
#                 make_directed=True, prune_augmented=False, rotate_augmented=False)

# check validation dataset?
#print(val_dataset[0])
#print(val_dataset[0].edge_index)

batch_size=256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

#check train dataset?
#train_data = train_dataset[0]
#print(train_data)
#print()
print('Number of Graphs in Train Dataset: ', len(train_dataset))
print('Number of Graphs in Test Dataset: ', len(val_dataset))
print()

# check batches
for batch in train_loader:
    print(batch)
    print('Number of graphs in batch: ', batch.num_graphs) 

for i in range(10): #Investigate this function. Generates oddly shaped "trees". find out wether this loop is at fault or if the dataset is imported wrong
    X = val_dataset[i].x[:,:3]
    #print(val_dataset[i].x[:,:3])
    Y = val_dataset[i].y[:,:3]
    #print(val_dataset[i].y[:,:3])
    force_node = val_dataset[i].force_node
    #print(force_node)
    edge_index = val_dataset[i].edge_index
    #print(edge_index)
    force = val_dataset[i].x[:,-3:]
    visualize_graph(X, Y, X, edge_index, force_node, force)

fig = plt.figure()
ax = fig.add_subplot(111)

# Setup GCN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
criterion = torch.nn.MSELoss()
#criterion = torch.nn.L1Loss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5, min_lr=5e-4)

# Train and validate model
train_loss_history = []
val_loss_history = []
best_loss = 1e9
for epoch in range(1, 2500):
    train_loss = train(model, optimizer, criterion, train_loader, epoch, device) #train model
    val_loss = validate(model, criterion, val_loader, epoch, device) # validate model
    if val_loss<best_loss:
        best_loss=val_loss
        best_model = copy.deepcopy(model)
    scheduler.step(best_loss)
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    #if epoch%10==0:
    #    print(scheduler._last_lr)
    
    ax.clear()
    ax.plot(train_loss_history, 'r', label='train')
    ax.plot(val_loss_history, 'b', label='validation')
    ax.legend(loc="upper right")
    #ax.set_ylim([0, 0.2])
    display(fig)
    clear_output(wait=True)

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#best_model = GCN().to(device)
#best_model.load_state_dict(torch.load('model_173_seed0.pt'))

# evaluate best model and save the state dict
best_model.eval()
test(best_model, test_loader, device)

torch.save(best_model.state_dict(), 'model_157_quat.pt')