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
import wandb
import argparse
import datetime
from torch.autograd import Variable
import math
import GPUtil
from torch_geometric_temporal import GConvLSTM

TIP_THICKNESS = 0.01 #tip radius for branch radius reconstruction.
TREE_PTS = [8,10] #list of tree points to search for in prefixes. admissable tree sizes

#seed = 0
#np.random.seed(seed)
#random.seed(seed)

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
        super().__init__()
        self.block = GCNResidualBlock(hidden_size)
        self.bn = torch.nn.BatchNorm1d(hidden_size)
        self.do = torch.nn.Dropout(p)
    
    def forward(self, x, edge_index):
        x = self.block(x, edge_index)
        x = self.bn(x)
        x = self.do(x)
        return x

# TODO: Does not really work. still in development
class GConvLSTMBlock(torch.nn.Module):
    def __init__(self, hidden_size, p):
        super().__init__()
        K = 7 # TODO: CHANGE LATER ONCE I FIGURED OUT WHAT THAT DOES AND THIS APPROACH LOOKS PROMISING
        self.block = GConvLSTM(hidden_size, hidden_size, K)
        self.bn = torch.nn.BatchNorm1d(hidden_size)
        self.do = torch.nn.Dropout(p)

    def forward(self, x, edge_index, C=None):
        x, new_c = self.block(x, edge_index, C=C)
        x = self.bn(x)
        x = self.do(x)
        return x, new_c

# TODO: Does not really work. still in development
class FGCNRec(torch.nn.Module):
    def __init__(self, n_layers, in_size, out_size, hidden_size=1280):
        super().__init__()
        #hidden_size = 1280 
        p = 0.4
        self.stem = torch.nn.Sequential(
            torch.nn.Linear(in_size, hidden_size),  
            torch.nn.ReLU(),
        ) 

        self.blocks = torch.nn.ModuleList([GConvLSTMBlock(hidden_size,p) for _ in range(0,n_layers)])

        self.out = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, out_size)
        ) 
        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        graph_size = 8 #int(len(data.x)/data.num_graphs)
        #print(graph_size)

        x = self.stem(x)

        idx = 0
        C = None
        while idx < graph_size:
            for block in self.blocks:
                x, C = block(x, edge_index, C=C)
            idx += 1

        x = self.out(x)        
        return x

class FGCN(torch.nn.Module):
    """
    flexibele GCN. allows for adjustable number of layers by setting n_graph_nodes
    """
    def __init__(self, n_graph_nodes, in_size, out_size, hidden_size=1280):
        super().__init__()
        #hidden_size = 1280 
        p = 0.4
        self.stem = torch.nn.Sequential(
            torch.nn.Linear(in_size, hidden_size),  
            torch.nn.ReLU(),
        ) 

        self.blocks = torch.nn.ModuleList([FGCNResidualBlock(hidden_size,p) for _ in range(0,n_graph_nodes)])

        self.out = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, out_size)
        ) 
        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.stem(x)

        idx = 0
        for block in self.blocks:
            x = block(x, edge_index)

        x = self.out(x)        
        return x

# Residual connection 
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 1280 
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

class QuatLoss(torch.nn.Module):
    """
    quaternion loss function
    """
    def __init__(self):
        super().__init__()
        #self.mse = torch.nn.MSELoss()
    
    def forward(self, quat_pred, quat_true):
        batch_size = len(quat_pred)

        ## Normalization
        divisor = torch.norm(quat_pred, dim=1)
        quat_pred = torch.transpose(quat_pred, 0, 1)
        norm_pred = torch.div(quat_pred, divisor)
        norm_pred = torch.transpose(norm_pred, 0, 1)

        ## loss calc
        dots = torch.bmm(norm_pred.view(batch_size, 1, 4), quat_true.view(batch_size, 4, 1))
        square_dots = torch.square(dots.view(batch_size))
        losses = torch.sub(torch.ones(batch_size).to(device), square_dots)

        loss = torch.sum(losses)/len(losses)
        return loss


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

class TreeTrunctuator():
    def __init__(self, graph):
        self.graph = graph
        self.new_graph = self.generate_trunk_graph(graph)
        self.index_translation_dict_to = {} # keys are initial indices, values are final indices
        self.index_translation_dict_from = {} # keys are final indices, values are initial indices

    def parents(node, edges):
        parents = []
        for parent, child in edges:
            if child == node:
                parents.append(parent)
        return parents

    def generate_trunk_graph(graph):
        force_node = graph.force_node
        X_vert = []
        Y_vert = []
        tmp_edges = []
        edges = []
        frontier = [force_node]
        explored = []
        while len(frontier) != 0:
            current_node = frontier.pop()
            parent = parents(current_node, graph.edges)[0] # should only have one parent, since only the trunk has two sided connections
            current_idx = len(X_vert)

            self.index_translation_dict_to[current_node] = current_idx
            self.index_translation_dict_from[current_idx] = current_node
            
            X_vert.append(graph.x[current_node])
            Y_vert.append(graph.y[current_node])

            if parent is not None: # edges will need to be translated later
                tmp_edges.append([parent, current_node])
                tmp_edges.append([current_node, parent])
                if parent not in explored:
                    frontier.append(parent)
        
        for parent, child in tmp_edges: # translate edge dictionary
            edges.append([self.index_translation_dict_to[parent], self.index_translation_dict_to[child]])


# 0: no graph alteration
# 1: transform into node representation (graph nodes are tree points)
# 2: transform into node representation with thickness parameter
# 3: load as orientational representation
# 4: load as orientational representation with rpy angles
def train(model, optimizer, criterion, train_loader, epoch, device, out_size, profile):
    model.train()
    running_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch.to(device)
        out = model(batch)
        if profile == 3 or profile == 4 or profile == 5: # if we learn orientation we handle input into criterion fuction differently, since we have different expectations
            out_end = 3 + out_size
            loss = criterion(out[:,:], batch.y[:,3:out_end]) # assumes that xyz are the first 3 dimensions of y
        else:
            loss = criterion(out[:,:], batch.y[:,:out_size])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss/len(train_loader)
    if epoch%10==0:
        print('[EPOCH {}] Train loss: {}'.format(epoch, train_loss))
    return train_loss

def first_value(e):
    return e[0]

def print_loss_by_tree(base, val):
    """
    generates a scatter plot with base displacement and prediction error as its dimenstion. Each point is one push prediction
    """
    zipped = list(zip(base, val))
    zipped.sort(key=first_value) 

    base = [x for (x,y) in zipped]
    val = [y for (x,y) in zipped]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(base, val, s=4)
    ax.plot(base, base, color="orange")
    ax.set_xlabel("base loss")
    ax.set_ylabel("validation loss")
    #display(fig)
    plt.savefig(results_path+"base_vs_loss_by_tree")
    plt.close()
    clear_output(wait=True)

def validate(model, criterion, val_loader, epoch, device, profile, out_size):
    model.eval()
    running_l2_norm = 0
    running_l2_norm_base = 0
    num_graphs = 0
    base_loss_by_tree_list = []
    val_loss_by_tree_list = []
    ori_end = 3 + out_size # last index of the orientation segment of each incoming datapoint

    for batch in val_loader:
        batch.to(device)
        out = model(batch)
        # does not work for mixed tree complexities, since its impossible to tell how to reshape the trees. Move to test function instead
        #if epoch%10 == 0:
        #    y_tree_array = batch.y.detach().cpu().numpy()
        #    x_tree_array = batch.x.detach().cpu().numpy()
        #    pred_array = out.detach().cpu().numpy()
        #    
        #    y_shape = np.shape(y_tree_array)
        #    x_shape = np.shape(x_tree_array)
        #    pred_shape = np.shape(pred_array)

        #    if profile == 3 or profile == 4 or profile == 5:
        #        y_tree_array = np.reshape(y_tree_array, (batch.num_graphs,int(len(batch.y)/batch.num_graphs), y_shape[1])) 
        #        x_tree_array = np.reshape(x_tree_array, (batch.num_graphs,int(len(batch.y)/batch.num_graphs), x_shape[1])) 
        #        pred_array = np.reshape(pred_array, (batch.num_graphs,int(len(batch.y)/batch.num_graphs), pred_shape[1]))

        #        base_loss_by_tree = []
        #        val_loss_by_tree = []
        #        for tree_idx in range(0,len(y_tree_array)):
        #            base_loss_by_tree.append(criterion(torch.tensor(x_tree_array[tree_idx,:,3:ori_end]).to(device), torch.tensor(y_tree_array[tree_idx,:,3:ori_end]).to(device))) 
        #            val_loss_by_tree.append(criterion(torch.tensor(pred_array[tree_idx]).to(device), torch.tensor(y_tree_array[tree_idx,:,3:ori_end]).to(device)))
        #        base_loss_by_tree = torch.tensor(base_loss_by_tree)
        #        val_loss_by_tree = torch.tensor(val_loss_by_tree)
        #    else:
        #        y_tree_array = np.reshape(y_tree_array, (batch.num_graphs,int(len(batch.y)/batch.num_graphs),y_shape[1])) # only works if we just take positions into account (no orientation)
        #        x_tree_array = np.reshape(x_tree_array, (batch.num_graphs,int(len(batch.y)/batch.num_graphs),x_shape[1])) # only works if we just take positions into account (no orientation)
        #        pred_array = np.reshape(pred_array, (batch.num_graphs,int(len(batch.y)/batch.num_graphs),pred_shape[1])) # only works if we just take positions into account (no orientation)

        #        base_loss_by_tree = np.sum(np.linalg.norm(x_tree_array[:,:,:3] - y_tree_array[:,:,:3], axis=2), axis=1)*batch.num_graphs/len(batch.y)
        #        val_loss_by_tree = np.sum(np.linalg.norm(pred_array[:,:,:] - y_tree_array[:,:,:3], axis=2), axis=1)*batch.num_graphs/len(batch.y)

        #    base_loss_by_tree_list = base_loss_by_tree_list + base_loss_by_tree.tolist()
        #    val_loss_by_tree_list = val_loss_by_tree_list + val_loss_by_tree.tolist()

        if profile == 3 or profile == 4 or profile == 5:
            running_l2_norm += criterion(out[:,:], batch.y[:,3:ori_end]).item()
            running_l2_norm_base += criterion(batch.x[:,3:ori_end], batch.y[:,3:ori_end]).item()   
        else:
            running_l2_norm += torch.sum(torch.norm(out[:,:] - batch.y[:,:out_size], dim=1)).item()
            running_l2_norm_base += torch.sum(torch.norm(batch.x[:,:out_size] - batch.y[:,:out_size], dim=1)).item()
            num_graphs+=out.size()[0]

    # print validation error per tree
    #if epoch%10 == 0:
    #    print_loss_by_tree(base_loss_by_tree_list, val_loss_by_tree_list, epoch)

    # average the loss functions  
    if profile == 3 or profile == 4 or profile == 5:
        val_loss = running_l2_norm/len(val_loader)
        base_loss = running_l2_norm_base/len(val_loader)
    else:
        val_loss = running_l2_norm/num_graphs
        base_loss = running_l2_norm_base/num_graphs

    # print loss
    if epoch%10==0:
        print('[EPOCH {}] Validation loss: {}'.format(epoch, val_loss))
        print("[EPOCH {}] Baseline loss: {}".format(epoch, base_loss))
    return val_loss, base_loss

def assign_position(node_index, original_tree, rotations, parent_pos, parent_rot, len_idx):
    rotations = torch.Tensor(rotations)
    distance = original_tree[node_index][len_idx]
    divisor = torch.norm(rotations[node_index], dim=0)
    quat = torch.div(rotations[node_index], divisor)
    rot_mat = Rotation.from_quat(quat) * parent_rot
    new_pos = parent_pos + rot_mat.as_matrix().dot(np.array([0,0,distance]))
    return new_pos, rot_mat

def get_parent(node, edges):
    for parent, child in edges:
        if child == node:
            return parent
    return None

def find_root(edges):
    for idx, _ in edges:
        parent = edges[idx][0]
        child = edges[idx][1]
        if get_parent(parent, edges) is None:
            return parent
    return None

def reconstruct_positional_data(rotations, edges, init_pos, rpy=False):
    len_idx = 7
    rot_shape = np.shape(rotations)
    if rot_shape[1] == 3 or rpy:
        len_idx = 6
        rotations = np.reshape(rotations,(-1,3))
        rotations = rot_from_rpy(rotations) # if we get rotations in rpy transform them into quaternion
    new_out = [[0,0,0]]*len(init_pos)
    help_edges = []
    for idx, _ in enumerate(edges[0]):
        help_edges.append((edges[0][idx], edges[1][idx]))
    
    frontier = [0] #ASSUMPTION: Root node is always indexed with 0
    explored = []
    parent_rots = {0:Rotation.from_quat(rotations[0])}
    while len(frontier) != 0:
        parent = frontier.pop(0)
        explored.append(parent)
        children = find_children(help_edges, parent)
        parent_pos = new_out[parent]
        for child in children:
            if child not in explored:
                pos, rot = assign_position(child, init_pos, rotations, parent_pos, parent_rots[parent], len_idx) #.detach().cpu().numpy()
                new_out[child] = list(pos)
                parent_rots[child] = rot
                frontier.append(child)

    new_out = torch.tensor(new_out)
    return new_out  

def test(model, test_loader, device, profile, out_size):
    model.eval()
    running_l2_norm_by_root_dist = {}
    root_dist_accumulator = {}
    running_l2_base_by_root_dist = {}
    l2_norms = []
    l2_norm_base = []
    l2_norm_averaged = []
    l2_norm_base_averaged = []
    running_l2_norm = 0
    running_l2_norm_base = 0
    num_graphs = 0
    idx = 0
    ori_end = 3+out_size
    for batch in test_loader:
        root_dist_dict = generate_root_distance_dict(batch.edge_index.detach().cpu().numpy().T)
        batch.to(device)
        with torch.no_grad():
            out = model(batch)
        y = batch.y.detach()
        x = batch.x.detach()
        force = batch.x[:,-3:].detach()
        if profile==3 or profile==4 or profile==5:
            roty = batch.y.detach().cpu().numpy()[:,3:ori_end]
            rotx = batch.x.detach().cpu().numpy()[:,3:ori_end]

            rpy = profile==4
            out = reconstruct_positional_data(out.detach().cpu().numpy(), batch.edge_index.detach().cpu().numpy(), batch.x.detach().cpu().numpy(), rpy=rpy).to(device)
            y = reconstruct_positional_data(roty, batch.edge_index.detach().cpu().numpy(), batch.x.detach().cpu().numpy(), rpy=rpy).to(device)
            x = reconstruct_positional_data(rotx, batch.edge_index.detach().cpu().numpy(), batch.x.detach().cpu().numpy(), rpy=rpy).to(device)

        for i,node_dist in enumerate(torch.norm(out[:,:3]-y[:,:3], dim=1).detach().tolist()):
            root_distance = root_dist_dict[i]
            if root_distance in running_l2_norm_by_root_dist.keys():
                running_l2_norm_by_root_dist[root_distance].append(node_dist)
                root_dist_accumulator[root_distance] += 1
            else:
                running_l2_norm_by_root_dist[root_distance] = [node_dist]
                root_dist_accumulator[root_distance] = 1

        for i,base_dist in enumerate(torch.norm(x[:,:3]-y[:,:3], dim=1).detach().tolist()):
            root_distance = root_dist_dict[i]
            if root_distance in running_l2_base_by_root_dist.keys():
                running_l2_base_by_root_dist[root_distance].append(base_dist)
            else:
                running_l2_base_by_root_dist[root_distance] = [base_dist]

        norm = torch.sum(torch.norm(out[:,:3]-y[:,:3], dim=1))
        base = torch.sum(torch.norm(x[:,:3]-y[:,:3], dim=1))

        l2_norms.append(norm.detach().item())
        l2_norm_base.append(base.detach().item())

        l2_norm_averaged.append(norm.detach().item()/out.size()[0])
        l2_norm_base_averaged.append(base.detach().item()/out.size()[0])

        running_l2_norm += norm
        running_l2_norm_base += base
        num_graphs+=out.size()[0]
        if idx < 10:
            visualize_graph(out[:,:3], 
                            y[:,:3], 
                            x[:,:3], 
                            batch.edge_index, batch.force_node[0], 
                            force, results_path+"prediction_example%s"%idx)
        idx += 1  

    l2_norm_by_node = []
    l2_norm_base_by_node = []
    root_dists = list(root_dist_accumulator.keys())
    root_dists.sort()
    for root_dist in root_dists:
        l2_norm_by_node.append(sum(running_l2_norm_by_root_dist[root_dist])/root_dist_accumulator[root_dist])
        l2_norm_base_by_node.append(sum(running_l2_base_by_root_dist[root_dist])/root_dist_accumulator[root_dist])

    l2_std_dev_by_node = []
    l2_std_dev_base_by_node = []
    for root_dist in root_dists:
        l2_std_dev_by_node.append(calc_variance(l2_norm_by_node[root_dist], running_l2_norm_by_root_dist[root_dist])**(1/2))
        l2_std_dev_base_by_node.append(calc_variance(l2_norm_base_by_node[root_dist], running_l2_base_by_root_dist[root_dist])**(1/2))


    X_axis = np.arange(len(root_dists))

    plt.bar(X_axis - 0.2, l2_norm_by_node, 0.4, label = 'prediction error', yerr=l2_std_dev_by_node)
    plt.bar(X_axis + 0.2, l2_norm_base_by_node, 0.4, label = 'baseline error', yerr=l2_std_dev_base_by_node)

    plt.xticks(X_axis, root_dists)
    plt.xlabel("Distance from root node")
    plt.ylabel("L2 distance")
    plt.title("error by distance to root node")
    plt.legend()

    plt.savefig(results_path+"error_by_node")
    plt.close()

    l2_norm = running_l2_norm/num_graphs
    l2_base = running_l2_norm_base/num_graphs

    l2_std_dev = calc_variance(l2_norm, l2_norms, N=num_graphs)**2
    l2_base_std_dev = calc_variance(l2_base, l2_norm_base, N=num_graphs)**2

    print_loss_by_tree(l2_norm_base_averaged, l2_norm_averaged)

    print('Mean node distance error: {}'.format(l2_norm))
    print('Mean base node displacement: {}'.format(l2_base))
    print('Variance of node distance error: {}'.format(l2_std_dev))
    print('Variance of base node displacement: {}'.format(l2_base_std_dev))

    return l2_norm, l2_base

def calc_variance(mean, scores, N=None):
    running_var = 0
    for score in scores:
        running_var += (mean - score)**2
    
    if N is not None:
        variance = running_var/N
    else:
        variance = running_var/len(scores)
    return variance

def generate_root_distance_dict(tree_edges): # assumes 0 is always the root node
    dist_dict = {0:0}
    frontier = [0]
    explored = []
    while len(frontier) != 0:
        explore = frontier.pop()
        explored.append(explore)
        children = find_children(tree_edges, explore)
        for child in children:
            if child not in explored:
                frontier.append(child)
                dist_dict[child] = dist_dict[explore] + 1
    return dist_dict


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
    plt.close()

def make_gif(X, Y, X_0, edge_index, force_node, force, id):
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


    fig = plt.figure()
    ax = Axes3D(fig)
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
    
    #custom_lines = [Line2D([0], [0], color=[0,0,1,1], lw=2),
    #                Line2D([0], [0], color=[1,0,0,1], lw=4),
    #                Line2D([0], [0], color=[0,1,0,1], lw=4)]

    #ax.legend(custom_lines, ['Input', 'GT', 'Predicted'])
    
    
    ax = set_axes_equal(ax)

    def init():
        return fig,

    def animate(i):
        ax.view_init(elev=10., azim=i)
        return fig,

    # Animate
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=360, interval=20, blit=True)
    anim.save('output/{}.gif'.format(id))

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

def make_directed_and_prune_augment(X_edges, X_force, X_pos, Y_pos, make_directed=True, prune_augmented=True, ori=False):
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
                if src == 0 and not ori:
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

def get_indirect_descendants(X_edges, node): #assumes non loopy graphs
    frontier = [node]
    descendants = []
    while len(frontier) > 0:
        explore_node = frontier.pop()
        for parent, child in X_edges:
            if parent == explore_node:
                if parent != node:
                    descendants.append(child)
                frontier.append(child)
    return descendants

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

def load_npy(data_dir, tree_num, orientational, prefix):
    # Load npy files from dataset_dir. A shortcut to 'sample_1_push' shared folder has been added to 'My Drive' 
    if orientational:
        X_stiffness_damping = np.load(os.path.join(data_dir, prefix + 'X_coeff_stiff_damp_tree%s.npy'%tree_num))
        X_edges = np.load(os.path.join(data_dir, prefix + 'X_edge_def_tree%s_ori.npy'%tree_num))
        X_force = np.load(os.path.join(data_dir, prefix + 'X_force_applied_tree%s_ori.npy'%tree_num))
        X_pos = np.load(os.path.join(data_dir, prefix + 'X_vertex_init_tree%s_ori.npy'%tree_num))
        Y_pos = np.load(os.path.join(data_dir, prefix + 'Y_vertex_final_tree%s_ori.npy'%tree_num))
    else:
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


def has_same_parent(i,j,edges):
    for parent, child in edges:
        if child == i:
            parent_i = parent
        if child == j:
            parent_j = parent
    return parent_i == parent_j

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

def find_children(edges, node):
    children = []
    for parent, child in edges:
        if parent == node:
            children.append(child)
    return children

def assign_thickness(edges, thickness_list, node):
    child_idxs = find_children(edges, node)
    if len(child_idxs) == 0:
        thickness_list[node] = TIP_THICKNESS
    else:
        radius = 0
        for child_idx in child_idxs:
            if thickness_list[child_idx] == 0:
                thickness_list = assign_thickness(edges, thickness_list, child_idx)
            radius += thickness_list[child_idx]**3
        radius = radius**(1/3)
        thickness_list[node] = radius
    return thickness_list

def add_thickness(X_edges, X_pos):
    thickness_list = [0]*np.shape(X_pos)[1]
    thickness_list = assign_thickness(X_edges, thickness_list, 0)
    thickness_arr = np.tile(np.array(thickness_list), (np.shape(X_pos)[0],1))
    thickness_arr = np.reshape(thickness_arr, (np.shape(X_pos)[0], np.shape(X_pos)[1], 1))
    X_pos = np.append(X_pos, thickness_arr, axis=2)
    return X_pos

def calculate_orientational_data(X_edges, X_pos, Y_pos, X_force): #something in this function is off -> gives incorrect rot data.
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

def rot_from_rpy(arr): #for use with batches with 2 dim
    shape = np.shape(arr)
    if shape[1] == 3: # only rpy present
        quat_arr = np.zeros((shape[0], 4))
        for datapoint_idx in range(0, shape[0]):
            rpy = arr[datapoint_idx]
            rot = Rotation.from_euler("xyz", rpy)
            quat_arr[datapoint_idx] = rot.as_quat()
        arr = quat_arr
    elif shape[1] >= 6: # rpy and position present
        quat_arr = np.zeros((shape[0], 4))

        for datapoint_idx in range(0, shape[0]):
            rpy = arr[datapoint_idx][3:7]
            rot = Rotation.from_euler("xyz", rpy)
            quat_arr[datapoint_idx] = rot.as_quat()

        arr = np.append(arr[:,:3], quat_arr)
    return arr

def rot_to_rpy(X_pos, Y_pos):
    shape = np.shape(X_pos)
    rot_x = np.zeros((shape[0], shape[1], 3))
    rot_y = np.zeros((shape[0], shape[1], 3))

    for tree_idx in range(0,shape[0]):
        for node_idx in range(0,shape[1]):
            quat_x = X_pos[tree_idx,node_idx,3:7]
            quat_y = Y_pos[tree_idx,node_idx,3:7]
            rotx = Rotation.from_quat(quat_x)
            roty = Rotation.from_quat(quat_y)
            rot_x[tree_idx][node_idx] = rotx.as_euler("xyz")
            rot_y[tree_idx][node_idx] = roty.as_euler("xyz")

    lengths = np.reshape(X_pos[:,:,7],(shape[0],shape[1],1))
    X_pos = np.append(X_pos[:,:,:3], rot_x, axis=2)
    X_pos = np.append(X_pos[:,:,:], lengths, axis=2)
    Y_pos = np.append(Y_pos[:,:,:3], rot_y, axis=2)
    return X_pos, Y_pos

def make_dataset(X_edges, X_force, X_pos, Y_pos, profile,
                 make_directed=True, prune_augmented=False, rotate_augmented=False): # CALLED PER TREE
    num_graphs = len(X_pos)
    
    if rotate_augmented:
        X_edges, X_force, X_pos, Y_pos = rotate_augment(X_edges, X_force, X_pos, Y_pos)
    ori = profile == 3 or profile == 4 or profile == 5
    if profile == 1 or profile == 2:
        X_edges, X_pos, Y_pos, X_force = remove_duplicate_nodes(X_edges, X_pos, Y_pos, X_force)
        if profile == 2:
            X_pos = add_thickness(X_edges, X_pos) #adds branch thickness as a fourth value to the 3 dimensional position argument for each node
    if profile == 4:
        X_pos, Y_pos = rot_to_rpy(X_pos, Y_pos)

    X_edges, X_force, X_pos, Y_pos = make_directed_and_prune_augment(X_edges, X_force, X_pos, Y_pos,
                                                                     make_directed=make_directed, 
                                                                     prune_augmented=prune_augmented,
                                                                     ori=ori)

    num_graphs = len(X_pos)
    dataset = []
    for i in range(num_graphs): 
        # Combine all node features: [position, force, stiffness] with shape (num_nodes, xyz(3)+force(3)+stiffness_damping(4)) 
        # stiffness damping is (4) because of bending stiffness/damping and torsional stiffness/damping
        root_feature = np.zeros((len(X_pos[i]), 1))
        #root_feature[0, 0] = 1.0
        #X_data = np.concatenate((X_pos[i], X_force[i], root_feature), axis=1)
        X_data = np.concatenate((X_pos[i], X_force[i]), axis=1)

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

def get_dataset_metrics(dataset, suffix):
    avg_displacements_per_node = []
    for tree_graph_push in dataset:
        avg_displacement_per_node = np.sum(np.linalg.norm(tree_graph_push.x.numpy()[:,:3] - tree_graph_push.y.numpy()[:,:3], axis=1))/len(tree_graph_push.x)
        avg_displacements_per_node.append(avg_displacement_per_node)
    column_diag_dict = {}
    for displacement in avg_displacements_per_node:
        if round(displacement, 2) in column_diag_dict.keys():
            column_diag_dict[round(displacement, 2)] += 1
        else:
            column_diag_dict[round(displacement, 2)] = 1

    accum_list = []
    dict_keys = list(column_diag_dict.keys())
    dict_keys.sort()
    for key in dict_keys:
        accum_list.append(column_diag_dict[key])

    X_axis = np.arange(len(dict_keys))

    plt.figure(figsize=(15,4))
    plt.xticks(rotation=90)
    plt.bar(X_axis, accum_list)
    plt.xticks(X_axis, dict_keys)
    plt.title("dataset analysis for "+suffix)
    plt.savefig(results_path+"dataset_analysis_"+suffix, format="png")
    #plt.show()
    plt.close()


##### FUNCTION DEF END #####

parser = argparse.ArgumentParser()
parser.add_argument("-dir", type=str, dest="file_directory", help="directory in which the dataset files are located", default="/mnt/hdd/jan-malte/8Nodes_new_by_tree/")
parser.add_argument("-id", type=str, dest="id", help="id to identify the results folder")
parser.add_argument("-gn", type=int, default=10, dest="graph_nodes", help="number of layers the GCN will have") #number of graph nodes
parser.add_argument("-ep", type=int, default=500, dest="n_epochs", help="number of epochs to run") # number of epochs
parser.add_argument("-pat", type=int, default=50, dest="sched_patience", help="patience of the scheduler") #patience of scheduler
parser.add_argument("-btchs", type=int, default=256, dest="batch_size", help="batch size")
parser.add_argument("-ilr", type=float, default=2e-3, dest="learn_rate", help="initial learning rate")
parser.add_argument("-cuda", type=int, dest="cuda", help="cuda gpu to run on")
parser.add_argument("-weights", type=str, dest="weights", help="saved model weights to load")
parser.add_argument("-topbrk", type=int, default=1, choices=[0,1], dest="topbrk", help="topology break: if on, the validation tree topologies will not be present in the train set")
parser.add_argument("-hidden_size", type=int, default=-1, dest="hidden_size", help="sets the hidden size of the network")
parser.add_argument("-sep_test", type=int, default=1, dest="sep_test", help="set to 0 when we test on the same dataset that provided weights were trained on")

parser.add_argument("-profile", type=int, default=0, choices=[0,1,2,3,4,5,6], dest="profile")

args = parser.parse_args()

profile = args.profile
topology_break = args.topbrk == 1
#profile choices:
# 0: no graph alteration
# 1: transform into node representation (graph nodes are tree points)
# 2: transform into node representation with thickness parameter
# 3: load as orientational representation with quaternion angles
# 4: load as orientational representation with rpy angles
# 5: load as orientational representation with quaternion angles and custom loss function
# 6: load as positional like in 0 but use FGCNRec as model

if profile == 1:
    in_size = 6     # 3 positional 3 force
    out_size = 3    # 3 positional
elif profile == 2:
    in_size = 7     # 3 positional 1 thickness 3 force
    out_size = 3    # 3 positional
elif profile == 3 or profile == 5:
    in_size = 11    # 3 positional 4 orientational 1 lenght 3 force
    out_size = 4    # 4 orientational
elif profile == 4:
    in_size = 10    # 3 positional 3 orientational 1 lenght 3 force
    out_size = 3    # 3 orientational
elif profile == 0 or profile == 6:
    in_size = 10    # 3 positional 4 orientational 3 force
    out_size = 3    # 3 positional



print("[%s] setting up result path"%datetime.datetime.now())
results_path = "../../results%s"%args.id
os.mkdir(results_path)
results_path = results_path+"/"
print("[%s] done"%datetime.datetime.now())

print("[%s] setting up wandb"%datetime.datetime.now())
wandb.init(project="gcn-tree-deformation")
wandb.config = {
  "GCN Layers": args.graph_nodes,
  "Epochs": args.n_epochs,
  "Profile": profile,
  "Scheduler Patience": args.sched_patience,
  "Batch Size": args.batch_size,
  "Initial learning rate": args.learn_rate
}
print("[%s] done"%datetime.datetime.now())

# Loading Datase
print("[%s] loading data"%datetime.datetime.now())
d = args.file_directory

dataset = []
val_dataset = []
for tree_pt in TREE_PTS:
    prefix = "[%s]"%tree_pt
    try:
        checkload = np.load(d + prefix + 'X_edge_def_tree%s.npy'%0) #assumes full dataset present (should be true anyways)
    except:
        prefix = ""

    i = 0
    val_idx = 0
    tree = 0
    if profile == 3 or profile == 4 or profile == 5:
        suffix = "_ori.npy"
        l_path = os.path.join(d, prefix + 'X_edge_def_tree%s'%0 + suffix)
    else:
        suffix = ".npy"
        l_path = os.path.join(d, prefix + 'X_edge_def_tree%s'%0 + suffix)
    while os.path.exists((d + prefix + 'X_edge_def_tree%s'%tree + suffix)):
        X_force_list = []
        X_pos_list = []
        Y_pos_list = []
        ori = profile == 3 or profile == 4 or profile == 5
        X_edges, X_force, X_pos, Y_pos = load_npy(d, tree, ori, prefix)
        print(len(X_force))
        X_force_list.append(X_force)
        X_pos_list.append(X_pos)
        Y_pos_list.append(Y_pos)
        X_force_arr = np.concatenate(X_force_list)
        X_pos_arr = np.concatenate(X_pos_list)
        Y_pos_arr = np.concatenate(Y_pos_list)
        dataset_tree = make_dataset(X_edges, X_force_arr, X_pos_arr, Y_pos_arr, profile=profile, make_directed=True, prune_augmented=False, rotate_augmented=False)
        i += 1
        tree += 1
        if i <= 3 and topology_break:
            val_dataset = val_dataset + dataset_tree
        else:
            dataset = dataset + dataset_tree

print("[%s] done"%datetime.datetime.now())
print("[%s] generating dataset metrics"%datetime.datetime.now())
get_dataset_metrics(dataset, "train")
if topology_break:
    get_dataset_metrics(val_dataset, "val")
# Shuffle Dataset
#X_force_arr, X_pos_arr, Y_pos_arr = shuffle_in_unison(X_force_arr, X_pos_arr, Y_pos_arr)
print("[%s] done"%datetime.datetime.now())

print("[%s] printing example trees"%datetime.datetime.now())
for _ in range(10):
    i = random.randint(0, len(dataset))
    print(len(dataset))
    print(i)
    print("shown tree index: %s"%i)
    if profile==3 or profile==4 or profile==5:
        ori_end = 3+out_size
        rpy = profile==4
        X = reconstruct_positional_data(dataset[i].x[:,3:ori_end], dataset[i].edge_index, dataset[i].x, rpy=rpy)
        Y = reconstruct_positional_data(dataset[i].y[:,3:ori_end], dataset[i].edge_index, dataset[i].x, rpy=rpy)
    else:
        print(np.around(dataset[i].x[:,3:7], 2))
        print(np.around(dataset[i].y[:,3:7], 2))
        X = dataset[i].x[:,:3]
        Y = dataset[i].y[:,:3]
    force_node = dataset[i].force_node
    print_edges = dataset[i].edge_index
    force = dataset[i].x[:,-3:]
    #visualize_graph(X, val_dataset[i].x[:,:3], X, print_edges, force_node, force, results_path+"comp_with_self_x%s"%i)
    #visualize_graph(Y, val_dataset[i].y[:,:3], Y, print_edges, force_node, force, results_path+"comp_with_self_y%s"%i)
    visualize_graph(X, Y, X, print_edges, force_node, force, results_path+"example_push%s"%i)

fig = plt.figure()
ax = fig.add_subplot(111)
print("[%s] done"%datetime.datetime.now())

print("[%s] preparing dataset"%datetime.datetime.now())
print(topology_break)
if topology_break:
    train_dataset = dataset
    random.shuffle(train_dataset)

    random.shuffle(val_dataset)
    print(len(val_dataset))

    test_dataset = val_dataset
else:
    print("works")
    random.shuffle(dataset)

    # setup validation/test split
    train_val_split = int(len(dataset)*0.9)

    train_dataset = dataset[:train_val_split]
    val_dataset = dataset[train_val_split:]
    
    random.shuffle(val_dataset)
    test_val_split = int(len(val_dataset))
    test_dataset = val_dataset

if args.weights is not None and args.sep_test == 1:
    print("loading full dataset")
    test_dataset = dataset

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

batch_size=args.batch_size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

print("[%s] done"%datetime.datetime.now()) 

# Setup GCN
print("[%s] setting up GCN"%datetime.datetime.now())
if args.cuda is not None:
    device = torch.device('cuda:%s'%args.cuda if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("running on CPU")
if profile == 6:
    model = FGCNRec(1, in_size, out_size).to(device) #TODO: make more adjustable once proven useful
else:
    if args.hidden_size != -1:
        model = FGCN(args.graph_nodes, in_size, out_size, args.hidden_size).to(device)
    else:
        model = FGCN(args.graph_nodes, in_size, out_size).to(device)
if args.weights is None:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)
    if profile == 5: 
        criterion = QuatLoss()
    elif profile == 4 or profile == 3:
        criterion = torch.nn.L1Loss()
    else:
        criterion = torch.nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.sched_patience, factor=0.5, min_lr=5e-4)
    print("[%s] done"%datetime.datetime.now())

    print("[%s] Training"%datetime.datetime.now())
    # Train and validate model
    train_loss_history = []
    val_loss_history = []
    base_loss_history = []
    best_loss = 1e9
    for epoch in range(1, args.n_epochs+1):
        train_loss = train(model, optimizer, criterion, train_loader, epoch, device, out_size=out_size, profile=profile) #train model
        val_loss, baseline_loss = validate(model, criterion, val_loader, epoch, device, out_size=out_size, profile=profile) # validate model

        if val_loss < best_loss:
            best_loss=val_loss
            best_model = copy.deepcopy(model)

        scheduler.step(best_loss)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        base_loss_history.append(baseline_loss)
        if epoch%10==0:
            print(scheduler._last_lr)
        if epoch%20==0:
            torch.save(best_model.state_dict(), results_path+'model_epoch%s.pt'%epoch)
        
        # log with wandb
        wandb.log({"training loss": train_loss, "validation loss": val_loss, "baseline_loss": baseline_loss, "epoch": epoch, "learning rate": scheduler._last_lr[0]})

        ax.clear()
        ax.plot(train_loss_history, 'r', label='train')
        ax.plot(val_loss_history, 'b', label='validation')
        ax.plot(base_loss_history, 'g', label='baseline loss')
        ax.legend(loc="upper right")
        ax.set_ylim([0, 0.2])
        #display(fig)
        plt.close()
        clear_output(wait=True)
        print("[%s] epoch %s"%(datetime.datetime.now(), epoch))

    print("[%s] done"%datetime.datetime.now())

    torch.save(best_model.state_dict(), results_path+'model.pt')

    # evaluate best model and save the state dict
    best_model.eval()
    test(best_model, test_loader, device, out_size=out_size, profile=profile)
else:
    print("[%s] Loading weights"%datetime.datetime.now())
    model.load_state_dict(torch.load(args.weights, map_location=device))
    print("[%s] weights loaded"%datetime.datetime.now())
    print("[%s] testing"%datetime.datetime.now())
    test(model, test_loader, device, out_size=out_size, profile=profile)
    print("[%s] done"%datetime.datetime.now())