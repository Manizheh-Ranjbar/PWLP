import numpy as np
import random
from tqdm import tqdm
import os, sys, pdb, math, time
import pickle as cp
#import _pickle as cp  # python3 compatability
import networkx as nx
import argparse
import scipy.io as sio
import scipy.sparse as ssp
from sklearn import metrics
from gensim.models import Word2Vec
import warnings
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append('%s/../../pytorch_DGCNN' % cur_dir)
import multiprocessing as mp
import torch

from torch_geometric.transforms import LineGraph
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.utils import from_networkx

import os
from collections import defaultdict

# from multiprocessing import Lock

# lock = Lock() 




class GNNGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy array of continuous node features
        '''
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = label
        self.node_features = node_features  # numpy array (node_num * feature_dim)
        self.degs = list(dict(g.degree).values())
         # Compute average feature per node (mean of features along axis 1)
        if node_features is None: 
            self.average_feature= None
        else: 
            # self.average_feature = torch.mean(self.node_features, axis=0)  # Using torch.mean
            # self.average_feature = torch.tensor(self.node_features[0]*self.node_features[1])

            neighbors_0 = list(g.neighbors(0))
            neighbors_1 = list(g.neighbors(1))
            average_feature= torch.mean(self.node_features, axis=0)  # Using torch.mean
            average_feature0 = torch.mean(self.node_features[neighbors_0], axis=0)  # Using torch.mean
            average_feature1 = torch.mean(self.node_features[neighbors_1], axis=0)  # Using torch.mean
            # self.average_feature = torch.tensor(self.node_features[0]*self.node_features[1]*average_feature0*average_feature1)  # Using torch.mean
            self.average_feature = torch.cat(((self.node_features[0] * self.node_features[1]), (average_feature0*average_feature1)),dim=0)
            # self.average_feature = torch.cat(((self.node_features[0] * self.node_features[1]), (average_feature)),dim=0)

        # Repeat the average feature across the feature dimension (if needed)
        # self.average_feature= self.average_feature.unsqueeze(1).repeat(1, self.node_features.shape[1])

        if len(g.edges()) != 0:
            x, y = list(zip(*g.edges()))
            self.num_edges = len(x)        
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = self.edge_pairs.flatten()
        else:
            self.num_edges = 0
            self.edge_pairs = np.array([])
        
        # see if there are edge features
        self.edge_features = None
        if nx.get_edge_attributes(g, 'features'):  
            # make sure edges have an attribute 'features' (1 * feature_dim numpy array)
            edge_features = nx.get_edge_attributes(g, 'features')
            assert(type(list(edge_features.values())[0]) == np.ndarray) 
            # need to rearrange edge_features using the e2n edge order
            edge_features = {(min(x, y), max(x, y)): z for (x, y), z in list(edge_features.items())}
            keys = sorted(edge_features)
            self.edge_features = []
            for edge in keys:
                self.edge_features.append(edge_features[edge])
                self.edge_features.append(edge_features[edge])  # add reversed edges
            self.edge_features = np.concatenate(self.edge_features, 0)

def sample_neg(net, test_ratio=0.1, train_pos=None, test_pos=None, max_train_num=None):
    # get upper triangular matrix
    net_triu = ssp.triu(net, k=1)
    # sample positive links for train/test
    row, col, _ = ssp.find(net_triu)
    # sample positive links if not specified
    if train_pos is None or test_pos is None:
        perm = random.sample(list(range(len(row))), len(row))
        row, col = row[perm], col[perm]
        split = int(math.ceil(len(row) * (1 - test_ratio)))
        train_pos = (row[:split], col[:split])
        test_pos = (row[split:], col[split:])
    # if max_train_num is set, randomly sample train links
    if max_train_num is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm])
    # sample negative links for train/test
    train_num, test_num = len(train_pos[0]), len(test_pos[0])
    neg = ([], [])
    n = net.shape[0]
    print('sampling negative links for train and test')
    while len(neg[0]) < train_num + test_num:
        i, j = random.randint(0, n-1), random.randint(0, n-1)
        if i < j and net[i, j] == 0:
            neg[0].append(i)
            neg[1].append(j)
        else:
            continue
    train_neg  = (neg[0][:train_num], neg[1][:train_num])
    test_neg = (neg[0][train_num:], neg[1][train_num:])
    return train_pos, train_neg, test_pos, test_neg

    
def links2subgraphs(folder_result, A, train_pos, train_neg, val_pos, val_neg, test_pos, test_neg, h=1, max_nodes_per_hop=None, node_information=None):
    # automatically select h from {1, 2}
    if h == 'auto':
        # split train into val_train and val_test
        _, _, val_test_pos, val_test_neg = sample_neg(A, 0.1)
        val_A = A.copy()
        val_A[val_test_pos[0], val_test_pos[1]] = 0
        val_A[val_test_pos[1], val_test_pos[0]] = 0
        val_auc_CN = CN(val_A, val_test_pos, val_test_neg)
        val_auc_AA = AA(val_A, val_test_pos, val_test_neg)
        print('\033[91mValidation AUC of AA is {}, CN is {}\033[0m'.format(val_auc_AA, val_auc_CN))
        if val_auc_AA >= val_auc_CN:
            h = 2
            print('\033[91mChoose h=2\033[0m')
        else:
            h = 1
            print('\033[91mChoose h=1\033[0m')

    # extract enclosing subgraphs
    max_n_label = {'value': 0}
    def helper(A, links, g_label):
        '''
        g_list = []
        for i, j in tqdm(zip(links[0], links[1])):
            g, n_labels, n_features = subgraph_extraction_labeling((i, j), A, h, max_nodes_per_hop, node_information)
            max_n_label['value'] = max(max(n_labels), max_n_label['value'])
            g_list.append(GNNGraph(g, g_label, n_labels, n_features))
        return g_list
        '''
        # the new parallel extraction code
        start = time.time()
        pool = mp.Pool(mp.cpu_count())
        results = pool.map_async(parallel_worker, [(folder_result,(i, j), A, h, max_nodes_per_hop, node_information) for i, j in zip(links[0], links[1])])
        remaining = results._number_left
        pbar = tqdm(total=remaining)
        while True:
            pbar.update(remaining - results._number_left)
            if results.ready(): break
            remaining = results._number_left
            time.sleep(1)
        results = results.get()
        pool.close()
        pbar.close()
        g_list = [GNNGraph(g, g_label, n_labels, n_features) for g, n_labels, n_features in results]
        max_n_label['value'] = max(max([max(n_labels) for _, n_labels, _ in results]), max_n_label['value'])
        end = time.time()
        print("Time eplased for subgraph extraction: {}s".format(end-start))
        return g_list
        

    print('Enclosing subgraph extraction begins...')
    # print(train_pos)
    train_graphs = helper(A, train_pos, 1) + helper(A, train_neg, 0)
    test_graphs = helper(A, test_pos, 1) + helper(A, test_neg, 0)
    val_graphs =helper(A, val_pos, 1) + helper(A, val_neg, 0)
    print(max_n_label)
    return train_graphs, val_graphs, test_graphs, max_n_label['value']

def parallel_worker(x):
    ##    return subgraph_extraction_labeling_RWLG_Top_betweeness(*x)
    return subgraph_extraction_labeling_RWLG_topNodes(*x)
    ### return subgraph_extraction_labeling_RWLG(*x)
    # return subgraph_extraction_labeling(*x)
    
def subgraph_extraction_labeling(folder_result,ind, A, h=1, max_nodes_per_hop=None, node_information=None):
    # extract the h-hop enclosing subgraph around link 'ind'
    dist = 0
    nodes = set([ind[0], ind[1]])
    visited = set([ind[0], ind[1]])
    fringe = set([ind[0], ind[1]])
    nodes_dist = [0, 0]
    for dist in range(1, h+1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(list(fringe), max_nodes_per_hop)

                # fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)
        nodes_dist += [dist] * len(fringe)
    # if len(nodes) < 2:
    #     print(nodes)
    #     print(ind)
    #     print("ERROR: Subgraph has less than 2 nodes!")
    #     exit()
    # print("FFFFFFFFFFFFFFFFFFFFFFF")
    # print(nodes)
    # print(ind)

    # move target nodes to top
    nodes.remove(ind[0])
    nodes.remove(ind[1])
    nodes = [ind[0], ind[1]] + list(nodes) 
    # print("oooooooooooooooooo")
    # print(nodes)
    # print(ind)
    # print((len(nodes)))
    # print(str(len(nodes)))
    # print("LLLLLLLLLLLLLLLLLLLLLLL")
    # if len(nodes) < 2:
    #     print(nodes)
    #     print(ind)
    #     print("ERROR: Subgraph has less than 2 nodes!")
    #     print(str(len(nodes)))
    #     exit()

    subgraph = A[nodes, :][:, nodes]
    # print(folder_result)
    # nodestr= str(len(nodes)) + '\n'
    # edgestr= str(np.sum(subgraph) / 2) + '\n'
    # with open('../results/RWLGLP/unmasked02_seed1/NumNodesADV_Subgraph.txt', 'a+') as f:

    # with lock:

    # with open(folder_result+'/NumNodes_Subgraph_simpleLGLP.txt', 'a+') as f:
    #     f.write(nodestr)
    # # with open('../results/RWLGLP/unmasked02_seed1_topNodes'+str(Num_top_Nodes)+'nW'+str(num_walks)+'wl'+str(walk_length)+'_withoutedge_win/NumedgesSMG_Subgraph.txt', 'a+') as f:
    # with open(folder_result+'/Numedges_Subgraph_simpleLGLP.txt', 'a+') as f:
    #     f.write(edgestr)


    # apply node-labeling
    labels = node_label(subgraph)
    # get node features
    features = None
    if node_information is not None:
        features = node_information[nodes]
    # construct nx graph
    g = nx.from_scipy_sparse_matrix(subgraph)
    # remove link between target nodes
    if not g.has_edge(0, 1):
        g.add_edge(0, 1)
    return g, labels.tolist(), features
################################## For Simple #################################### 
def subgraph_extraction_labeling_RWLG(folder_result,ind, A, h=1, max_nodes_per_hop=None, node_information=None):
    # extract the randomwalk subgraph around link 'ind'
    # new_G = nx.from_numpy_matrix(A)
    all_rwnodes = []
    for i in range(10):
        # print("LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL1", ind)
        
        randomwalk_nodes_path_0 = RandomWalkMatrix(A, starting_node=ind[0],seed = 42,num_steps=20)
        #randomwalk_nodes_path_0 = RandomWalkGraph(new_G, starting_node=ind[0],seed = 42,num_steps=20)
        # print("LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL2",ind)
        randomwalk_nodes_path_1 = RandomWalkMatrix(A, starting_node=ind[1],seed = 42,num_steps=20)
        #randomwalk_nodes_path_1 = RandomWalkGraph(new_G, starting_node=ind[1],seed = 42,num_steps=20)
        # print("LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL3",ind)
        all_rwnodes.extend(randomwalk_nodes_path_0)  # Extend the list with the new random walk
        all_rwnodes.extend(randomwalk_nodes_path_1)  # Extend the list with the new random walk
        # print("LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL4",ind)
        # print(all_rwnodes)

    #     randomwalk_nodes = set(randomwalk_nodes_path_0).union(set(randomwalk_nodes_path_1))
    randomwalk_nodes= set(all_rwnodes)
    # print(randomwalk_nodes)
    randomwalk_nodes.remove(ind[0])
    randomwalk_nodes.remove(ind[1])
    randomwalk_nodes = [ind[0], ind[1]] + list(randomwalk_nodes) 

    # print("Lenght Nodes: ", len(randomwalk_nodes))
    # print("********************")

    #subgraph = A[randomwalk_nodes, :][:, randomwalk_nodes]
    subgraph = A[np.ix_(randomwalk_nodes, randomwalk_nodes)]

    '''
    nodestr= str(len(randomwalk_nodes)) + '\n'
    edgestr= str(np.sum(subgraph) / 2) + '\n'
    # with open('../results/RWLGLP/unmasked02_seed1/NumNodesADV_Subgraph.txt', 'a+') as f:
    with open(folder_result+'/NumNodes_Subgraph.txt', 'a+') as f:
        f.write(nodestr)
    # with open('../results/RWLGLP/unmasked02_seed1/NumedgesADV_Subgraph.txt', 'a+') as f:
    with open(folder_result+'/Numedges_Subgraph.txt', 'a+') as f:
        f.write(edgestr)
    '''

    # apply node-labeling
    labels = node_label(subgraph)
    features = None
    if node_information is not None:
        features = node_information[randomwalk_nodes]
    # construct nx graph
    g = nx.from_scipy_sparse_matrix(subgraph)
    # remove link between target nodes
    if not g.has_edge(0, 1):
        g.add_edge(0, 1)
    return g, labels.tolist(), features 
def RandomWalkMatrix(adj_matrix, starting_node=1, num_steps=5, seed=42):
    # random.seed(seed)
    current_node = starting_node
    visited_nodes = [current_node]
    
    for _ in range(num_steps):
        neighbors = adj_matrix[current_node].nonzero()[1]  # Get non-zero entries in row
        if len(neighbors) == 0:
            break  # Stop if there are no neighbors
        current_node = random.choice(neighbors)
        visited_nodes.append(current_node)
    
    return visited_nodes
#_simple
def RandomWalkGraph( G, starting_node=1,seed = 42,num_steps=5):
       # self.starting_node = random.choice(list(self.G.nodes))
        current_node = starting_node
        visited_nodes = [current_node]
        visited_edges = []
        for _ in range(num_steps):
            previous_node = current_node
            neighbors = list(G.neighbors(previous_node))
            if not neighbors:
                current_node= previous_node
            else :
                current_node=random.choice(neighbors)

    #             current_node = random.choice(list(G.neighbors(previous_node)))
            visited_edges.append(sorted((previous_node, current_node)))
            visited_nodes.append(current_node)
        return visited_nodes
############################### For Selecting most important nodes By Random Walk########################################
def subgraph_extraction_labeling_RWLG_topNodes(folder_result,ind, A, h=1, max_nodes_per_hop=None, node_information=None):
    # extract the randomwalk subgraph around link 'ind'
    # new_G = nx.from_numpy_matrix(A)
    # new_G.add_edge(ind[0],ind[1])
    # new_G.add_edge(ind[1],ind[0])

    # Step 3: Set parameters for random walks
    target_nodes = [ind[0],ind[1] ]  # Your two target nodes
    ############## this was for first results
    num_walks = 50  # Number of random walks per target node
    walk_length = 5  # Length of each random walk
    Num_top_Nodes = 100
    ################## 10-20-50
    # num_walks = 10  # Number of random walks per target node
    # walk_length = 20  # Length of each random walk
    # Num_top_Nodes =50
     # Step 4: Execute random walks and get visit frequencies
    visit_counts = most_important_nodes_by_random_walks(A, target_nodes, num_walks, walk_length)
    # visit_counts = most_important_nodes_by_random_walks(new_G, target_nodes, num_walks, walk_length)

    # Step 5: Sort nodes by their visit frequencies (most important first)
    sorted_nodes = sorted(visit_counts.items(), key=lambda x: x[1], reverse=True)

    # # Display results
    # print("Node importance based on random walk visit frequencies:")
    # for node, count in sorted_nodes:
    #     print(f"Node {node}: Visited {count} times")

    # Visualizing most important nodes
    most_important_nodes = [node for node, count in sorted_nodes[:Num_top_Nodes]]  # Top 3 most important nodes
    # print(f"Top 3 most important nodes: {most_important_nodes}")

    #     randomwalk_nodes = set(randomwalk_nodes_path_0).union(set(randomwalk_nodes_path_1))
    randomwalk_nodes= set(most_important_nodes)
    randomwalk_nodes.remove(ind[0])
    randomwalk_nodes.remove(ind[1])
    randomwalk_nodes = [ind[0], ind[1]] + list(randomwalk_nodes) 
    # print("Lenght Nodes: ", len(randomwalk_nodes))
    # print("********************")


    subgraph = A[randomwalk_nodes, :][:, randomwalk_nodes]
    # print(folder_result)
    # with open('../results/RWLGLP/unmasked02_seed1_topNodes'+str(Num_top_Nodes)+'nW'+str(num_walks)+'wl'+str(walk_length)+'_withoutedge_win/NumNodesSMG_Subgraph.txt', 'a+') as f:
    '''
    with open(folder_result+'/NumNodes_Subgraph.txt', 'a+') as f:
        f.write(str(len(randomwalk_nodes)) + '\n')
    # with open('../results/RWLGLP/unmasked02_seed1_topNodes'+str(Num_top_Nodes)+'nW'+str(num_walks)+'wl'+str(walk_length)+'_withoutedge_win/NumedgesSMG_Subgraph.txt', 'a+') as f:
    with open(folder_result+'/Numedges_Subgraph.txt', 'a+') as f:
        f.write(str(np.sum(subgraph) / 2) + '\n')
    '''

    # apply node-labeling
    labels = node_label(subgraph)
    features = None
    if node_information is not None:
        features = node_information[randomwalk_nodes]
    # construct nx graph
    g = nx.from_scipy_sparse_matrix(subgraph)
    # remove link between target nodes
    if not g.has_edge(0, 1):
        g.add_edge(0, 1)
    return g, labels.tolist(), features 

def most_important_nodes_by_random_walks(A, target_nodes, num_walks, walk_length):
    visit_counts = defaultdict(int)  # To keep track of visit frequency for each node
    
    for target_node in target_nodes:
        for _ in range(num_walks):
            walk = RandomWalkMatrix(A, target_node, walk_length)
            for node in walk:
                visit_counts[node] += 1
    # print(visit_counts)
    return visit_counts


def most_important_nodes_by_random_walks_graph(graph, target_nodes, num_walks, walk_length):
    visit_counts = defaultdict(int)  # To keep track of visit frequency for each node
    
    for target_node in target_nodes:
        for _ in range(num_walks):
            walk = random_walk(graph, target_node, walk_length)
            for node in walk:
                visit_counts[node] += 1

    return visit_counts

def random_walk(graph, start_node, walk_length):
    walk = [start_node]
    for _ in range(walk_length - 1):
        neighbors = list(graph.neighbors(walk[-1]))
        if not neighbors:  # If no more neighbors to visit
            break
        next_node = random.choice(neighbors)
        walk.append(next_node)
    return walk
################################ For Select from 2 hop ############################################
def subgraph_extraction_labeling_RWLG_Top_betweeness(ind, A, h=1, max_nodes_per_hop=None, node_information=None):
    # extract the randomwalk subgraph around link 'ind'
    new_G = nx.from_numpy_matrix(A)
    all_rwnodes = []
    for i in range(10):
        randomwalk_nodes_path_0 = RandomWalkGraph(new_G, starting_node=ind[0],seed = 42,num_steps=20)
        randomwalk_nodes_path_1 = RandomWalkGraph(new_G, starting_node=ind[1],seed = 42,num_steps=20)
        all_rwnodes.extend(randomwalk_nodes_path_0)  # Extend the list with the new random walk
        all_rwnodes.extend(randomwalk_nodes_path_1)  # Extend the list with the new random walk


    #     randomwalk_nodes = set(randomwalk_nodes_path_0).union(set(randomwalk_nodes_path_1))
    randomwalk_nodes= set(all_rwnodes)
    # print(randomwalk_nodes)
    randomwalk_nodes.remove(ind[0])
    randomwalk_nodes.remove(ind[1])
    randomwalk_nodes = [ind[0], ind[1]] + list(randomwalk_nodes) 

    # print("Lenght Nodes: ", len(randomwalk_nodes))
    # print("********************")

    subgraph_betweeness = A[randomwalk_nodes, :][:, randomwalk_nodes]

    top_betweeness=get_top_n_betweenness_nodes(subgraph_betweeness, 0,1, 50)
    subgraph = subgraph_betweeness[top_betweeness, :][:, top_betweeness]

    # with open('../results/RWLGLP/unmasked02_seed1_betw/NumNodesSMG_Subgraph.txt', 'a+') as f:
    #     f.write(str(len(top_betweeness)) + '\n')
    # with open('../results/RWLGLP/unmasked02_seed1_betw/NumedgesSMG_Subgraph.txt', 'a+') as f:
    #     f.write(str(np.sum(subgraph) / 2) + '\n')


    # apply node-labeling
    labels = node_label(subgraph)
    features = None
    if node_information is not None:
        features = node_information[randomwalk_nodes]
    # construct nx graph
    g = nx.from_scipy_sparse_matrix(subgraph)
    # remove link between target nodes
    if not g.has_edge(0, 1):
        g.add_edge(0, 1)
    return g, labels.tolist(), features 


def get_top_n_betweenness_nodes(graph, source, target, n):
    # Step 1: Calculate all shortest paths between source and target
    shortest_paths = list(nx.all_shortest_paths(graph, source=source, target=target))

    # Flatten the paths into a set of unique nodes that are on the shortest paths
    nodes_on_shortest_paths = set()
    for path in shortest_paths:
        nodes_on_shortest_paths.update(path)
    
    # Remove the source and target nodes since we don't want them in the ranking
    nodes_on_shortest_paths.discard(source)
    nodes_on_shortest_paths.discard(target)
    
    # Step 2: Compute betweenness centrality for all nodes
    betweenness = nx.betweenness_centrality(graph)

    # Step 3: Filter the nodes that are on the shortest paths
    filtered_betweenness = {node: betweenness[node] for node in nodes_on_shortest_paths}

    # Step 4: Sort the filtered nodes by their betweenness centrality and get top n nodes
    top_n_nodes = sorted(filtered_betweenness, key=filtered_betweenness.get, reverse=True)[:n]

    return top_n_nodes
#################################################################### 

def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res

def node_label(subgraph):
    # an implementation of the proposed double-radius node labeling (DRNL)
    K = subgraph.shape[0]
    subgraph_wo0 = subgraph[1:, 1:]
    subgraph_wo1 = subgraph[[0]+list(range(2, K)), :][:, [0]+list(range(2, K))]
    dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)
    dist_to_0 = dist_to_0[1:, 0]
    dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)
    dist_to_1 = dist_to_1[1:, 0]
    d = (dist_to_0 + dist_to_1).astype(int)
    d_over_2, d_mod_2 = np.divmod(d, 2)
    labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
    labels = np.concatenate((np.array([1, 1]), labels))
    labels[np.isinf(labels)] = 0
    labels[labels>1e6] = 0  # set inf labels to 0
    labels[labels<-1e6] = 0  # set -inf labels to 0
    return labels

def AA(A, test_pos, test_neg):
    # Adamic-Adar score
    A_ = A / np.log(A.sum(axis=1))
    A_[np.isnan(A_)] = 0
    A_[np.isinf(A_)] = 0
    sim = A.dot(A_)
    return CalcAUC(sim, test_pos, test_neg)
    
        
def CN(A, test_pos, test_neg):
    # Common Neighbor score
    sim = A.dot(A)
    return CalcAUC(sim, test_pos, test_neg)


def CalcAUC(sim, test_pos, test_neg):
    pos_scores = np.asarray(sim[test_pos[0], test_pos[1]]).squeeze()
    neg_scores = np.asarray(sim[test_neg[0], test_neg[1]]).squeeze()
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

def single_line(batch_graphs):
    pbar = tqdm(batch_graphs, unit='iteration')
    graphs = []
    for graph in pbar:
        #line_graph, labels = to_line(graph, graph.node_tags)
        line_test(graph, graph.node_tags)
        #graphs.append(line_graph)
    return graphs

def gnn_to_line(batch_graph, max_n_label):
    start = time.time()
    pool = mp.Pool(16)
    #pool = mp.Pool(mp.cpu_count())
    results = pool.map_async(parallel_line_worker, [(graph, max_n_label) for graph in batch_graph])
    remaining = results._number_left
    pbar = tqdm(total=remaining)
    while True:
        pbar.update(remaining - results._number_left)
        if results.ready(): break
        remaining = results._number_left
        time.sleep(1)
    results = results.get()
    pool.close()
    pbar.close()
    g_list = [g for g in results]
    return g_list

def parallel_line_worker(x):
    return to_line(*x)

def to_line(graph, max_n_label):
    edges = graph.edge_pairs
    edge_feas = edge_fea(graph, max_n_label)/2
    edges, feas = to_undirect(edges, edge_feas)
    edges = torch.tensor(edges)
    data = Data(edge_index=edges, edge_attr=feas)
    data.num_nodes = graph.num_nodes
    data = LineGraph()(data)
    data.num_nodes = graph.num_edges
    data['y'] = torch.tensor([graph.label])
    return data

def to_edgepairs(graph):
    x, y = zip(*graph.edges())
    num_edges = len(x)
    edge_pairs = np.ndarray(shape=(num_edges, 2), dtype=np.int32)
    edge_pairs[:, 0] = x
    edge_pairs[:, 1] = y
    edge_pairs = edge_pairs.flatten()
    return edge_pairs

def to_linegraphs(batch_graphs, max_n_label):
    graphs = []
    pbar = tqdm(batch_graphs, unit='iteration')
    for graph in pbar:
        edges = graph.edge_pairs
        edge_feas = edge_fea(graph, max_n_label)/2
        edges, feas = to_undirect(edges, edge_feas)
        edges = torch.tensor(edges)
        data = Data(edge_index=edges, edge_attr=feas)
        data.num_nodes = graph.num_nodes
        data = LineGraph()(data)
        data['y'] = torch.tensor([graph.label])
        data.num_nodes = graph.num_edges
        # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        # print(graph.average_feature)
        # # Assuming graph.average_feature has been calculated
        # print("Before reshaping:", graph.average_feature.shape)  # Should be [2000]

        # Reshape to [200, 10]
        # data['average_feature'] = graph.average_feature.view(-1, graph.average_feature.shape[0])
        if graph.average_feature is None: 
            data.average_feature = torch.tensor([])
        else: 
            data.average_feature = graph.average_feature.view(-1, graph.average_feature.shape[0]).float()

        # print("After reshaping:", data['average_feature'].shape)  # Should be [200, 10]

        # # data['average_feature']= torch.tensor([graph.average_feature])#.unsqueeze(1).repeat(1, data.num_nodes)
        # print(data.average_feature.shape)
        graphs.append(data)
    return graphs
def to_linegraphs_usefeature(batch_graphs, max_n_label):
    graphs = []
    pbar = tqdm(batch_graphs, unit='iteration')
    for graph in pbar:
        edges = graph.edge_pairs
        edge_feas = edge_fea(graph, max_n_label)/2
        if graph.node_features is not None:
            edges, feas = to_undirect_feature(edges, edge_feas,graph.node_features)
        else: 
            edges, feas = to_undirect(edges, edge_feas)
        edges = torch.tensor(edges)
 
        data = Data(edge_index=edges, edge_attr=feas)
        data.num_nodes = graph.num_nodes
        data = LineGraph()(data)
        data['y'] = torch.tensor([graph.label])
        data.num_nodes = graph.num_edges
        graphs.append(data)
    return graphs


def edge_fea(graph, max_n_label):
    node_tag = torch.zeros(graph.num_nodes, max_n_label+1)
    tags = graph.node_tags
    tags = torch.LongTensor(tags).view(-1,1)
    node_tag.scatter_(1, tags, 1)
    return node_tag

def edge_fea2(labels, edges):
    feas = []
    for i in range(edges.shape[1]):
        fea = [labels[edges[0][i]], labels[edges[1][i]]]
        fea.sort()
        feas.append(fea)
    feas = np.reshape(feas, [-1, 2])
    feas = np.array([feas[:,0], feas[:,1]], dtype=np.float32)
    return torch.tensor(feas/2)
    
def to_undirect2(edges):
    edges = np.reshape(edges, (-1,2 ))
    sr = np.array([edges[:,0], edges[:,1]], dtype=np.int64)
    rs = np.array([edges[:,1], edges[:,0]], dtype=np.int64)
    target_edge = np.array([[0,1],[1,0]])
    return np.concatenate([target_edge, sr, rs], axis=1)
    
def to_undirect(edges, edge_fea):
    edges = np.reshape(edges, (-1,2 ))
    sr = np.array([edges[:,0], edges[:,1]], dtype=np.int64)
    fea_s = edge_fea[sr[0,:], :]
    fea_s = fea_s.repeat(2,1)
    fea_r = edge_fea[sr[1,:], :]
    fea_r = fea_r.repeat(2,1)
    fea_body = torch.cat([fea_s, fea_r], 1)
    rs = np.array([edges[:,1], edges[:,0]], dtype=np.int64)
    return np.concatenate([sr, rs], axis=1), fea_body
def to_undirect_feature(edges, edge_fea , nodefeature):
    edges = np.reshape(edges, (-1,2 ))
    sr = np.array([edges[:,0], edges[:,1]], dtype=np.int64)
    fea_s = edge_fea[sr[0,:], :]
    fea_s = fea_s.repeat(2,1)

    nodefea_s = nodefeature[sr[0,:], :]
    nodefea_s = nodefea_s.repeat(2,1)

    fea_r = edge_fea[sr[1,:], :]
    fea_r = fea_r.repeat(2,1)

    nodefea_r = nodefeature[sr[1,:], :]
    nodefea_r = nodefea_r.repeat(2,1)

    # nodefea_s = torch.tensor(nodefea_s)  # Convert to PyTorch tensor
    # nodefea_r = torch.tensor(nodefea_r)  # Convert to PyTorch tensor
    nodefea_sum = nodefea_s + nodefea_r  # shape: (10, 20)
    # print(nodefea_sum.dtype)
    nodefea_sum = nodefea_sum.float()
    nodefea_s = nodefea_s.float()
    nodefea_r = nodefea_r.float()
    # print(nodefea_sum.dtype)

    # print(fea_s.shape)  # Output: torch.Size([10, 204])
    # print(fea_r.shape)  # Output: torch.Size([10, 204])
    # print(nodefea_s.shape)  # Output: torch.Size([10, 204])
    # print(nodefea_r.shape)  # Output: torch.Size([10, 204])
    # print(nodefea_sum.shape)  # Output: torch.Size([10, 204])

    #fea_body = torch.cat([fea_s, fea_r ,nodefea_sum], 1)
#    fea_body = torch.cat([fea_s, fea_r ], 1)
    fea_body = torch.cat([fea_s, fea_r ,nodefea_s , nodefea_r], 1)
    #fea_body = torch.cat([nodefea_s , nodefea_r], 1)
    rs = np.array([edges[:,1], edges[:,0]], dtype=np.int64)
    return np.concatenate([sr, rs], axis=1), fea_body


def line_test(graph, label):
    edges = graph.edge_pairs
    edges= to_undirect2(edges)
    feas = edge_fea2(label, edges)
    data = Data(edge_index=torch.tensor(edges), edge_attr=feas.T)
    data = LineGraph()(data)
    elist = data['edge_index'].numpy()
    #elist = [(elist[0][i], elist[1][i]) for i in range(len(elist[0]))]
    #nx_graph = nx.Graph()
    #nx_graph.add_edges_from(elist)
    #return nx_graph, data['x'].numpy()
    #return nx
    
    
    
    