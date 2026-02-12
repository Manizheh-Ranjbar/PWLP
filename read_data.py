import scipy.io as sio
import os.path
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import coalesce, to_undirected , to_scipy_sparse_matrix
import scipy.sparse as ssp
import numpy as np
from ogb.linkproppred import PygLinkPropPredDataset
import random , math
from torch_geometric.utils import (negative_sampling, add_self_loops, train_test_split_edges)
import torch


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")



def get_root_dir():
    file_dir = os.path.dirname(os.path.realpath('__file__'))
    return file_dir

def store_data_as_mat(adj_matrix , node_features , split_edge , edge_index , data_name):
    file_dir= get_root_dir()
    data_dir = os.path.join(file_dir, 'data/{}.mat'.format(data_name))
    # data_to_save = {'net': adj_matrix , 'node_features': node_features,'split_edge': split_edge,'edge_index': edge_index}
    data_to_save = {'net': adj_matrix , 'node_features': node_features,'edge_index': edge_index}

    # Save the dictionary to a .mat file
    sio.savemat(data_dir, data_to_save)

    print(f"Data saved to {data_dir}")

def load_data(dataset_name,test_ratio,val_ratio, max_train_num):
    node_features = None
    adj_matrix = None
    split_edge = None
    edge_index = None
    node_labels = None

    if dataset_name in ["Pubmed","Cora","Citeseer"]:
        adj_matrix,edge_index, node_features, node_labels =Read_data_Cora_PubMed_CiteSeer(dataset_name)
        train_pos, train_neg, test_pos, test_neg = sample_neg(adj_matrix, test_ratio, max_train_num=max_train_num)
        num_val = int(len(edge_index[0]) * val_ratio/2)  
        split_edge = {'train': {}, 'valid': {}, 'test': {}}
        split_edge['train']['edge'] = (
            train_pos[0][num_val:],  # Slice the first array
            train_pos[1][num_val:]   # Slice the second array
        )
        # print(split_edge['train']['edge'] )
        split_edge['train']['negedge'] =  (
            train_neg[0][num_val:],  # Slice the first array
            train_neg[1][num_val:]   # Slice the second array
        )
        split_edge['valid']['edge'] = (
            train_pos[0][:num_val],  # Slice the first array
            train_pos[1][:num_val]   # Slice the second array
        )     
        split_edge['valid']['edge_neg'] = (
            train_neg[0][:num_val],  # Slice the first array
            train_neg[1][:num_val]   # Slice the second array
        )#train_neg[num_val:]
        split_edge['test']['edge'] = test_pos
        split_edge['test']['edge_neg'] = test_neg

    elif dataset_name in ["Router","ADV","SMG", "HPD", "EML", "KHN" , "ZWL" ]:

        adj_matrix= read_data_mat(dataset_name)
        split_edge = split_edges(adj_matrix)

    elif dataset_name in ["collab", "citation2","ddi", "ppa"]:
    # elif dataset_name in ["ogbl-collab", "ogbl-citation2","ogbl-ddi"]:
        adj_matrix,edge_index, node_features , split_edge = read_data_OGBL(dataset_name)
        if dataset_name in ["citation2"]:
            split_edge= CorrectSplit_citation2(split_edge)

        # return adj_matrix , node_features , split_edge , edge_index
    else:
        print("dataset not found!!!")
        return adj_matrix , node_features , split_edge , edge_index
    print("data name: ", dataset_name)
    print("Shape: ",adj_matrix.shape)
    print("#Node: ",adj_matrix.shape[0])
    print("#Edges: ",adj_matrix.nnz //2)
    # print("#Edges: ", adj_matrix.data.shape)
    if node_features is not None:
        print("#NodeFeature:", node_features.shape[1] )

    return adj_matrix , node_features , split_edge , edge_index,split_edge



def CorrectSplit_citation2(split_edge):
    split_edge_new = {'train': {}, 'valid': {}, 'test': {}}
    for split in ['train', 'valid', 'test']:
        print(split_edge[split])
        source = split_edge[split]['source_node'].cpu().numpy()#.T
        # source_ = split_edge[split]['source_node'].cpu()
        print("source")
        print(source)
        print(source.shape)
        # source1 = source_.view(-1, 1)
        # print(source1)
        # print(source1.shape)
        # source1 = source_.view(-1, 1).repeat(1, 1000)
        # print(source1)
        # print(source1.shape)
        # source1 = source_.view(-1, 1).repeat(1, 1000).view(-1)
        # print(source1)
        # print(source1.shape)

        # print(source)
        # print(source.shape)
        print("********************")
        # for edge_type in ['target_node', 'target_node_neg']:
        target = split_edge[split]["target_node"].cpu().numpy()
        split_edge_new[split]['edge'] = np.vstack([source , target])
        print(len(source))
        print(len(target))
        print(split_edge_new[split]['edge'] )
        if 'target_node_neg' in split_edge[split]:  # Check if the key exists
            # target = split_edge[split]['target_node_neg'].cpu()
            # print(len(target))
            # print((target.shape))
            # print((target))
            target_neg = split_edge[split]['target_node_neg'].cpu().view(-1).numpy()
            print(len(target_neg))
            print((target_neg.shape))
            print((target_neg))
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            source_repeat = torch.tensor(source)
            source_repeat = source_repeat.view(-1, 1).repeat(1, 1000).view(-1).numpy()
            print(len(source_repeat))
            print((source_repeat.shape))
            print((source_repeat))

            split_edge_new[split]['edge_neg'] = np.vstack([source_repeat , target_neg])
    print(split_edge_new["test"]["edge_neg"])
    print(split_edge_new["test"]["edge_neg"].shape)
    return split_edge_new


def split_edges(dataset,val_ratio: float=0.10, test_ratio: float=0.2):
    net = dataset
    attributes = None
    train_pos, train_neg, test_pos, test_neg = sample_neg(net, test_ratio)
    total_pos = int(len(test_pos[0])+len(train_pos[0]))
    total_neg = int(len(test_neg[0])+len(train_neg[0]))
    num_val = int((total_pos+total_neg)* val_ratio/2)
    num_test = int(len(test_pos[0]))
    num_train = int(len(train_pos[0])-num_val)

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = (
        train_pos[0][num_val:],  # Slice the first array
        train_pos[1][num_val:]   # Slice the second array
    )
    # print(split_edge['train']['edge'] )
    split_edge['train']['negedge'] =  (
        train_neg[0][num_val:],  # Slice the first array
        train_neg[1][num_val:]   # Slice the second array
    )
    split_edge['valid']['edge'] = (
        train_pos[0][:num_val],  # Slice the first array
        train_pos[1][:num_val]   # Slice the second array
    )        
    split_edge['valid']['edge_neg'] = (
        train_neg[0][:num_val],  # Slice the first array
        train_neg[1][:num_val]   # Slice the second array
    )
    split_edge['test']['edge'] = test_pos
    split_edge['test']['edge_neg'] = test_neg



    return split_edge

def read_data_mat(dataset_name):
    file_dir =get_root_dir()
    data_dir = os.path.join(file_dir, '../dataset/{}.mat'.format(dataset_name))
    # data_dir = os.path.join(file_dir, '../dataset/{}.mat'.format(dataset_name))
    data = sio.loadmat(data_dir)
    net = data['net']
    return net

def read_data_OGBL(dataset_name):
    file_dir =get_root_dir()
    data = PygLinkPropPredDataset(name=f'ogbl-{dataset_name}', root=os.path.join(get_root_dir(), "../dataset", f'ogbl-{dataset_name}'))
    # data = PygLinkPropPredDataset(name=dataset_name, root=os.path.join(get_root_dir(), "../dataset", dataset_name))
    net = data[0]
    split_edge = data.get_edge_split()
    # Assume data is your torch_geometric.data.Data object
    edge_index = net.edge_index
    node_features = net.x
    num_nodes = net.num_nodes
    if 'edge_neg' not in split_edge["train"]:
        pos_edge = split_edge["train"]['edge']
        # use presampled  negative training edges for ogbl-vessel
        new_edge_index, _ = add_self_loops(edge_index)
        neg_edge = negative_sampling(
            new_edge_index, num_nodes=num_nodes,
            num_neg_samples=pos_edge.size(1))
        split_edge["train"]['negedge'] = neg_edge
    edge_index = to_undirected(edge_index)
    edge_index = coalesce(edge_index)
    mask = edge_index[0] <= edge_index[1]
    edge_index = edge_index[:, mask]
    num_edges = edge_index.shape[1] 
    adj_matrix = ssp.csc_matrix((np.ones(num_edges), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))
    adj_matrix = adj_matrix.maximum(adj_matrix.transpose())
    adj_matrix.data = np.clip(adj_matrix.data, 0, 1)  # Ensure no values greater than 1
    nodesindex=np.arange(num_nodes)
    adj_matrix[nodesindex,nodesindex] = 0  # remove self-loops
    split_edge= convert_tensor_to_numpyarray_split_edge(split_edge)
    
    #########
    print(f"Negative samples: {neg_edge}")

    # assert np.all(neg_edge[0] < num_nodes) and np.all(neg_edge[1] < num_nodes), "Negative sampling returned invalid node indices!"
    num_nodes_ = max(edge_index.flatten().numpy()) + 1  # Ensure it covers all nodes
    if(num_nodes_>num_nodes):
        print("error num nodes")
        exit(0)
    max_node_id = max(edge_index.flatten().numpy())  # Convert to NumPy array for safety
    print(f"Max node ID in edge_index: {max_node_id}, Num nodes: {num_nodes}")
    if max_node_id >= num_nodes: 
        print("error num nodes")
        exit(0)
    print("&&&&&&&&&&&&&&&&&&&")
    return adj_matrix, edge_index, node_features,split_edge


def convert_tensor_to_numpyarray_split_edge(split_edge):
    for split in ['train', 'valid', 'test']:
        for edge_type in ['edge', 'edge_neg']:
            if edge_type in split_edge[split]:  # Check if the key exists
                numpy_array = split_edge[split][edge_type].cpu().numpy().T
                split_edge[split][edge_type] = numpy_array
    return  split_edge


def Read_data_Cora_PubMed_CiteSeer(dataset_name):
    file_dir =get_root_dir()
    dataset =Planetoid(name=dataset_name, root=os.path.join(get_root_dir(), "../dataset"))
    # dataset =Planetoid(name=dataset_name, root=os.path.join(get_root_dir(), "../dataset"))
    data = dataset[0]
    node_features = data.x
    node_labels = data.y
    edge_index= data.edge_index
    edge_index = to_undirected(edge_index)
    edge_index = coalesce(edge_index)
    mask = edge_index[0] <= edge_index[1]
    edge_index = edge_index[:, mask]
    num_nodes = node_features.shape[0]
    num_edges = edge_index.shape[1] 
    adj_matrix = ssp.csc_matrix((np.ones(num_edges), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))
    adj_matrix = adj_matrix.maximum(adj_matrix.transpose())
    adj_matrix.data = np.clip(adj_matrix.data, 0, 1)  # Ensure no values greater than 1
    nodesindex=np.arange(num_nodes)
    adj_matrix[nodesindex,nodesindex] = 0  # remove self-loops
 
    return adj_matrix,edge_index, node_features, node_labels

def sample_neg(net, test_ratio=0.2, train_pos=None, test_pos=None, max_train_num=None):
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

def split_data(net, data_name,split_edge,edge_index,test_ratio , max_train_num):
    if data_name in ["ogbl-collab", "ogbl-citation2","ogbl-ddi"]:
        if data_name != 'ogbl-citation2':
            train_pos = split_edge['train']['edge'].t().numpy()
            train_neg = split_edge['train']['edge'].t().numpy()
            # #
            # num_nodes= net.shape[0]
            # new_edge_index, _ = add_self_loops(edge_index)
            # neg_edge = negative_sampling(
            # new_edge_index, num_nodes=num_nodes,
            # num_neg_samples=train_pos.size(1))
            
        
            # # subsample for neg_edge
            # np.random.seed(123)
            # num_neg = neg_edge.size(1)
            # perm = np.random.permutation(num_neg)
            # perm = perm[:int(num_neg)]
            # train_neg = neg_edge[:, perm]
            # #
            valid_pos = split_edge['valid']['edge'].t().numpy()
            valid_neg = split_edge['valid']['edge_neg'].t().numpy()
            test_pos = split_edge['test']['edge'].t().numpy()
            test_neg = split_edge['test']['edge_neg'].t().numpy()
        # else:
        #     source_edge, target_edge = split_edge['train']['source_node'], split_edge['train']['target_node']
        #     pos_train_edge = torch.cat([source_edge.unsqueeze(0), target_edge.unsqueeze(0)], dim=0)

        #     # idx = torch.randperm(split_edge['train']['source_node'].numel())[:split_edge['valid']['source_node'].size(0)]
        #     # source, target = split_edge['train']['source_node'][idx], split_edge['train']['target_node'][idx]
        #     # train_val_edge = torch.cat([source.unsqueeze(0), target.unsqueeze(0)], dim=0)

        #     source, target = split_edge['valid']['source_node'],  split_edge['valid']['target_node']
        #     pos_valid_edge = torch.cat([source.unsqueeze(0), target.unsqueeze(0)], dim=0)
        #     val_neg_edge = split_edge['valid']['target_node_neg'] 

        #     neg_valid_edge = torch.stack([source.repeat_interleave(val_neg_edge.size(1)), 
        #                             val_neg_edge.view(-1)])

        #     source, target = split_edge['test']['source_node'],  split_edge['test']['target_node']
        #     pos_test_edge = torch.cat([source.unsqueeze(0), target.unsqueeze(0)], dim=0)
        #     test_neg_edge = split_edge['test']['target_node_neg']

        #     neg_test_edge = torch.stack([source.repeat_interleave(test_neg_edge.size(1)), 
        #                             test_neg_edge.view(-1)])

        return train_pos, train_neg, test_pos, test_neg
    else:
        train_pos, train_neg, test_pos, test_neg = sample_neg(net, test_ratio=test_ratio, max_train_num=max_train_num)
    return train_pos, train_neg, test_pos, test_neg

