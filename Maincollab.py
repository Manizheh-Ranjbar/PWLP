from timeit import default_timer
from torch_geometric.profile import profileit, timeit

import torch
import numpy as np
import sys, copy, math, time, pdb
import pickle as pickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
import argparse
from util_functions import *
from torch_geometric.data import DataLoader
from model import Net
from sklearn.metrics import average_precision_score
import torch.optim as optim
from Evaluate import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from read_data import load_data , create_directory_if_not_exists
from scipy.sparse import csc_matrix


def matrix_factorization(R, K, steps=5000, alpha=0.0002, beta=0.02):
    """
    Perform Matrix Factorization to predict missing values in the adjacency matrix.

    Parameters:
    R (numpy.ndarray): The adjacency matrix (with 0 indicating missing values).
    K (int): Number of latent features.
    steps (int): Number of iterations for gradient descent.
    alpha (float): Learning rate.
    beta (float): Regularization parameter.

    Returns:
    numpy.ndarray: Reconstructed adjacency matrix.
    """
    # Initialize user and item latent feature matrices
    num_users, num_items = R.shape
    P = np.random.rand(num_users, K)
    Q = np.random.rand(num_items, K)

    # Transpose Q for easier calculations
    Q = Q.T

    # Gradient descent
    for step in range(steps):
        for i in range(num_users):
            for j in range(num_items):
                if R[i, j] > 0:  # Only update for existing values
                    eij = R[i, j] - np.dot(P[i, :], Q[:, j])
                    for k in range(K):
                        P[i, k] += alpha * (2 * eij * Q[k, j] - beta * P[i, k])
                        Q[k, j] += alpha * (2 * eij * P[i, k] - beta * Q[k, j])

        # Compute total error
        error = 0
        for i in range(num_users):
            for j in range(num_items):
                if R[i, j] > 0:
                    error += (R[i, j] - np.dot(P[i, :], Q[:, j])) ** 2
                    for k in range(K):
                        error += (beta / 2) * (P[i, k]**2 + Q[k, j]**2)

        if step % 100 == 0:
            print(f"Step {step}, Error: {error}")
        if error < 0.001:
            break

    # Reconstruct the matrix
    return np.dot(P, Q)


def loop_dataset_gem(data_name, result_dir, classifier, loader, seed, optimizer=None,save_plot=False ):
    total_loss = []
    all_targets = []
    all_scores = []
    all_embeddings = []

    pbar = tqdm(loader, unit='batch')

    n_samples = 0
    for batch in pbar:
        all_targets.extend(batch.y.tolist())
        logits, loss, acc, embeddings = classifier(batch)
        all_scores.append(logits[:, 1].cpu().detach())
        all_embeddings.append(embeddings.cpu().detach())

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.data.cpu().detach().numpy()
        
        pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc) )
        total_loss.append( np.array([loss, acc]) * len(batch.y))
        
        n_samples += len(batch.y)

    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    all_scores = torch.cat(all_scores).cpu().numpy()
    all_embeddings = torch.cat(all_embeddings)
    # np.savetxt('test_scores.txt', all_scores)  # output test predictions
    all_targets = np.array(all_targets)
    avg_precision = average_precision_score(all_targets, all_scores)
    fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    #     avg_loss = np.concatenate((avg_loss, [auc, avg_precision]))

    k_list  = [1, 3, 10, 20, 30, 50, 100]
    # hits_ogb_n = hits_at_n_ogb(all_scores, all_targets, k_list)
    # mrr_obg_value=  evaluate_mrr_scaled(all_scores, all_targets)
    pos_test_pred = all_scores[all_targets == 1]
    neg_test_pred = all_scores[all_targets == 0]

    hits_ogb_n = evaluate_hits(pos_test_pred, neg_test_pred, k_list,data_name)#hits_at_n_ogb(all_scores, all_targets, k_list)
    mrr_obg_value=  evaluate_mrr_scaled(pos_test_pred, neg_test_pred,data_name)

    pltval = ''
    if optimizer==None:
        pltval='_val'
    
    if save_plot==True: 
        draw_TSNE_embeding_nodeclass(data_name, all_embeddings, all_targets, save_path=result_dir+'/acc_results'+data_name+pltval+str(seed)+'_tsne.png')
        save_data_Embedding(all_embeddings,all_targets,data_name+pltval, result_dir)
    
    avg_loss = np.concatenate((avg_loss, [auc, avg_precision, hits_ogb_n['Hits@1'],hits_ogb_n['Hits@3'],hits_ogb_n['Hits@10'],hits_ogb_n['Hits@20'],hits_ogb_n['Hits@30'],hits_ogb_n['Hits@50'],hits_ogb_n['Hits@100'], mrr_obg_value]))

    return avg_loss

def main():
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", RuntimeWarning)

    cmd_opt = argparse.ArgumentParser(description='Argparser for graph_classification')
    cmd_opt.add_argument('-mode', default='cpu', help='cpu/gpu')
    cmd_opt.add_argument('-gm', default='DGCNN', help='gnn model to use')
    cmd_opt.add_argument('-data', default=None, help='data folder name')
    cmd_opt.add_argument('-batch_size', type=int, default=50, help='minibatch size')
    # cmd_opt.add_argument('-seed', type=int, default=1, help='seed')
    cmd_opt.add_argument('-feat_dim', type=int, default=0, help='dimension of discrete node feature (maximum node tag)')
    cmd_opt.add_argument('-edge_feat_dim', type=int, default=0, help='dimension of edge features')
    cmd_opt.add_argument('-num_class', type=int, default=0, help='#classes')
    cmd_opt.add_argument('-fold', type=int, default=1, help='fold (1..10)')
    cmd_opt.add_argument('-test_number', type=int, default=0, help='if specified, will overwrite -fold and use the last -test_number graphs as testing data')
    cmd_opt.add_argument('-num_epochs', type=int, default=1000, help='number of epochs')
    cmd_opt.add_argument('-latent_dim', type=str, default='64', help='dimension(s) of latent layers')
    cmd_opt.add_argument('-sortpooling_k', type=float, default=30, help='number of nodes kept after SortPooling')
    cmd_opt.add_argument('-conv1d_activation', type=str, default='ReLU', help='which nn activation layer to use')
    cmd_opt.add_argument('-out_dim', type=int, default=1024, help='graph embedding output size')
    cmd_opt.add_argument('-hidden', type=int, default=100, help='dimension of mlp hidden layer')
    cmd_opt.add_argument('-max_lv', type=int, default=4, help='max rounds of message passing')
    cmd_opt.add_argument('-learning_rate', type=float, default=0.0001, help='init learning_rate')
    cmd_opt.add_argument('-dropout', type=bool, default=False, help='whether add dropout after dense layer')
    cmd_opt.add_argument('-printAUC', type=bool, default=False, help='whether to print AUC (for binary classification only)')
    cmd_opt.add_argument('-extract_features', type=bool, default=False, help='whether to extract final graph features')

    cmd_args, _ = cmd_opt.parse_known_args()

    cmd_args.latent_dim = [int(x) for x in cmd_args.latent_dim.split('-')]
    if len(cmd_args.latent_dim) == 1:
        cmd_args.latent_dim = cmd_args.latent_dim[0]

    #################

    parser = argparse.ArgumentParser(description='Link Prediction')
    # general settings
    parser.add_argument('--mask',action='store_true', default=False, help='mask test data')
    parser.add_argument('--result-dir', default='../results/PWSLP', help='network name')
    parser.add_argument('--data-name', default='BUP', help='network name')
    parser.add_argument('--train-name', default=None, help='train name')
    parser.add_argument('--test-name', default=None, help='test name')
    parser.add_argument('--max-train-num', type=int, default=1000000, 
                        help='set maximum number of train links (to fit into memory)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--test-ratio', type=float, default=0.2,   help='ratio of test links')
    parser.add_argument('--val-ratio', type=float, default=0.1,   help='ratio of test links')
    parser.add_argument('--feature', type=int, default=0, help="3: Random Feature, 1: Degree one hot feature, 2:actual feature, 3:None")
    parser.add_argument('--pca', type=int, default=10, help="3: dimension of PAC, Cora 10, citeseer 50")
    # model settings
    parser.add_argument('--hop', default=2, metavar='S', 
                        help='enclosing subgraph hop number, \
                        options: 1, 2,..., "auto"')
    parser.add_argument('--max-nodes-per-hop', default=100, 
                        help='if > 0, upper bound the # nodes per hop by subsampling')



    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed) 
    torch.manual_seed(args.seed)
    if args.hop != 'auto':
        args.hop = int(args.hop)
    if args.max_nodes_per_hop is not None:
        args.max_nodes_per_hop = int(args.max_nodes_per_hop)




    '''Prepare data'''
    args.file_dir = os.path.dirname(os.path.realpath('__file__'))
    # args.result_dir = os.path.join(args.file_dir, '../results/PWSLP/{}'.format(args.data_name))
    if args.mask:
        args.result_dir = os.path.join(args.file_dir, '../results/PWSLP/{}/Masked/{}/{}'.format(args.pca, args.data_name,args.feature))
    else:
        args.result_dir = os.path.join(args.file_dir, '../results/PWSLP/{}/UnMasked/{}/{}'.format(args.pca, args.data_name,args.feature))
    create_directory_if_not_exists(args.result_dir)
    
    ############ read data 
    # args.data_dir = os.path.join(args.file_dir, '../dataset/{}.mat'.format(args.data_name))
    # data = sio.loadmat(args.data_dir)
    # net = data['net']
    # attributes = None
    #Sample train and test links
    # print(args.test_ratio,args.val_ratio)
    net , features, _,_,split_edge = load_data(args.data_name,args.test_ratio,args.val_ratio, max_train_num=args.max_train_num)
    ############# PCA on node feature###################
    # print(type(net))
    # R = net.toarray()

    # reconstructed_matrix = matrix_factorization(R, 2)
    # # Convert the NumPy array back to a CSC matrix
    # net = csc_matrix(np.round(reconstructed_matrix, 2).astype(int))
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    if features==None or args.feature==3 : 
        attributes = None
    else: 
        # this dont work for random or other kind of feature except real feature
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Apply PCA
        pca = PCA(n_components=args.pca)  # You can choose the number of components
        attributes = pca.fit_transform(features_scaled)
        attributes = torch.tensor(attributes)  # Convert to PyTorch tensor

        # print(features.shape)
        # print(attributes.shape)
        
    ###########################################################################
    # train_pos, train_neg, test_pos, test_neg = sample_neg(net, args.test_ratio, max_train_num=args.max_train_num)
    train_pos= split_edge['train']['edge'] 
    train_neg =    split_edge['train']['negedge'] 
    val_pos= split_edge['valid']['edge'] 
    val_neg =    split_edge['valid']['edge_neg'] 
    test_pos=  split_edge['test']['edge'] 
    test_neg=    split_edge['test']['edge_neg'] 
    # start = default_timer()

    #save neg_link_train
    torch.save({
            'train_neg': train_neg,
        }, '../dataset/subgraphs_PWLP/'+args.data_name+'_train_neg_link'+str(args.seed)+'.pt')

    train_ratio = 1
    train_size = train_pos[0].shape[0]  # Number of samples
    # Select only 10% of the training set
    num_train_samples = int(train_ratio * train_size)  # Compute 10% of train set size
    # sampled_indices = np.random.choice(train_size, num_train_samples, replace=False)
    # Select the 10% from train_pos
    # print(sampled_indices)
    # print(train_pos[0].shape)
    train_pos = (train_pos[0][:num_train_samples], train_pos[1][:num_train_samples])
    train_neg = (train_neg[0][:num_train_samples], train_neg[1][:num_train_samples])


    '''Train and apply classifier'''
    A = net.copy()  # the observed network
    # print("args.mask",args.mask)
    if args.mask:
        # Your code here
        print("Masked Test Data......................")
        A[test_pos[0], test_pos[1]] = 0  # mask test links
        A[test_pos[1], test_pos[0]] = 0  # mask test links
    else: 
        print("UnMasked Test Data*******************")
    A.eliminate_zeros()
    ############## load or create subgraphs
    for split in ['pos_train', 'neg_train', 'pos_valid', 'neg_valid', 'pos_test', 'neg_test']:
        if args.mask:
            save_path = '../dataset/subgraphs_PWLP/'+args.data_name+split+'_saved_graphs_Masked_'+str(args.seed)+'.pt'
            # save_path = '../dataset/subgraphs_PWLP/'+args.data_name+'saved_graphs_Masked_'+str(args.seed)+'.pt'
        else: 
            save_path = '../dataset/subgraphs_PWLP/'+args.data_name+split+'_saved_graphs_UnMasked_'+str(args.seed)+'.pt'
            # save_path = '../dataset/subgraphs_PWLP/'+args.data_name+'saved_graphs_UnMasked_'+str(args.seed)+'.pt'

        if os.path.exists(save_path):
            print("📂 Loading cached graphs...")
            data_dict = torch.load(save_path)
            train_graphs = data_dict['train_graphs']
            val_graphs = data_dict['val_graphs']
            test_graphs = data_dict['test_graphs']
            max_n_label = data_dict['max_n_label']
        else:
            create_directory_if_not_exists("../dataset/subgraphs_PWLP/")
            print("🔄 Processing subgraphs from scratch...")
            train_graphs, val_graphs, test_graphs, max_n_label = links2subgraphs(
                args.result_dir, A, train_pos, train_neg, val_pos, val_neg, test_pos, test_neg,
                args.hop, args.max_nodes_per_hop, attributes
            )
            torch.save({
                'train_graphs': train_graphs,
                'val_graphs': val_graphs,
                'test_graphs': test_graphs,
                'max_n_label': max_n_label
            }, save_path)

    ############## end of load or create subgraphs

    # train_graphs,val_graphs, test_graphs, max_n_label = links2subgraphs(args.result_dir, A, train_pos, train_neg,val_pos, val_neg, test_pos, test_neg, args.hop, args.max_nodes_per_hop, attributes)
    print(('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs))))
    start = default_timer()
    '''
    tnumgraph_node = 0
    tnumgraph_edge = 0
    count=0
    for graph in train_graphs:
        if(graph.num_nodes==1):
            if(count<len(train_pos)):
                print(train_pos[count])
            else:
                print(train_neg[count-len(train_pos)])
        count=count+1
        tnumgraph_node=tnumgraph_node+graph.num_nodes
        tnumgraph_edge=tnumgraph_edge+graph.num_edges
        with open(args.result_dir+'/NumNodes_Subgraph_simpleSEAL.txt', 'a+') as f:
            f.write(str(graph.num_nodes)+"\n")
        # with open('../results/RWLGLP/unmasked02_seed1_topNodes'+str(Num_top_Nodes)+'nW'+str(num_walks)+'wl'+str(walk_length)+'_withoutedge_win/NumedgesSMG_Subgraph.txt', 'a+') as f:
        with open(args.result_dir+'/Numedges_Subgraph_simpleSEAL.txt', 'a+') as f:
            f.write(str(graph.num_edges)+"\n")

    for graph in val_graphs:
        tnumgraph_node=tnumgraph_node+graph.num_nodes
        tnumgraph_edge=tnumgraph_edge+graph.num_edges
        with open(args.result_dir+'/NumNodes_Subgraph_simpleSEAL.txt', 'a+') as f:
            f.write(str(graph.num_nodes)+"\n")
        # with open('../results/RWLGLP/unmasked02_seed1_topNodes'+str(Num_top_Nodes)+'nW'+str(num_walks)+'wl'+str(walk_length)+'_withoutedge_win/NumedgesSMG_Subgraph.txt', 'a+') as f:
        with open(args.result_dir+'/Numedges_Subgraph_simpleSEAL.txt', 'a+') as f:
            f.write(str(graph.num_edges)+"\n")

    for graph in test_graphs:
        tnumgraph_node=tnumgraph_node+graph.num_nodes
        tnumgraph_edge=tnumgraph_edge+graph.num_edges
        with open(args.result_dir+'/NumNodes_Subgraph_simpleSEAL.txt', 'a+') as f:
            f.write(str(graph.num_nodes)+"\n")
        # with open('../results/RWLGLP/unmasked02_seed1_topNodes'+str(Num_top_Nodes)+'nW'+str(num_walks)+'wl'+str(walk_length)+'_withoutedge_win/NumedgesSMG_Subgraph.txt', 'a+') as f:
        with open(args.result_dir+'/Numedges_Subgraph_simpleSEAL.txt', 'a+') as f:
            f.write(str(graph.num_edges)+"\n")
    '''
    train_lines = to_linegraphs(train_graphs, max_n_label)      
    val_lines = to_linegraphs(val_graphs, max_n_label)      
    test_lines = to_linegraphs(test_graphs, max_n_label)
    print("converted to line graph")
    ##########################
    # tnumgraph_node = 0
    # tnumgraph_edge = 0
    # count=0
    # for graph in train_lines:
    #     # if(graph.num_nodes==1):
    #     #     print(graph.node_tags)
    #     #     print(count)
    #     #     print(len(train_pos[0]))
    #     #     print(len(train_neg[0]))
    #     #     print(count-len(train_pos[0]))
    #     #     if(count<len(train_pos)):
    #     #         print(train_pos[0][count], train_pos[1][count])
    #     #     else:
    #     #         print(train_neg[0][count], train_neg[1][count])
    #     count=count+1
    #     tnumgraph_node=tnumgraph_node+graph.num_nodes
    #     tnumgraph_edge=tnumgraph_edge+graph.num_edges
    #     with open(args.result_dir+'/NumNodes_Subgraph_simpleLGLP.txt', 'a+') as f:
    #         f.write(str(graph.num_nodes)+"\n")
    #     # with open('../results/RWLGLP/unmasked02_seed1_topNodes'+str(Num_top_Nodes)+'nW'+str(num_walks)+'wl'+str(walk_length)+'_withoutedge_win/NumedgesSMG_Subgraph.txt', 'a+') as f:
    #     with open(args.result_dir+'/Numedges_Subgraph_simpleLGLP.txt', 'a+') as f:
    #         f.write(str(graph.num_edges)+"\n")

    # for graph in val_lines:
    #     tnumgraph_node=tnumgraph_node+graph.num_nodes
    #     tnumgraph_edge=tnumgraph_edge+graph.num_edges
    #     with open(args.result_dir+'/NumNodes_Subgraph_simpleLGLP.txt', 'a+') as f:
    #         f.write(str(graph.num_nodes)+"\n")
    #     # with open('../results/RWLGLP/unmasked02_seed1_topNodes'+str(Num_top_Nodes)+'nW'+str(num_walks)+'wl'+str(walk_length)+'_withoutedge_win/NumedgesSMG_Subgraph.txt', 'a+') as f:
    #     with open(args.result_dir+'/Numedges_Subgraph_simpleLGLP.txt', 'a+') as f:
    #         f.write(str(graph.num_edges)+"\n")

    # for graph in test_lines:
    #     tnumgraph_node=tnumgraph_node+graph.num_nodes
    #     tnumgraph_edge=tnumgraph_edge+graph.num_edges
    #     with open(args.result_dir+'/NumNodes_Subgraph_simpleLGLP.txt', 'a+') as f:
    #         f.write(str(graph.num_nodes)+"\n")
    #     # with open('../results/RWLGLP/unmasked02_seed1_topNodes'+str(Num_top_Nodes)+'nW'+str(num_walks)+'wl'+str(walk_length)+'_withoutedge_win/NumedgesSMG_Subgraph.txt', 'a+') as f:
    #     with open(args.result_dir+'/Numedges_Subgraph_simpleLGLP.txt', 'a+') as f:
    #         f.write(str(graph.num_edges)+"\n")
    # ##################


    
    # tnumgraph_node/len(train_graphs+val_graphs+test_graphs)
    # tnumgraph_edge/len(train_graphs+val_graphs+test_graphs)
    # Model configurations

    cmd_args.latent_dim = [32, 32, 32, 1]
    cmd_args.hidden = 128
    cmd_args.out_dim = 0
    cmd_args.dropout = True
    cmd_args.num_class = 2
    cmd_args.mode = 'gpu'
    cmd_args.num_epochs = 50
    cmd_args.learning_rate = 5e-3
    cmd_args.batch_size = 50
    # cmd_args.batch_size = 500000
    cmd_args.printAUC = True
    
    if args.feature == 3 :
        cmd_args.feat_dim = (max_n_label + 1)*2
    else: 
        cmd_args.feat_dim = (max_n_label + 1)*2

        # cmd_args.feat_dim = (max_n_label + 1)*2+(len(attributes[0]))
        # cmd_args.feat_dim = (max_n_label + 1)*2+(len(attributes[0]))*2

    cmd_args.attr_dim = 0

    train_loader = DataLoader(train_lines, batch_size=cmd_args.batch_size, shuffle=True)
    val_loader = DataLoader(val_lines, batch_size=cmd_args.batch_size, shuffle=False)
    test_loader = DataLoader(test_lines, batch_size=cmd_args.batch_size, shuffle=False)
    if attributes is None:
        len_atrib = 0
    else: 
        len_atrib= len(attributes[0])*2
    classifier = Net(cmd_args.feat_dim, cmd_args.hidden, cmd_args.latent_dim, cmd_args.dropout, len_atrib)
    if cmd_args.mode == 'gpu':
        classifier = classifier.to("cuda")

    optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)
 
#mark sanderson 

    # best_auc = 0
    # best_auc_acc = 0
    # best_acc = 0
    # best_acc_auc = 0
    start_training = default_timer()
    bestscore , bestscoretest =train(classifier,cmd_args,args,train_loader,val_loader,test_loader,optimizer)    
    epoch = cmd_args.num_epochs-1
    end = default_timer()
    print(f"Time taken for training run: {end - start_training:.2f} seconds")
    print(f"Time taken for run: {end - start:.2f} seconds")



    # end = default_timer()
    # end_training = end
    # print(f"Time taken for run: {end - start:.2f} seconds")
    # print(f"Time taken for training: {end_training - start_training:.2f} seconds")


    with open(args.result_dir+'/'+args.data_name+'acc_results'+'.txt', 'a+') as f:
        f.write(str(bestscoretest[1]) + '\n')
    with open(args.result_dir+'/'+args.data_name+'loss_results'+'.txt', 'a+') as f:
        f.write(str(bestscoretest[0]) + '\n')
    with open(args.result_dir+'/'+args.data_name+'auc_results'+'.txt', 'a+') as f:
        f.write(str(bestscoretest[2]) + '\n')
    with open(args.result_dir+'/'+args.data_name+'ap_results'+'.txt', 'a+') as f:
        f.write(str(bestscoretest[3]) + '\n')
    with open(args.result_dir+'/'+args.data_name+'hit1_results'+'.txt', 'a+') as f:
        f.write(str(bestscoretest[4]) + '\n')
    with open(args.result_dir+'/'+args.data_name+'hit3_results'+'.txt', 'a+') as f:
        f.write(str(bestscoretest[5]) + '\n')
    with open(args.result_dir+'/'+args.data_name+'hit10_results'+'.txt', 'a+') as f:
        f.write(str(bestscoretest[6]) + '\n')
    with open(args.result_dir+'/'+args.data_name+'hit20_results'+'.txt', 'a+') as f:
        f.write(str(bestscoretest[7]) + '\n')
    with open(args.result_dir+'/'+args.data_name+'hit30_results'+'.txt', 'a+') as f:
        f.write(str(bestscoretest[8]) + '\n')

    with open(args.result_dir+'/'+args.data_name+'hit50_results'+'.txt', 'a+') as f:
        f.write(str(bestscoretest[9]) + '\n')

    with open(args.result_dir+'/'+args.data_name+'hit100_results'+'.txt', 'a+') as f:
        f.write(str(bestscoretest[10]) + '\n')
    with open(args.result_dir+'/'+args.data_name+'mrr_results'+'.txt', 'a+') as f:
        f.write(str(bestscoretest[11]) + '\n')
    print('\033[94maverage best_val of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f hit1 %.5f hit3 %.5f hit10 %.5f hit20 %.5f hit30 %.5f hit50 %.5f hit100 %.5f mmr %.5f\033[0m' % (epoch, bestscore[0], bestscore[1], bestscore[2], bestscore[3],bestscore[4],bestscore[5],bestscore[6],bestscore[7],bestscore[8],bestscore[9],bestscore[10],bestscore[11])) 
    print('\033[94maverage best_test of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f hit1 %.5f hit3 %.5f hit10 %.5f hit20 %.5f hit30 %.5f hit50 %.5f hit100 %.5f mmr %.5f\033[0m' % (epoch, bestscoretest[0], bestscoretest[1], bestscoretest[2], bestscoretest[3],bestscoretest[4],bestscoretest[5],bestscoretest[6],bestscoretest[7],bestscoretest[8],bestscoretest[9],bestscoretest[10],bestscoretest[11]))




    max_memory_allocated = torch.cuda.max_memory_allocated()
    print(f"Max allocated CUDA memory: {max_memory_allocated / (1024 ** 2)} MB")

    model_size_params = sum(p.numel() for p in classifier.parameters())
    print(f"NUM parameters: {model_size_params} parameters")

    # Calculate the size of the model in bytes
    model_size_bytes = model_size_params * 4  # 4 bytes per float32 parameter

    # Convert size to megabytes (1 MB = 1024 * 1024 bytes)
    model_size_mb = model_size_bytes / (1024 ** 2)

    print(f"NUM parameters: {model_size_params / 1e6:.3f}M parameters")
    print(f"Model size: {model_size_mb:.2f} MB")


    with open(args.result_dir+'/'+args.data_name+'cuda_size_param_results'+'.txt', 'a+') as f:
        f.write(str(max_memory_allocated / (1024 ** 2)) + '\n')
        f.write(str(model_size_mb) + '\n')
        f.write(str(model_size_params / 1e6) + '\n')
        f.write(str(end - start_training) + '\n')
        f.write(str(end - start) + '\n')




    #//

    parameters = list(classifier.parameters())
    # emb = torch.nn.Embedding(net[0], cmd_args.hidden).to(args.device)
    # torch.nn.init.xavier_uniform_(emb.weight)
    # parameters += list(emb.parameters())

    # if args.train_node_embedding:
    #     torch.nn.init.xavier_uniform_(emb.weight)
        # parameters += list(emb.parameters())
    # total_params = sum(p.numel() for param in parameters for p in param)
    total_params = sum(p.numel() for p in classifier.parameters())
    # total_params+=list(emb.parameters())
    print(f'Total number of parameters like scaled is {total_params}')

    ##########################
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


    total_size = 0
    for param_state in optimizer.state.values():
        for key, value in param_state.items():
            if isinstance(value, torch.Tensor):  # Ensure it's a tensor
                total_size += value.numel() * value.element_size()  # Compute size in bytes
    Optimizer_size_mb = total_size / 1024**2
    print(f"Optimizer state size: {Optimizer_size_mb:.2f} MB")  # Convert to MB
    print(f"Model size: {model_size_mb:.2f} MB\n")
    print(f"total size: {Optimizer_size_mb+model_size_mb:.2f} MB\n")

    #####################

    with open(args.result_dir+'/'+args.data_name+'cuda_size_param_results_details'+'.txt', 'a+') as f:
        f.write(f"Max allocated CUDA memory: {max_memory_allocated / (1024 ** 2)} MB" + '\n')
        f.write(f"allocated CUDA memory: {torch.cuda.memory_allocated() / (1024 ** 2)} MB" + '\n')
        f.write(f"Reserved CUDA memory: {torch.cuda.memory_reserved() / (1024 ** 2)} MB" + '\n')
        f.write(f"NUM parameters: {model_size_params} parameters\n")
        f.write(f"NUM parameters: {model_size_params / 1e6:.3f}M parameters\n")
        f.write(f"Model size: {model_size_mb:.2f} MB\n")
        f.write(f'Total number of parameters like scaled is {total_params}\n')
        f.write(f"Time taken for training run: {end - start_training:.2f} seconds" + '\n')
        f.write(f"Time taken for run: {end - start:.2f} seconds" + '\n')
        
        f.write(f"Optimizer state size: {Optimizer_size_mb:.2f} MB\n")  # Convert to MB
        f.write(f"Model size: {model_size_mb:.2f} MB\n")
        f.write(f"total  model_optimizer size: {Optimizer_size_mb+model_size_mb:.2f} MB\n")


    with open(args.result_dir+'/'+args.data_name+'memory_summary'+'.txt', 'a+') as f:
            f.write(torch.cuda.memory_summary())

@timeit()
def train(classifier,cmd_args,args,train_loader,val_loader,test_loader,optimizer):
    
    save_plot=False
    bestscore = None
    bestscoretest = None
    start_training = default_timer()

    for epoch in range(cmd_args.num_epochs):
        if epoch==cmd_args.num_epochs-1: 
            save_plot= True

        classifier.train()
        avg_loss = loop_dataset_gem(args.data_name, args.result_dir,classifier, train_loader,args.seed, optimizer=optimizer, save_plot=save_plot)
        
        if not cmd_args.printAUC:
            avg_loss[2] = 0.0
        # print(('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2], avg_loss[3])))
        print(('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f hit1 %.5f hit3 %.5f hit10 %.5f hit20 %.5f hit30 %.5f hit50 %.5f hit100 %.5f mmr %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2], avg_loss[3],avg_loss[4],avg_loss[5],avg_loss[6],avg_loss[7],avg_loss[8],avg_loss[9],avg_loss[10],avg_loss[11])))
        # with open(args.result_dir+'/acc_train_results'+args.data_name+'.txt', 'a+') as f:
        #     f.write(str(avg_loss[1]) + '\n')
        # with open(args.result_dir+'/loss_train_results'+args.data_name+'.txt', 'a+') as f:
        #     f.write(str(avg_loss[0]) + '\n')


        classifier.eval()
        val_loss = loop_dataset_gem(args.data_name, args.result_dir,classifier, val_loader,args.seed, None, save_plot) 
        test_loss = loop_dataset_gem(args.data_name, args.result_dir,classifier, test_loader,args.seed, None, save_plot) 
        if bestscore is None: 
            bestscore = val_loss #{key: list(val_loss[key]) for key in val_loss} 
            bestscoretest = test_loss 
        # if val_loss[7] > bestscore[7]: #Hit100 
        if val_loss[2] > bestscore[2]: 
            bestscore = val_loss
            bestscoretest = test_loss 

        if not cmd_args.printAUC:
            test_loss[2] = 0.0
        # print(('average test of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f' % (epoch, test_loss[0], test_loss[1], test_loss[2], test_loss[3])))
        print('\033[94maverage val of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f hit1 %.5f hit3 %.5f hit10 %.5f hit20 %.5f hit30 %.5f hit50 %.5f hit100 %.5f mmr %.5f\033[0m' % (epoch, val_loss[0], val_loss[1], val_loss[2], val_loss[3],val_loss[4],val_loss[5],val_loss[6],val_loss[7],val_loss[8], val_loss[9], val_loss[10], val_loss[11]))
        print('\033[94maverage test of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f hit1 %.5f hit3 %.5f hit10 %.5f hit20 %.5f hit30 %.5f hit50 %.5f hit100 %.5f mmr %.5f\033[0m' % (epoch, test_loss[0], test_loss[1], test_loss[2], test_loss[3],test_loss[4],test_loss[5],test_loss[6],test_loss[7],test_loss[8],test_loss[9],test_loss[10],test_loss[11]))

        with open(args.result_dir+'/'+args.data_name+'train_loss_results.txt', 'a+') as f:
            f.write(str(avg_loss[0]) + '\n')
        with open(args.result_dir+'/'+args.data_name+'train_acc_results.txt', 'a+') as f:
            f.write(str(avg_loss[1]) + '\n')
        with open(args.result_dir+'/'+args.data_name+'train_auc_results.txt', 'a+') as f:
            f.write(str(avg_loss[2]) + '\n')
        with open(args.result_dir+'/'+args.data_name+'train_ap_results.txt', 'a+') as f:
            f.write(str(avg_loss[3]) + '\n')
        with open(args.result_dir+'/'+args.data_name+'all_results.txt', 'a+') as f:
            f.write(str(  '\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f hit1 %.5f hit3 %.5f hit10 %.5f hit100 %.5f mmr %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2], avg_loss[3],avg_loss[4],avg_loss[5],avg_loss[6],avg_loss[7],avg_loss[8])) + '\n')
            f.write(str(   '\033[94maverage test of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f hit1 %.5f hit3 %.5f hit10 %.5f hit100 %.5f mmr %.5f\033[0m' % (epoch, test_loss[0], test_loss[1], test_loss[2], test_loss[3],test_loss[4],test_loss[5],test_loss[6],test_loss[7],test_loss[8])) + '\n')


        # if best_auc < test_loss[2]:
        #     best_auc = test_loss[2]
        #     best_auc_acc = test_loss[3]

        # if best_acc < test_loss[3]:
        #     best_acc = test_loss[3]
        #     best_acc_auc = test_loss[2]
    
    return bestscore , bestscoretest

if __name__ == '__main__':
    mp.freeze_support()  # For Windows support in multiprocessing
    main()

        


