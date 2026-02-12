from sklearn import metrics
from sklearn.metrics import average_precision_score
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import torch
import numpy as np



import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle
def save_data_Embedding(all_embeddings,all_targets,data_name, dir_to_save):

    # Variables to save
    data_to_save = {
        'embeddings': all_embeddings,
        'targets': all_targets,
    }

    # Save to a .dat file
    with open(dir_to_save + '/tsne_data_' + data_name + '.dat', 'wb') as f:
        pickle.dump(data_to_save, f)


def draw_TSNE_embeding_nodeclass(dataname, node_features, node_labels, save_path=None):
    # Apply t-SNE to reduce dimensions to 2D
    tsne = TSNE(n_components=2, random_state=42)
    node_embeddings_2d = tsne.fit_transform(node_features)

    # Plot the t-SNE results
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(node_embeddings_2d[:, 0], node_embeddings_2d[:, 1], c=node_labels, cmap="bwr", s=10)
    # scatter = plt.scatter(node_embeddings_2d[:, 0], node_embeddings_2d[:, 1], c=node_labels, cmap="tab10", s=10)
    plt.colorbar(scatter)
    plt.title("t-SNE Visualization of " + dataname + " Dataset")

    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)  # Adjust format and dpi as needed
        print(f"Plot saved to {save_path}")

    # Display the plot
    # plt.show()
    plt.close()  # Close the figure to free memory if not displayed

def evaluate_hits(pos_pred, neg_pred, k_list,data_name):
    evaluator = Evaluator(name='ogbl-collab')
    if data_name=="citation2":
        neg_pred = torch.tensor(neg_pred).view(len(pos_pred),-1)
        neg_pred = neg_pred[:,1]
        pos_pred = torch.tensor(pos_pred).view(-1)
        # print("hhhhhhhhhhhhhhhhhhhhhhhh")
        # print(neg_pred.shape)
        # print(pos_pred.shape)

    results = {}
    for K in k_list:
        evaluator.K = K

        hits = evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })[f'hits@{K}']
        hits = round(hits, 4)
        results[f'Hits@{K}'] = hits
    return results




# def hits_at_n_ogb(pos_test_pred, neg_test_pred, k_list):
#     # pos_test_pred = scores[targets == 1]
#     # neg_test_pred = scores[targets == 0]
#     result_hit_test = evaluate_hits(pos_test_pred, neg_test_pred, k_list)
#     return result_hit_test

def evaluate_mrr_scaled(pos_test_pred, neg_test_pred, dataname):
    try:
        if dataname=="citation2":
            print("citation2")
            evaluator_mrr = Evaluator(name='ogbl-citation2')
            pos_test_pred = torch.tensor(pos_test_pred)
            neg_test_pred = torch.tensor(neg_test_pred)
            # print(pos_test_pred.shape)
            # print(neg_test_pred.shape)
            pos_test_pred = pos_test_pred.view(-1)
            neg_test_pred = neg_test_pred.view(-1, 1000)
            # print(pos_test_pred.shape)
            # print(neg_test_pred.shape)
            # neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
            # neg_test_pred = neg_test_pred.repeat(1000, 1)
            # neg_test_pred = neg_test_pred.repeat(pos_test_pred.size(0), 1)
            # print(neg_test_pred) 
            test_mrr = evaluator_mrr.eval({
                'y_pred_pos': pos_test_pred,
                'y_pred_neg': neg_test_pred,
            })['mrr_list'].mean().item()
        else: 
            evaluator_mrr = Evaluator(name='ogbl-citation2')
            pos_test_pred = torch.tensor(pos_test_pred)
            neg_test_pred = torch.tensor(neg_test_pred)
            # print(pos_test_pred.shape)
            # print(neg_test_pred.shape)

            # print(sum(pos_test_pred)) 
            # print(max(pos_test_pred)) 
            # print(min(pos_test_pred)) 
            # print((neg_test_pred[neg_test_pred>0])) 
            # print(max(neg_test_pred)) 
            # print(min(neg_test_pred)) 

            # print(neg_test_pred) 
            # neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
            # neg_test_pred = neg_test_pred.repeat(1000, 1)
            # print(neg_test_pred)
            neg_test_pred = neg_test_pred.repeat(pos_test_pred.size(0), 1)
            # neg_test_pred = neg_test_pred.repeat(pos_test_pred.size(0), 2)
            # print(pos_test_pred.shape)
            # print(neg_test_pred.shape)
            # print(sum(neg_test_pred))

            # print(neg_test_pred) 
            test_mrr = evaluator_mrr.eval({
                'y_pred_pos': pos_test_pred,
                'y_pred_neg': neg_test_pred,
            })['mrr_list'].mean().item()
    except:
        test_mrr=-1
    return test_mrr
'''
def evaluate_mrr_scaled(scores, targets):
    try:
        evaluator_mrr = Evaluator(name='ogbl-citation2')
        pos_test_pred = torch.tensor(scores[targets == 1])
        neg_test_pred = torch.tensor(scores[targets == 0])
        # print(sum(pos_test_pred)) 
        # print(max(pos_test_pred)) 
        # print(min(pos_test_pred)) 
        # print((neg_test_pred[neg_test_pred>0])) 
        # print(max(neg_test_pred)) 
        # print(min(neg_test_pred)) 

        # print(neg_test_pred) 
        # neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
        # neg_test_pred = neg_test_pred.repeat(1000, 1)
        neg_test_pred = neg_test_pred.repeat(pos_test_pred.size(0), 1)
        # print(neg_test_pred) 
        test_mrr = evaluator_mrr.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })['mrr_list'].mean().item()
    except:
        test_mrr=-1
    return test_mrr
'''
