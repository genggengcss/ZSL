from __future__ import division
from __future__ import print_function
import random
import argparse
import time
import pickle as pkl
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
import os
import json
from N2v_GAE.model import GCNModelVAE
from N2v_GAE.optimizer import loss_function
from N2v_GAE.utils import load_data_att, mask_test_edges, preprocess_graph, get_roc_score
import os.path as osp
import shutil



'''
using the graph and corresponding w2v to train GAE, get node embedding (n2v)
'''
DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet/KG-GAN'
Exp_NAME = 'Exp15'
graphf = 'att/graph.pkl'
# input_embed = 'g_embed.mat'
input_embed = 'att/g_embed.pkl'
save_path = os.path.join(DATA_DIR, Exp_NAME, 'att/embed')
parser = argparse.ArgumentParser()
# parser.add_argument('--seed', type=int, default=8220, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=150, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=50, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--manual_seed', type=int, help='set the random seed')


args = parser.parse_args()

if args.manual_seed is None:
    ManualSeed = random.randint(1, 10000)
else:
    ManualSeed = args.manual_seed
print("Random Seed: ", ManualSeed)

# set random seed
random.seed(ManualSeed)
torch.manual_seed(ManualSeed)


def ensure_path(path):
    if osp.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.mkdir(path)
    else:
        os.mkdir(path)

ensure_path(save_path)


def gae_for(args):
    graph_file = os.path.join(DATA_DIR, Exp_NAME, graphf)
    input_fea_file = os.path.join(DATA_DIR, Exp_NAME, input_embed)
    adj, features = load_data_att(graph_file, input_fea_file)
    n_nodes, feat_dim = features.shape
    print("nodes, fea dim:", n_nodes, feat_dim)

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    # adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    # adj = adj_train
    adj = adj
    adj_train = adj

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    # pos_weight = torch.FloatTensor(pos_weight)
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    hidden_emb = None
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        recovered, mu, logvar = model(features, adj_norm)
        loss = loss_function(preds=recovered, labels=adj_label,
                             mu=mu, logvar=logvar, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight)
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        hidden_emb = mu.data.numpy()


    #     roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)
    #
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),

              "time=", "{:.5f}".format(time.time() - t)
              )
    #
        if (epoch+1) >= 500 and (epoch+1) % 50 == 0:
            # roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
            # print('Test ROC score: ' + str(roc_score))
            # print('Test AP score: ' + str(ap_score))
            # save well-trained embedding


            filename = str(epoch+1)+'.pkl'
            save_file = os.path.join(save_path, filename)
            # seen_embed = hidden_emb[seen_corresp]
            # unseen_embed = hidden_emb[unseen_corresp]
            # embed = np.vstack((seen_embed, unseen_embed))
            # print("embed shape:", embed.shape)
            with open(save_file, 'wb') as fp:
                pkl.dump(hidden_emb, fp)
    #
    # print("Optimization Finished!")
    #
    # roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    # print('Test ROC score: ' + str(roc_score))
    # print('Test AP score: ' + str(ap_score))


if __name__ == '__main__':

    # seen_corresp, unseen_corresp = load_corresp()

    gae_for(args)