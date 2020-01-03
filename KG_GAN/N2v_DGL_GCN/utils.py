import dgl
import pickle as pkl
import numpy as np
import os
import torch
import sys
import json
import random
import torch.nn.functional as F
from N2v_DGL_GCN.models import GCN
import json
import os.path as osp
import shutil

def ensure_path(path):
    if osp.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.mkdir(path)
    else:
        os.mkdir(path)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def encode_onehot(labels):
    classes = set(labels)
    print(classes)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def load_index(path):
    seen_corresp_file = os.path.join(path, 'g_seen_corresp.json')
    with open(seen_corresp_file) as fp:
        seen_corresp = json.load(fp)
    unseen_corresp_file = os.path.join(path, 'g_unseen_corresp.json')
    with open(unseen_corresp_file) as fp:
        unseen_corresp = json.load(fp)
    seen_corresp.extend(unseen_corresp)
    return seen_corresp

def load_data(path, node_file, edge_file):

    print('Loading dataset...')
    with open(os.path.join(path, edge_file), 'rb') as f:
        if sys.version_info > (3, 0):
            edges = pkl.load(f, encoding='latin1')
        else:
            edges = pkl.load(f)

    node_idx_features_labels = np.genfromtxt(os.path.join(path, node_file), delimiter=",",
                                             dtype=np.dtype(str))
    features = np.array(node_idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(node_idx_features_labels[:, -1])

    # build graph
    nodes = np.array(node_idx_features_labels[:, 0], dtype=np.str)
    print(nodes.shape)
    g = dgl.DGLGraph()
    # add  nodes into the graph; nodes are labeled from 0~len(nodes)
    g.add_nodes(nodes.shape[0])

    nodes_map = {j: i for i, j in enumerate(nodes)}
    edges = np.array(list(map(nodes_map.get, edges.flatten())),
                     dtype=np.int32).reshape(edges.shape)
    edge_list = edges.tolist()

    # add edges two lists of nodes: src and dst
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    # edges are directional in DGL; make them bi-directional
    g.add_edges(dst, src)



    # load index and features
    train_index = load_index(path)
    labeled_nodes = torch.LongTensor(np.array(train_index))
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])


    return g, features, labels, labeled_nodes

def readPKL(filename):
    with open(filename, 'rb') as f:
        if sys.version_info > (3, 0):
            items = pkl.load(f, encoding='latin1')
        else:
            items = pkl.load(f)
    return items

def load_Hete_Data(path):
    print('Loading dataset...')


    nodes_embed = readPKL(os.path.join(path, 'g_embed.pkl'))
    edges_c = readPKL(os.path.join(path, 'edges-c-c.pkl'))

    edges_a = readPKL(os.path.join(path, 'edges-c-a.pkl'))


    with open(os.path.join(path, 'g_cls_nodes.json')) as fp:
        cls_nodes = json.load(fp)
    with open(os.path.join(path, 'g_att_nodes.json')) as fp:
        att_nodes = json.load(fp)

    cls_nodes_map = {j: i for i, j in enumerate(cls_nodes)}
    att_nodes_map = {j: i for i, j in enumerate(att_nodes)}
    cls_att_map = cls_nodes_map
    cls_att_map.update(att_nodes_map)
    new_edges_c = [(cls_nodes_map[edge[0]], cls_nodes_map[edge[1]]) for edge in edges_c]
    new_edges_a = [(cls_att_map[edge[0]], cls_att_map[edge[1]]) for edge in edges_a]

    new_edges_c_inverse = [(edge[1], edge[0]) for edge in new_edges_c]
    new_edges_a_inverse = [(edge[1], edge[0]) for edge in new_edges_a]


    g_edge = dgl.heterograph(
        {('cls', 'adj', 'cls'): new_edges_c,
         ('cls', 'has', 'att'): new_edges_a,
         ('cls', 'iadj', 'cls'): new_edges_c_inverse,
         ('att', 'ownedby', 'cls'): new_edges_a_inverse})

    print('Node types:', g_edge.ntypes)
    print('Edge types:', g_edge.etypes)

    print("cls nodes number", g_edge.number_of_nodes('cls'))
    print("att nodes number", g_edge.number_of_nodes('att'))

    # read labels
    with open(os.path.join(path, 'g_labels.json')) as fp:
        labels = json.load(fp)
    labels = encode_onehot(labels)

    train_index = load_index(path)

    labeled_nodes = torch.LongTensor(np.array(train_index))
    # nodes_embed = torch.FloatTensor(nodes_embed)
    labels = torch.LongTensor(np.where(labels)[1])

    return g_edge, nodes_embed, labels, labeled_nodes
