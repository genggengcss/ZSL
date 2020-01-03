import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import os
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
    idx = np.array(node_idx_features_labels[:, 0], dtype=np.str)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = edges


    print("edge...")
    print(edges_unordered)


    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    train_index = load_index(path)

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(np.array(train_index))


    return adj, features, labels, idx_train



def load_data2(path="cora/", dataset="cora"):


    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    print(idx_features_labels[0][-1])
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    print("edge...")
    print(edges_unordered)
    print(type(edges_unordered))
    print(type(edges_unordered[0]))

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
