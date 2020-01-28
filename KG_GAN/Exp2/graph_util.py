import pickle as pkl

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import sys
import scipy.io as scio
import os.path as osp
import shutil
import os


def load_data_att(graph_file, input_fea_file):
    # load the data: input, graph (adj)

    with open(graph_file, 'rb') as f:
        if sys.version_info > (3, 0):
            graph = pkl.load(f, encoding='latin1')
        else:
            graph = pkl.load(f)

    with open(input_fea_file, 'rb') as f:
        if sys.version_info > (3, 0):
            features = pkl.load(f, encoding='latin1')
        else:
            features = pkl.load(f)


    features = torch.FloatTensor(np.array(features))
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features