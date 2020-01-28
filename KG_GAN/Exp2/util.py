# import h5py
import numpy as np
import scipy.io as scio
import torch
from sklearn import preprocessing
from sklearn.cluster import KMeans
import os
import time
import pickle as pkl
import json
import networkx as nx
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import sys


import scipy.sparse as sp
import torch
import sys
import os

def GetNowTime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def load_json(path):
    with open(path) as fp:
        corresp = json.load(fp)
    return corresp

def load_graph_data(graph_file, input_fea_file):
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


    features = np.array(features)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)





def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())  # 19832
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i
    # print(mapped_label)
    return mapped_label


class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        f = open(self.filename + '.log', "a")
        f.close()

    def write(self, message):
        f = open(self.filename + '.log', "a")
        f.write(message)
        f.close()


class DATA_LOADER(object):
    def __init__(self, args):

        if args.DATASET == 'ImageNet':
            self.read_imagenet(args)

        self.index_in_epoch = 0
        self.epochs_completed = 0

        self.feature_dim = self.train_feature.shape[1]  # 2048
        # self.sem_dim = self.semantic.shape[1]  # 500
        # self.text_dim = self.sem_dim  # 500
        self.train_cls_num = self.seenclasses.shape[0]
        self.test_cls_num = self.unseenclasses.shape[0]



    def readTxt(self, file_name):
        class_list = list()
        wnids = open(file_name, 'rU')
        try:
            for line in wnids:
                class_list.append(line[:-1])
        finally:
            wnids.close()
        return class_list

    def ID2Index(self, wnids, class_file):
        class_wnids = self.readTxt(class_file)
        index_list = list()
        for wnid in class_wnids:
            idx = wnids.index(wnid)
            index_list.append(idx+1)
        return index_list, class_wnids

    def readFeatures(self, args, folder, index_set, type, nsample=None):
        fea_set = list()
        label_set = list()
        for idx in index_set:
            file = os.path.join(args.DATADIR, args.DATASET, folder, str(idx)+'.mat')
            feature = np.array(scio.loadmat(file)['features'])
            if type == 'seen':
                if nsample and feature.shape[0] > nsample:
                    feature = feature[:nsample]
            if type == 'unseen':
                if nsample and feature.shape[0] > nsample:
                    feature = feature[:nsample]

            label = np.array((idx-1), dtype=int)
            label = label.repeat(feature.shape[0])
            fea_set.append(feature)
            label_set.append(label)
        fea_set = tuple(fea_set)
        label_set = tuple(label_set)
        features = np.vstack(fea_set)
        labels = np.hstack(label_set)
        return features, labels

    def read_imagenet(self, args):
        # split.mat : wnids, words
        matcontent = scio.loadmat(os.path.join(args.DATADIR, args.DATASET, args.SplitFile))
        wnids = matcontent['allwnids'].squeeze().tolist()
        self.wnids = wnids[:2549]
        words = matcontent['allwords'].squeeze()[:2549]
        seen_index, seen = self.ID2Index(wnids, os.path.join(args.DATADIR, args.DATASET, 'KG-GAN', args.ExpName, 'seen.txt'))
        unseen_index, unseen = self.ID2Index(wnids, os.path.join(args.DATADIR, args.DATASET, 'KG-GAN', args.ExpName, 'unseen.txt'))
        # print(seen_index)
        self.seen = seen
        self.unseen = unseen


        # load graph data
        self.cls_seen_corresp = load_json(args.cls_seen_corresp)
        self.cls_unseen_corresp = load_json(args.cls_unseen_corresp)
        self.att_seen_corresp = load_json(args.att_seen_corresp)
        self.att_unseen_corresp = load_json(args.att_unseen_corresp)
        self.cls_nodes = load_json(args.cls_nodes)
        self.att_nodes = load_json(args.att_nodes)

        cls_adj, cls_feat = load_graph_data(args.cls_graph, args.cls_feat)
        att_adj, att_feat = load_graph_data(args.att_graph, args.att_feat)

        self.cls_feat = torch.from_numpy(cls_feat).float()
        self.att_feat = torch.from_numpy(att_feat).float()

        self.cls_adj = preprocess_graph(cls_adj)
        self.att_adj = preprocess_graph(att_adj)

        # self.cls_adj = torch.from_numpy(att_adj).float()
        # self.att_adj = torch.from_numpy(att_adj).float()




        # read seen features
        seen_features, seen_labels = self.readFeatures(args, args.SeenFeaFile, seen_index, 'seen')
        print("seen features shape:", seen_features.shape)
        # print("seen labels:", seen_labels)
        seen_features1, seen_labels1 = self.readFeatures(args, args.SeenFeaFile, seen_index, 'seen', args.SeenSynNum)
        print("seen features shape:", seen_features1.shape)
        # print("seen labels:", seen_labels1)

        # read unseen features for testing
        unseen_features, unseen_labels = self.readFeatures(args, args.UnseenFeaFile, unseen_index, 'unseen', args.Unseen_NSample)
        print("unseen features shape:", unseen_features.shape)
        # read seen features for testing
        seen_features_test, seen_labels_test = self.readFeatures(args, args.SeenTestFeaFile, seen_index, 'seen', args.Unseen_NSample)
        print("seen features shape:", seen_features_test.shape)
        # print("seen labels:", seen_labels_test)

        if args.PreProcess:
            print('MinMaxScaler PreProcessing...')
            scaler = preprocessing.MinMaxScaler()

            seen_features = scaler.fit_transform(seen_features)
            seen_features_test = scaler.transform(seen_features_test)
            unseen_features = scaler.transform(unseen_features)


        self.train_feature = torch.from_numpy(seen_features).float()
        self.train_label = torch.from_numpy(seen_labels).long()
        self.train_feature1 = torch.from_numpy(seen_features1).float()
        self.train_label1 = torch.from_numpy(seen_labels1).long()
        self.test_unseen_feature = torch.from_numpy(unseen_features).float()
        self.test_unseen_label = torch.from_numpy(unseen_labels).long()
        self.test_seen_feature = torch.from_numpy(seen_features_test).float()
        self.test_seen_label = torch.from_numpy(seen_labels_test).long()

        # if args.SemEmbed == 'w2v':
        #     # w2v.mat : word embedding
        #     matcontent = scio.loadmat(os.path.join(args.DATADIR, args.DATASET, args.SemFile))
        #     w2v = matcontent['w2v'][:2549]  # nodes of 1k+2hops
        #     print("semantic embedding shape:", w2v.shape)
        #     self.semantic = torch.from_numpy(w2v).float()
        # if args.SemEmbed == 'n2v':
        #     # n2v.mat: node embedding
        #     matcontent = scio.loadmat(os.path.join(args.DATADIR, args.DATASET, args.SemFile))
        #     n2v = matcontent['n2v']
        #     print("semantic embedding shape:", n2v.shape)
        #     self.semantic = torch.from_numpy(n2v).float()



        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        # print("seen classes:", self.seenclasses)
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        # print("unseen classes:", self.unseenclasses)
        # load class name
        self.unseennames = list()
        unseennames = words[self.unseenclasses.numpy()]
        for i in range(len(unseennames)):
            name = unseennames[i][0]
            name = name.split(',')[0]
            self.unseennames.append(name)


        self.ntrain = self.train_feature.size()[0]  # number of training samples
        self.ntrain_class = self.seenclasses.size(0)  # number of seen classes
        self.ntest_class = self.unseenclasses.size(0)  # number of unseen classes
        self.train_class = self.seenclasses.clone()  # copy
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()

        self.train_mapped_label = map_label(self.train_label, self.seenclasses)

        # self.train_sem = self.semantic[self.seenclasses]
        # self.test_sem = self.semantic[self.unseenclasses]
        self.train_cls_num = self.ntrain_class
        self.test_cls_num = self.ntest_class



    # def next_batch_one_class(self, batch_size):
    #     if self.index_in_epoch == self.ntrain_class:
    #         self.index_in_epoch = 0
    #         perm = torch.randperm(self.ntrain_class)
    #         self.train_class[perm] = self.train_class[perm]
    #
    #     iclass = self.train_class[self.index_in_epoch]
    #     idx = self.train_label.eq(iclass).nonzero().squeeze()
    #     perm = torch.randperm(idx.size(0))
    #     idx = idx[perm]
    #     iclass_feature = self.train_feature[idx]
    #     iclass_label = self.train_label[idx]
    #     self.index_in_epoch += 1
    #     return iclass_feature[0:batch_size], iclass_label[0:batch_size], self.semantic[iclass_label[0:batch_size]]

    def next_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        # batch_sem = self.semantic[batch_label]
        # return batch_feature, batch_label, batch_sem
        return batch_feature, batch_label

    # # select batch samples by randomly drawing batch_size classes
    # def next_batch_uniform_class(self, batch_size):
    #     batch_class = torch.LongTensor(batch_size)
    #     for i in range(batch_size):
    #         idx = torch.randperm(self.ntrain_class)[0]
    #         batch_class[i] = self.train_class[idx]
    #
    #     batch_feature = torch.FloatTensor(batch_size, self.train_feature.size(1))
    #     batch_label = torch.LongTensor(batch_size)
    #     batch_sem = torch.FloatTensor(batch_size, self.semantic.size(1))
    #     for i in range(batch_size):
    #         iclass = batch_class[i]
    #         idx_iclass = self.train_label.eq(iclass).nonzero().squeeze()
    #         idx_in_iclass = torch.randperm(idx_iclass.size(0))[0]
    #         idx_file = idx_iclass[idx_in_iclass]
    #         batch_feature[i] = self.train_feature[idx_file]
    #         batch_label[i] = self.train_label[idx_file]
    #         batch_sem[i] = self.semantic[batch_label[i]]
    #     return batch_feature, batch_label, batch_sem
