# import h5py
import numpy as np
import scipy.io as scio
import torch
from sklearn import preprocessing
import sys
from sklearn.cluster import KMeans
import config
import os
import time
import pickle
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
        else:
            self.read_dataset(args)
        self.index_in_epoch = 0
        self.epochs_completed = 0

        self.feature_dim = self.train_feature.shape[1]  # 2048
        self.sem_dim = self.semantic.shape[1]  # 500
        self.text_dim = self.sem_dim  # 500
        self.train_cls_num = self.seenclasses.shape[0]
        self.test_cls_num = self.unseenclasses.shape[0]
        self.tr_cls_centroid = np.zeros([self.seenclasses.shape[0], self.feature_dim], np.float32)  # (249, 2048)
        # torch.nonzero: return the index of non-zero value
        # function: average the sample feature of seen classes
        # for i in range(self.seenclasses.shape[0]):
        #     self.tr_cls_centroid[i] = np.mean(
        #         self.train_feature[torch.nonzero(self.train_mapped_label == i), :].numpy(), axis=0)
        # cluster_path = os.path.join(args.DATADIR, args.DATASET, 'KG-GAN', args.ExpName, args.Cluster_Save_Dir)
        # if os.path.exists(cluster_path):
        #     path = os.path.join(cluster_path, 'real_proto.pickle')
        #     with open(path, 'rb') as file:
        #         real_proto = pickle.load(file)
        #     self.real_proto = real_proto
        # else:
        #     os.mkdir(cluster_path)
        #     n_cluster = args.NClusters  # 3
        #     real_proto = torch.zeros(n_cluster * self.train_cls_num, self.feature_dim)  # (40*3, 2048)
        #     for i in range(self.train_cls_num):
        #         sample_idx = (self.train_mapped_label == i).nonzero().squeeze()  # i-th seen samples index
        #         # torch.numel(): return the number of sample_idx's elements
        #         if sample_idx.numel() == 0:
        #             real_proto[n_cluster * i: n_cluster * (i + 1)] = torch.zeros(n_cluster, self.feature_dim)
        #         else:
        #             real_sample_cls = self.train_feature[sample_idx, :]  # i-th seen sample's all features
        #             print(i, " begin clustering ... :", GetNowTime())
        #             y_pred = KMeans(n_clusters=n_cluster, random_state=3).fit_predict(real_sample_cls)
        #             print(i, "  end  clustering ... :", GetNowTime())
        #             for j in range(n_cluster):
        #                 real_proto[n_cluster * i + j] = torch.from_numpy(
        #                     real_sample_cls[torch.nonzero(torch.from_numpy(y_pred) == j), :].mean(dim=0).cpu().numpy())
        #     path = os.path.join(cluster_path, 'real_proto.pickle')
        #     file = open(path, 'wb')
        #     pickle.dump(real_proto, file)
        #     file.close()
        #
        #     self.real_proto = real_proto  # (249*5, 2048)
        # print("real proto shape:", self.real_proto.shape)






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
        return index_list

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
        words = matcontent['allwords'].squeeze()[:2549]
        seen_index = self.ID2Index(wnids, os.path.join(args.DATADIR, args.DATASET, 'KG-GAN', args.ExpName, 'seen.txt'))
        unseen_index = self.ID2Index(wnids, os.path.join(args.DATADIR, args.DATASET, 'KG-GAN', args.ExpName, 'unseen.txt'))
        # print(seen_index)

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

        if args.SemEmbed == 'w2v':
            # w2v.mat : word embedding
            matcontent = scio.loadmat(os.path.join(args.DATADIR, args.DATASET, args.SemFile))
            w2v = matcontent['w2v'][:2549]  # nodes of 1k+2hops
            print("semantic embedding shape:", w2v.shape)
            self.semantic = torch.from_numpy(w2v).float()
        if args.SemEmbed == 'n2v':
            # n2v.mat: node embedding
            matcontent = scio.loadmat(os.path.join(args.DATADIR, args.DATASET, args.SemFile))
            n2v = matcontent['n2v']
            print("semantic embedding shape:", n2v.shape)
            self.semantic = torch.from_numpy(n2v).float()



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

        self.train_sem = self.semantic[self.seenclasses]
        self.test_sem = self.semantic[self.unseenclasses]
        self.train_cls_num = self.ntrain_class
        self.test_cls_num = self.ntest_class

    def read_dataset(self, args):

        matcontent = scio.loadmat(os.path.join(args.DATADIR, args.DATASET, args.FeaFile))

        feature = matcontent['features'].T  # (30475, 2048), i.e., (sample_num, feature_dim)
        # label index starts from 0, matlab starts from 1
        label = matcontent['labels'].astype(int).squeeze() - 1  # (30475), sample_num
        matcontent = scio.loadmat(os.path.join(args.DATADIR, args.DATASET, args.SemFile))
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1  # (19832), train_sample num
        # train_loc = matcontent['train_loc'].squeeze() - 1
        # val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
        # shared neighbors as additional attributes -> new semantic file
        new_M_content = scio.loadmat(os.path.join(args.DATADIR, args.DATASET, 'class-attribute-neighbor.mat'))
        self.semantic = torch.from_numpy(new_M_content['attribute_neighbor_M']).float()
        print('attribute shape:', self.semantic.shape)  # (50, 103)
        # self.attribute = torch.from_numpy(matcontent['att'].T).float()
        # print('attrute shape:', self.attribute.shape)

        if not args.Cross_Validation:
            if args.PreProcess:
                if args.Standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()

                _train_feature = scaler.fit_transform(feature[trainval_loc])
                _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
                self.train_feature = torch.from_numpy(_train_feature).float()  # get data from numpy, and share memory
                mx = self.train_feature.max()
                self.train_feature.mul_(1 / mx)  # in-place operation: train_feature = torch.mul(train_feature,(1/mx))
                self.train_label = torch.from_numpy(label[trainval_loc]).long()  # seen class index when training
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1 / mx)
                self.test_unseen_label = torch.from_numpy(
                    label[test_unseen_loc]).long()  # unseen class index when testing
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
                self.test_seen_feature.mul_(1 / mx)
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()  # seen class index when testing
            else:
                self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
                self.train_label = torch.from_numpy(label[trainval_loc]).long()
                self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float()
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
        else:
            pass
            # self.train_feature = torch.from_numpy(feature[train_loc]).float()
            # self.train_label = torch.from_numpy(label[train_loc]).long()
            # self.test_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
            # self.test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long()

        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))  # 40
        print("seen classes:", self.seenclasses)
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        print("unseen classes:", self.unseenclasses)
        self.ntrain = self.train_feature.size()[0]  # number of training samples
        self.ntrain_class = self.seenclasses.size(0)  # number of seen classes
        self.ntest_class = self.unseenclasses.size(0)  # number of unseen classes
        self.train_class = self.seenclasses.clone()  # copy
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()

        self.train_mapped_label = map_label(self.train_label, self.seenclasses)  # filter the unseen/test label, set to 0

        self.train_sem = self.semantic[self.seenclasses].numpy()
        self.test_sem = self.semantic[self.unseenclasses].numpy()
        self.train_cls_num = self.ntrain_class
        self.test_cls_num = self.ntest_class

    def next_batch_one_class(self, batch_size):
        if self.index_in_epoch == self.ntrain_class:
            self.index_in_epoch = 0
            perm = torch.randperm(self.ntrain_class)
            self.train_class[perm] = self.train_class[perm]

        iclass = self.train_class[self.index_in_epoch]
        idx = self.train_label.eq(iclass).nonzero().squeeze()
        perm = torch.randperm(idx.size(0))
        idx = idx[perm]
        iclass_feature = self.train_feature[idx]
        iclass_label = self.train_label[idx]
        self.index_in_epoch += 1
        return iclass_feature[0:batch_size], iclass_label[0:batch_size], self.semantic[iclass_label[0:batch_size]]

    def next_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_sem = self.semantic[batch_label]
        return batch_feature, batch_label, batch_sem

    # select batch samples by randomly drawing batch_size classes
    def next_batch_uniform_class(self, batch_size):
        batch_class = torch.LongTensor(batch_size)
        for i in range(batch_size):
            idx = torch.randperm(self.ntrain_class)[0]
            batch_class[i] = self.train_class[idx]

        batch_feature = torch.FloatTensor(batch_size, self.train_feature.size(1))
        batch_label = torch.LongTensor(batch_size)
        batch_sem = torch.FloatTensor(batch_size, self.semantic.size(1))
        for i in range(batch_size):
            iclass = batch_class[i]
            idx_iclass = self.train_label.eq(iclass).nonzero().squeeze()
            idx_in_iclass = torch.randperm(idx_iclass.size(0))[0]
            idx_file = idx_iclass[idx_in_iclass]
            batch_feature[i] = self.train_feature[idx_file]
            batch_label[i] = self.train_label[idx_file]
            batch_sem[i] = self.semantic[batch_label[i]]
        return batch_feature, batch_label, batch_sem
