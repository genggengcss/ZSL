import numpy as np
import scipy.io as sio
from termcolor import cprint
from sklearn import preprocessing
import torch
import scipy.io as scio
import os

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i

    return mapped_label

class DATA_LOADER(object):
    def __init__(self, opt):
        if opt.matdataset:
            if opt.dataset == 'ImageNet':
                self.read_matimagenet(opt)
            # else:
                # self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.feature_dim = self.train_feature.shape[1]
        self.sem_dim = self.semantic.shape[1]
        self.text_dim = self.sem_dim
        self.train_cls_num = self.seenclasses.shape[0]
        self.test_cls_num = self.unseenclasses.shape[0]
        self.tr_cls_centroid = np.zeros([self.seenclasses.shape[0], self.feature_dim], np.float32)  # .astype(np.float32)
        for i in range(self.seenclasses.shape[0]):
            self.tr_cls_centroid[i] = np.mean(self.train_feature[self.train_label == i].numpy(), axis=0)

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

    def readFeatures(self, opt, folder, index_set, type, nsample=None):
        fea_set = list()
        label_set = list()
        for idx in index_set:
            file = os.path.join(opt.dataroot, opt.matdataset, folder, str(idx)+'.mat')
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

    def read_matimagenet(self, opt):
        matcontent = scio.loadmat(os.path.join(opt.dataroot, opt.matdataset, 'split.mat'))

        wnids = matcontent['allwnids'].squeeze().tolist()
        words = matcontent['allwords'].squeeze()[:2549]
        seen_index = self.ID2Index(wnids, os.path.join(opt.dataroot, opt.matdataset, 'KG-GAN', opt.exp_name, 'seen.txt'))
        unseen_index = self.ID2Index(wnids, os.path.join(opt.dataroot, opt.matdataset, 'KG-GAN', opt.exp_name, 'unseen.txt'))

        seen_features, seen_labels = self.readFeatures(opt, opt.SeenFeaFile, seen_index, 'seen')
        print("seen features shape:", seen_features.shape)
        seen_features1, seen_labels1 = self.readFeatures(opt, opt.SeenFeaFile, seen_index, 'seen', opt.nSample)
        # print("seen labels:", seen_labels1)

        # read unseen features for testing
        unseen_features, unseen_labels = self.readFeatures(opt, opt.UnseenFeaFile, unseen_index, 'unseen')
        print("unseen features shape:", unseen_features.shape)
        # read seen features for testing
        seen_features_test, seen_labels_test = self.readFeatures(opt, opt.SeenTestFeaFile, seen_index, 'seen')
        print("seen features shape:", seen_features_test.shape)

        if opt.preprocessing:
            print('MinMaxScaler PreProcessing...')
            scaler = preprocessing.MinMaxScaler()

            seen_features = scaler.fit_transform(seen_features)
            seen_features_test = scaler.transform(seen_features_test)
            unseen_features = scaler.transform(unseen_features)


        self.train_feature = torch.from_numpy(seen_features).float()
        self.train_label = torch.from_numpy(seen_labels).long()
        # self.train_feature1 = torch.from_numpy(seen_features1).float()
        # self.train_label1 = torch.from_numpy(seen_labels1).long()
        self.test_unseen_feature = torch.from_numpy(unseen_features).float()
        self.test_unseen_label = torch.from_numpy(unseen_labels).long()
        self.test_seen_feature = torch.from_numpy(seen_features_test).float()
        self.test_seen_label = torch.from_numpy(seen_labels_test).long()


        # matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".mat")
        # self.attribute = torch.from_numpy(matcontent['w2v']).float()

        if opt.class_embedding == 'w2v':
            # w2v.mat : word embedding
            matcontent = scio.loadmat(os.path.join(opt.dataroot, opt.matdataset, 'w2v.mat'))
            w2v = matcontent['w2v'][:2549]  # nodes of 1k+2hops
            print("semantic embedding shape:", w2v.shape)
            self.semantic = torch.from_numpy(w2v).float()
        if opt.class_embedding == 'g2v':
            # n2v.mat: node embedding
            matcontent = scio.loadmat(os.path.join(opt.dataroot, opt.matdataset, 'KG-GAN', opt.exp_name, 'g2v-att.mat'))
            n2v = matcontent['n2v']
            print("semantic embedding shape:", n2v.shape)
            self.semantic = torch.from_numpy(n2v).float()

        self.ntrain = self.train_feature.size()[0]


        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.train_class = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)

        self.train_label = map_label(self.train_label, self.seenclasses)
        self.test_unseen_label = map_label(self.test_unseen_label, self.unseenclasses)
        self.test_seen_label = map_label(self.test_seen_label, self.seenclasses)
        self.train_sem = self.semantic[self.seenclasses].numpy()
        self.test_sem = self.semantic[self.unseenclasses].numpy()



    # def read_matdataset(self, opt):
    #     matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
    #     feature = matcontent['features'].T
    #     label = matcontent['labels'].astype(int).squeeze() - 1
    #     matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
    #     # numpy array index starts from 0, matlab starts from 1
    #     trainval_loc = matcontent['trainval_loc'].squeeze() - 1
    #     train_loc = matcontent['train_loc'].squeeze() - 1
    #     val_unseen_loc = matcontent['val_loc'].squeeze() - 1
    #     test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
    #     test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
    #
    #     self.attribute = torch.from_numpy(matcontent['att'].T).float()
    #     if not opt.validation:
    #         if opt.preprocessing:
    #             if opt.standardization:
    #                 print('standardization...')
    #                 scaler = preprocessing.StandardScaler()
    #             else:
    #                 scaler = preprocessing.MinMaxScaler()
    #
    #             _train_feature = scaler.fit_transform(feature[trainval_loc])
    #             _test_seen_feature = scaler.transform(feature[test_seen_loc])
    #             _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
    #             self.train_feature = torch.from_numpy(_train_feature).float()
    #             mx = self.train_feature.max()
    #             self.train_feature.mul_(1 / mx)
    #             self.train_label = torch.from_numpy(label[trainval_loc]).long()
    #             self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
    #             self.test_unseen_feature.mul_(1 / mx)
    #             self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
    #             self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
    #             self.test_seen_feature.mul_(1 / mx)
    #             self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
    #         else:
    #             self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
    #             self.train_label = torch.from_numpy(label[trainval_loc]).long()
    #             self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
    #             self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
    #             self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float()
    #             self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
    #     else:
    #         self.train_feature = torch.from_numpy(feature[train_loc]).float()
    #         self.train_label = torch.from_numpy(label[train_loc]).long()
    #         self.test_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
    #         self.test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long()
    #
    #     self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
    #     self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
    #     self.ntrain = self.train_feature.size()[0]
    #     self.ntrain_class = self.seenclasses.size(0)
    #     self.ntest_class = self.unseenclasses.size(0)
    #     self.train_class = self.seenclasses.clone()
    #     self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()
    #
    #     self.train_label = map_label(self.train_label, self.seenclasses)
    #     self.test_unseen_label = map_label(self.test_unseen_label, self.unseenclasses)
    #     self.test_seen_label = map_label(self.test_seen_label, self.seenclasses)
    #     self.train_att = self.attribute[self.seenclasses].numpy()
    #     self.test_att  = self.attribute[self.unseenclasses].numpy()
    #     self.train_cls_num = 150
    #     self.test_cls_num  = 50


class FeatDataLayer(object):   # by Ethan provide the ROI feature data for ZSL learning.
    def __init__(self, label, feat_data,  opt):
        """Set the roidb to be used by this layer during training."""
        #self._roidb = roidb
        assert len(label) == feat_data.shape[0]
        self._opt = opt
        self._feat_data = feat_data
        self._label = label
        self._shuffle_roidb_inds()
        self._epoch = 0
    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._label)))
        # self._perm = np.arange(len(self._roidb))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""

        if self._cur + self._opt.batchsize >= len(self._label):
            self._shuffle_roidb_inds()
            self._epoch += 1

        db_inds = self._perm[self._cur:self._cur + self._opt.batchsize]
        self._cur += self._opt.batchsize

        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.
        """
        new_epoch = False
        if self._cur + self._opt.batchsize >= len(self._label):
            self._shuffle_roidb_inds()
            self._epoch += 1
            new_epoch = True

        db_inds = self._perm[self._cur:self._cur + self._opt.batchsize]
        self._cur += self._opt.batchsize

        minibatch_feat = np.array([self._feat_data[i] for i in db_inds])
        minibatch_label = np.array([self._label[i] for i in db_inds])
        blobs = {'data': minibatch_feat, 'labels':minibatch_label, 'newEpoch':new_epoch}
        return blobs

    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        return blobs

    def get_whole_data(self):
        blobs = {'data': self._feat_data, 'labels': self._label}
        return blobs


