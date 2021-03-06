# coding=gbk
# -*- coding: utf-8 -*-

import argparse
import json
import numpy as np
import pickle as pkl
from scipy import sparse
import os
import scipy.io as scio
from tensorflow.python import pywrap_tensorflow

from io_graph import prepare_graph
'''
convert to gcn data: input, output, graph
input: word embedding;
'''

DATA_DIR = '/home/gyx/ZSL/data/ImageNet/Baseline/GCNZ'
# DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet/Baseline/GCNZ'
# Material_DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet'
Material_DATA_DIR = '/home/gyx/ZSL/data/ImageNet'

# save embedding vectors of all the vertices of the graph
def convert_input(w2v_file, save_dir):


    # load all graph nodes
    inv_wordn_file = os.path.join(DATA_DIR, args.exp_name, 'invdict_wordn.json')
    with open(inv_wordn_file) as fp:
        nodes = json.load(fp)

    # load w2v.mat
    matcontent = scio.loadmat(w2v_file)
    w2v = matcontent['w2v']
    allwnids = matcontent['wnids'].squeeze().tolist()
    all_w_feats = list()
    for wnid in nodes:
        wnid_index = allwnids.index(wnid)
        w2v_feat = w2v[wnid_index]
        all_w_feats.append(w2v_feat)
    all_w_feats = np.array(all_w_feats)  # (3695, 500)



    dense_file = os.path.join(save_dir, 'all_x_dense.pkl')  # embedding vectors of all vertices
    with open(dense_file, 'wb') as fp:
        pkl.dump(all_w_feats, fp)


    print('Save vectors of all vertices')


def convert_label(seen_feat_path, save_dir):  # get output's label and mask
    ''' average the seen samples' features, and take as visual classifier '''
    corresp_file = os.path.join(DATA_DIR, args.exp_name, 'corresp.json')
    with open(corresp_file) as fp:
        corresp_list = json.load(fp)


    seen_file = os.path.join(Material_DATA_DIR, 'KG-GAN', args.mtr_exp_name, 'seen.txt')  # num:398
    seen_list = list()
    with open(seen_file) as fp:
        for line in fp.readlines():
            seen_list.append(line.strip())
    print('seen list', len(seen_list))  # 398



    fc_labels = np.zeros((len(corresp_list), fc_dim))  # (3695, 2048)
    print('fc labels dim ', fc_labels.shape)

    matcontent = scio.loadmat(os.path.join(Material_DATA_DIR, 'split.mat'))
    allwnids = matcontent['allwnids'].squeeze().tolist()

    for i, corresp in enumerate(corresp_list):
        vertex_type = corresp[1]
        class_id = corresp[0]
        if vertex_type == 0:
            seen_wnid = seen_list[class_id]
            feat_index = allwnids.index(seen_wnid) + 1
            feat_file = os.path.join(seen_feat_path, str(feat_index) + '.mat')
            seen_feat = np.array(scio.loadmat(feat_file)['features']).astype(np.float32)
            feats = np.mean(seen_feat, axis=0)  # (2048)
            fc_labels[i, :] = feats



    label_file = os.path.join(save_dir, 'train_y.pkl')   #  和all_a_dense.pkl同样的行数，seen classes的feature wegiths写到对应的行，其它的行为0；
    with open(label_file, 'wb') as fp:
        pkl.dump(fc_labels, fp)

    # the position that is 1 means the vertex of that position is an unseen class
    test_index = []
    for corresp in corresp_list:
        if corresp[0] == -1:
            test_index.append(-1)
        else:
            test_index.append(corresp[1])  # corresp[1]: 0/1, the value 0 means seen class, 1 means unseen classes
    test_file = os.path.join(save_dir, 'test_index.pkl')  # 和all_a_dense.pkl同样的行数
    with open(test_file, 'wb') as fp:
        pkl.dump(test_index, fp)


def convert_graph(save_dir):
    graph_file = os.path.join(DATA_DIR, args.exp_name, 'graph.pkl')
    if not os.path.exists(graph_file):
        prepare_graph()
    save_file = os.path.join(save_dir, 'graph.pkl')
    if os.path.exists(save_file):
        cmd = 'rm  %s' % save_file
        os.system(cmd)
    cmd = 'ln -s %s %s' % (graph_file, save_file)  # set soft link, i.e., link the graph.pkl to save_dir
    os.system(cmd)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mtr_exp_name', default='Exp2', help='the folder to store Material files')
    parser.add_argument('--exp_name', default='Exp2_1949', help='the folder to store experiment files')

    parser.add_argument('--proposed_split', action='store_true', default=False, help='use the specified dataset split')

    parser.add_argument('--fc_dim', type=int, default='2048', help='feature dimension of CNN features')
    parser.add_argument('--seen_feat', default='', help='path of seen pre-trained features')

    args = parser.parse_args()


    if args.proposed_split:
        args.seen_feat = os.path.join(Material_DATA_DIR, 'Res101_Features', 'ProposedSplit/Seen_train')
    else:
        args.seen_feat = os.path.join(Material_DATA_DIR, 'Res101_Features', 'StandardSplit/ILSVRC2012_train')


    fc_dim = args.fc_dim
    w2v_file = os.path.join(Material_DATA_DIR, 'w2v.mat')


    save_dir = os.path.join(DATA_DIR, args.exp_name, 'w2v_res101')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    print('Converting input')
    convert_input(w2v_file, save_dir)

    print('Converting graph')
    convert_graph(save_dir)

    print('Converting label')
    convert_label(args.seen_feat, save_dir)
    print('Prepared data to %s' % save_dir)
