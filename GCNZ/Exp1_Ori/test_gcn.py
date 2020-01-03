# coding=gbk
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import json
import numpy as np
import os
import pickle as pkl
import time
import scipy.io as scio
import tensorflow as tf

'''
testing file 
test the total accuracy of gcn model
'''


DATA_DIR = '/home/gyx/ZSL/data/ImageNet/Baseline/GCNZ'
# DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet/Baseline/GCNZ'
EXP_NAME = 'Exp1_Ori'
# Material_DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet'
Material_DATA_DIR = '/home/gyx/ZSL/data/ImageNet'
Material_EXP_NAME = 'Exp1'

UnseenFeaFile = 'Res101_Features/ILSVRC2011_Res101_Features'

# global varc
w2v_file = os.path.join(DATA_DIR, EXP_NAME, 'w2v_res101', 'all_x_dense.pkl')
classids_file_retrain = os.path.join(DATA_DIR, EXP_NAME, 'corresp.json')



def test_imagenet_zero(weight_pred_file, has_train=False):

    with open(classids_file_retrain) as fp:  # corresp.json
        classids = json.load(fp)

    with open(w2v_file, 'rb') as fp:  # glove
        # word2vec_feat = pkl.load(fp, encoding='iso-8859-1')
        word2vec_feat = pkl.load(fp)

    # obtain training results
    with open(weight_pred_file, 'rb') as fp:  #
        weight_pred = pkl.load(fp)
    weight_pred = np.array(weight_pred)

    print('weight_pred output shape', weight_pred.shape)

    # process 'train' classes. they are possible candidates during inference
    invalid_wv = 0  # count the number of invalid class embedding
    labels_testval, word2vec_testval = [], []  # zsl: unseen label and its class embedding
    weight_pred_testval = []  # zsl: unseen output feature
    for j in range(len(classids)):
        t_wpt = weight_pred[j]
        if has_train:
            if classids[j][0] < 0:
                continue
        else:
            if classids[j][1] == 0:
                continue
        if classids[j][0] >= 0:
            t_wv = word2vec_feat[j]
            if np.linalg.norm(t_wv) == 0:  # 求范数
                invalid_wv = invalid_wv + 1
                continue
            labels_testval.append(classids[j][0])
            word2vec_testval.append(t_wv)
            weight_pred_testval.append(t_wpt)
    weight_pred_testval = np.array(weight_pred_testval)
    print('skip candidate class due to no word embedding: %d / %d:' % (invalid_wv, len(labels_testval) + invalid_wv))
    print('candidate class shape: ', weight_pred_testval.shape)

    weight_pred_testval = weight_pred_testval.T
    labels_testval = np.array(labels_testval)
    print('final test classes: ', len(labels_testval))
    # print(labels_testval)

    # remove invalid unseen classes(wv = 0)
    valid_class = np.zeros(22000)
    invalid_unseen_wv = 0
    for j in range(len(classids)):
        if classids[j][1] == 1:  # unseen classes
            t_wv = word2vec_feat[j]
            t_wv = t_wv / (np.linalg.norm(t_wv) + 1e-6)

            if np.linalg.norm(t_wv) == 0:
                invalid_unseen_wv = invalid_unseen_wv + 1
                continue
            valid_class[classids[j][0]] = 1


    # load test data start
    seen_file = os.path.join(Material_DATA_DIR, 'KG-GAN', Material_EXP_NAME, 'seen.txt')  # nun:249
    unseen_file = os.path.join(Material_DATA_DIR, 'KG-GAN', Material_EXP_NAME, 'unseen.txt')  # num: 361
    seen_list = list()
    with open(seen_file) as fp:
        for line in fp.readlines():
            seen_list.append(line.strip())
    seen_num = len(seen_list)
    unseen_list = list()
    with open(unseen_file) as fp:
        for line in fp.readlines():
            unseen_list.append(line.strip())

    matcontent = scio.loadmat(os.path.join(Material_DATA_DIR, 'split.mat'))
    allwnids = matcontent['allwnids'].squeeze().tolist()
    words = matcontent['allwords'].squeeze()[:2549]

    ## imagenet 2-hops topK result
    top_retrv = [1, 2, 5, 10, 20]
    micro_count = np.zeros((len(top_retrv)))
    macro_count = np.zeros((len(top_retrv)))


    valid_classes = 0  # count testing classes
    total_imgs = 0  # count testing images

    print("********* Testing **********")
    for i, unseen in enumerate(unseen_list):
        unseen_label = i + seen_num
        if valid_class[unseen_label] == 0:   # remove invalid unseen classes
            continue
        valid_classes += 1

        unseen_index = allwnids.index(unseen) + 1
        file = os.path.join(Material_DATA_DIR, UnseenFeaFile, str(unseen_index) + '.mat')
        feature = np.array(scio.loadmat(file)['features']).astype(np.float32)
        if args.nsample and feature.shape[0] > args.nsample:
                feature = feature[:args.nsample]

        per_total = 0
        hits = np.zeros(len(top_retrv))
        for j in range(len(feature)):
            per_total += 1
            test_feat = feature[j]
            scores = np.dot(test_feat, weight_pred_testval).squeeze()

            scores = scores - scores.max()
            scores = np.exp(scores)
            scores = scores / scores.sum()

            ids = np.argsort(-scores)

            # for top in range(len(topKs)):
            for k in range(len(top_retrv)):
                current_len = top_retrv[k]
                for sort_id in range(current_len):
                    lbl = labels_testval[ids[sort_id]]
                    if int(lbl) == unseen_label:
                        micro_count[k] = micro_count[k] + 1
                        hits[k] = hits[k] + 1
                        break
        per_hits = hits * 1.0 / per_total
        macro_count += per_hits
        total_imgs += per_total

        # print('{}/{}, {}, total: {} : '.format(i, len(unseen_list), unseen, per_total))
        # output = ['{:.2f}'.format(i * 100) for i in per_hits]
        # print('results: ', output)



    print('total images: ', total_imgs)

    print('-----------------------------------------------')
    print('model : ', training_outputs)

    # ### Macro Acc
    print("************ Macro Acc ************")
    macro_hits = macro_count * 1.0 / valid_classes
    output = ['{:.2f}'.format(i * 100) for i in macro_hits]
    print('results: ', output)

    # ### Micro Acc
    print("************ Micro Acc ************")
    micro_hits = micro_count * 1.0 / total_imgs
    output = ['{:.2f}'.format(i * 100) for i in micro_hits]
    print('results: ', output)

    print('-----------------------------------------------')





if __name__ == '__main__':

    # python test_gcn.py --feat 800 --nsample 500
    parser = argparse.ArgumentParser()

    parser.add_argument('--feat', type=str, default='850', help='the predicted classifier name')
    parser.add_argument('--nsample', type=int, help='extract sub-testing set')
    args = parser.parse_args()

    training_outputs = os.path.join(DATA_DIR, EXP_NAME, 'w2v_res101/output', 'feat_'+args.feat)
    print('\nEvaluating ...\nPlease be patient for it takes a few minutes...')

    test_imagenet_zero(weight_pred_file=training_outputs)



