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
test the accuracy of gcn model
'''

DATA_DIR = '/home/gyx/ZSL/data/ImageNet/Baseline/GCNZ'
# DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet/Baseline/GCNZ'
# Material_DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet'
Material_DATA_DIR = '/home/gyx/ZSL/data/ImageNet'





def test(weight_pred_file, w2v_file, classids_file_retrain, has_train=False):

    with open(classids_file_retrain) as fp:  # corresp.json
        classids = json.load(fp)

    with open(w2v_file, 'rb') as fp:  # glove
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
    seen_file = os.path.join(Material_DATA_DIR, 'KG-GAN', args.mtr_exp_name, 'seen.txt')  # nun:249
    unseen_file = os.path.join(Material_DATA_DIR, 'KG-GAN', args.mtr_exp_name, 'unseen.txt')  # num: 361
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

    # test unseen data
    unseen_micro_count = np.zeros((len(top_retrv)))
    unseen_macro_count = np.zeros((len(top_retrv)))

    unseen_valid_classes = 0  # count testing classes
    unseen_total_imgs = 0  # count testing images

    print("********* Testing Unseen Data **********")
    for i, unseen in enumerate(unseen_list):
        unseen_label = i + seen_num
        if valid_class[unseen_label] == 0:   # remove invalid unseen classes
            continue
        unseen_valid_classes += 1

        unseen_index = allwnids.index(unseen) + 1
        file = os.path.join(Material_DATA_DIR, args.unseen_feat, str(unseen_index) + '.mat')
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
                        unseen_micro_count[k] = unseen_micro_count[k] + 1
                        hits[k] = hits[k] + 1
                        break
        per_hits = hits * 1.0 / per_total
        unseen_macro_count += per_hits
        unseen_total_imgs += per_total

        if args.per_class_acc:
            print('{}/{}, {}, total: {} : '.format(i, len(unseen_list), unseen, per_total))
            output = ['{:.2f}'.format(i * 100) for i in per_hits]
            print('results: ', output)




    print('total images: ', unseen_total_imgs)

    print('-----------------------------------------------')
    print('model : ', weight_pred_file)

    # ### Macro Acc
    print("************ Unseen Macro Acc ************")
    unseen_macro_hits = unseen_macro_count * 1.0 / unseen_valid_classes
    output = ['{:.2f}'.format(i * 100) for i in unseen_macro_hits]
    print('results: ', output)

    # ### Micro Acc
    # print("************ Micro Acc ************")
    # micro_hits = unseen_micro_count * 1.0 / unseen_total_imgs
    # output = ['{:.2f}'.format(i * 100) for i in micro_hits]
    # print('results: ', output)
    #
    # print('-----------------------------------------------')

    if args.gzsl:

        # test seen data
        seen_micro_count = np.zeros((len(top_retrv)))
        seen_macro_count = np.zeros((len(top_retrv)))

        seen_valid_classes = 0  # count testing classes
        seen_total_imgs = 0  # count testing images

        print("********* Testing Seen Data **********")
        for i, seen in enumerate(seen_list):
            seen_label = i
            # if valid_class[seen_label] == 0:  # remove invalid unseen classes
            #     continue
            seen_valid_classes += 1

            seen_index = allwnids.index(seen) + 1
            file = os.path.join(Material_DATA_DIR, args.seen_feat, str(seen_index) + '.mat')
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
                        if int(lbl) == seen_label:
                            seen_micro_count[k] = seen_micro_count[k] + 1
                            hits[k] = hits[k] + 1
                            break
            per_hits = hits * 1.0 / per_total
            seen_macro_count += per_hits
            seen_total_imgs += per_total

            # if args.per_class_acc:
            #     print('{}/{}, {}, total: {} : '.format(i, len(seen_list), seen, per_total))
            #     output = ['{:.2f}'.format(i * 100) for i in per_hits]
            #     print('results: ', output)

        # print('total images: ', seen_total_imgs)

        print("seen valid classes:", seen_valid_classes)

        # ### Macro Acc
        print("************ Seen Macro Acc ************")
        seen_macro_hits = seen_macro_count * 1.0 / seen_valid_classes
        output = ['{:.2f}'.format(i * 100) for i in seen_macro_hits]
        print('results: ', output)

        print("************* H value ************")
        acc_H = 2 * seen_macro_hits * unseen_macro_hits / (seen_macro_hits + unseen_macro_hits)
        output = ['{:.2f}'.format(i * 100) for i in acc_H]
        print('results: ', output)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument('--mtr_exp_name', default='', help='the folder to store Material files')
    parser.add_argument('--exp_name', default='', help='the folder to store experiment files')

    parser.add_argument('--feat', type=str, default='850', help='the predicted classifier name')
    parser.add_argument('--nsample', type=int, help='extract sub-testing set')
    parser.add_argument('--gzsl', action='store_true', default=False, help='enables gzsl')
    parser.add_argument('--per_class_acc', action='store_true', default=False, help='test the accuracy of each class')

    parser.add_argument('--proposed_split', action='store_true', default=False)
    parser.add_argument('--unseen_feat', default='')
    parser.add_argument('--seen_feat', default='')
    args = parser.parse_args()

    if args.proposed_split:
        args.seen_feat = os.path.join(Material_DATA_DIR, 'Res101_Features', 'ProposedSplit/Seen_val')
        args.unseen_feat = os.path.join(Material_DATA_DIR, 'Res101_Features', 'ProposedSplit/Unseen')
    else:
        args.seen_feat = os.path.join(Material_DATA_DIR, 'Res101_Features', 'StandardSplit/ILSVRC2012_val')
        args.unseen_feat = os.path.join(Material_DATA_DIR, 'Res101_Features', 'StandardSplit/ILSVRC2011')



    # training outputs
    weight_pred_file = os.path.join(DATA_DIR, args.exp_name, 'w2v_res101/output', 'feat_'+args.feat)
    print('\nEvaluating ...\nPlease be patient for it takes a few minutes...')

    w2v_file = os.path.join(DATA_DIR, args.exp_name, 'w2v_res101', 'all_x_dense.pkl')
    classids_file_retrain = os.path.join(DATA_DIR, args.exp_name, 'corresp.json')

    test(weight_pred_file,  w2v_file, classids_file_retrain, has_train=args.gzsl)



