import argparse
import json
import os
import os.path as osp
import sys
import time
import torch
sys.path.append('../../')
from DGP.gpm import utils
import scipy.io as scio
import os
import numpy as np

DATA_DIR = '/home/gyx/ZSL/data/ImageNet/Baseline/DGP'
# DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet/Baseline/DGP'
# Material_DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet'
Material_DATA_DIR = '/home/gyx/ZSL/data/ImageNet'



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # python test_gpm.py --pred 500 --nsample 500
    parser.add_argument('--mtr_exp_name', default='Exp2')
    parser.add_argument('--exp_name', default='Exp2_1949')
    parser.add_argument('--pred', type=str, default='', help='the predicted classifier name')
    parser.add_argument('--nsample', type=int, help='extract sub-testing set')
    parser.add_argument('--per_class_acc', action='store_true', default=False, help='test the accuracy of each class')
    parser.add_argument('--gzsl', action='store_true', default=False, help='test the accuracy of each class')

    parser.add_argument('--proposed_split', action='store_true', default=False)
    parser.add_argument('--unseen_feat', default=os.path.join(Material_DATA_DIR, 'Res101_Features', 'ProposedSplit/Unseen'))
    parser.add_argument('--seen_feat', default=os.path.join(Material_DATA_DIR, 'Res101_Features', 'ProposedSplit/Seen_val'))
    parser.add_argument('--gpu', default='3')

    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    print('Exp_Name: {:s}, Class Ref: {:s}, Dataset Split: {:s}, GZSL:{:s}'.format(
        args.exp_name, args.mtr_exp_name, str(args.proposed_split), str(args.gzsl)))

    if args.proposed_split:
        args.seen_feat = os.path.join(Material_DATA_DIR, 'Res101_Features', 'ProposedSplit/Seen_val')
        args.unseen_feat = os.path.join(Material_DATA_DIR, 'Res101_Features', 'ProposedSplit/Unseen')
    else:
        args.seen_feat = os.path.join(Material_DATA_DIR, 'Res101_Features', 'StandardSplit/ILSVRC2012_val')
        args.unseen_feat = os.path.join(Material_DATA_DIR, 'Res101_Features', 'StandardSplit/ILSVRC2011')



    pred_file = os.path.join(DATA_DIR, args.exp_name, 'save-gpm', 'epoch-'+args.pred+'.pred')
    split_file = os.path.join(DATA_DIR, args.exp_name, 'seen-unseen-split.json')
    # set_gpu(args.gpu)
    '''
    seen-unseen-split.json:
    split[seen], split[unseen]
    '''

    split = json.load(open(split_file, 'r'))
    seen_wnids = split['seen']
    unseen_wnids = split['unseen']

    print('seen: {}, unseen: {}'.format(len(seen_wnids), len(unseen_wnids)))
    print('consider train classifiers: {}'.format(args.gzsl))


    preds = torch.load(pred_file)
    pred_wnids = preds['wnids']
    pred_vectors = preds['pred']  # (3969, 2049)

    pred_dic = dict(zip(pred_wnids, pred_vectors))  # packed into tuple
    # select unseen pred_vectors
    pred_vectors = utils.pick_vectors(pred_dic, seen_wnids+unseen_wnids, is_tensor=True)
    print("pred_vector shape:", pred_vectors.shape)
    n = len(seen_wnids)
    m = len(unseen_wnids)


    # test_names = awa2_split['test_names']

    ave_acc = 0
    ave_acc_n = 0

    results = {}

    # testing data features
    # imagenet_test_path = args.test_feat

    matcontent = scio.loadmat(os.path.join(Material_DATA_DIR, 'split.mat'))
    wnids = matcontent['allwnids'].squeeze().tolist()
    words = matcontent['allwords'].squeeze()[:2549]
    top = [1, 2, 5, 10, 20]


    print("********* Testing Unseen Data **********")
    # total_hits, total_imgs = 0, 0
    unseen_micro_hits = torch.FloatTensor([0, 0, 0, 0, 0])  # top 1 2 5 10 20
    unseen_macro_hits = torch.FloatTensor([0, 0, 0, 0, 0])  # top 1 2 5 10 20
    unseen_total_imgs = 0
    # unseen_wnids = unseen_wnids[0]
    for i, wnid in enumerate(unseen_wnids, 1):
        all_label = n + i - 1

        hits = torch.zeros(len(top))
        tot = 0

        # load test features begin
        feat_index = wnids.index(wnid) + 1
        feat_path = os.path.join(args.unseen_feat, str(feat_index)+'.mat')
        features = np.array(scio.loadmat(feat_path)['features'])
        if args.nsample and features.shape[0] > args.nsample:
            features = features[:args.nsample]

        feat = torch.from_numpy(features).float()


        fcs = pred_vectors.t()  # [2048, 883]
        table = torch.matmul(feat, fcs)
        # False: filter seen classifiers
        if not args.gzsl:
            table[:, :n] = -1e18

        # for hit@1 and hit@2
        gth_score = table[:, all_label].repeat(table.shape[1], 1).t()
        rks = (table >= gth_score).sum(dim=1)
        assert (table[:, all_label] == gth_score[:, all_label]).min() == 1
        for j, k in enumerate(top):
            hits[j] += (rks <= k).sum().item()
        tot += len(features)

        unseen_micro_hits += hits
        unseen_total_imgs += tot
        per_hits = hits / float(tot)
        unseen_macro_hits += per_hits

        # output per class accuracy
        if args.per_class_acc:
            name = str(words[wnids.index(wnid)][0]).split(',')[0]
            print('{}/{}, {}, {}, total: {} : '.format(i, len(unseen_wnids), wnid, name, tot))
            # hits = float(hits) / float(tot)
            hits = [float(hit) / float(tot) for hit in hits]
            output = ['{:.2f}'.format(i * 100) for i in hits]
            print('results: ', output)

    # print('total images: ', unseen_total_imgs)

    print('-----------------------------------------------')
    print('model : ', pred_file)

    # ### Macro Acc
    print("************ Unseen Macro Acc ************")
    # unseen_macro_hits = [float(hit) / float(len(unseen_wnids)) for hit in unseen_macro_hits]
    unseen_macro_hits = unseen_macro_hits * 1.0 / len(unseen_wnids)
    output = ['{:.2f}'.format(i * 100) for i in unseen_macro_hits]
    print('results: ', output)
    # # total_hits = float(total_hits) / float(total_imgs)
    # ### Micro Acc
    # print("************ Unseen Micro Acc ************")
    # total_hits = [float(hit) / float(unseen_total_imgs) for hit in unseen_micro_hits]
    # output = ['{:.2f}'.format(i * 100) for i in total_hits]
    # print('results: ', output)
    #
    # print('-----------------------------------------------')

    if args.gzsl:
        print("********* Testing Seen Data **********")
        # total_hits, total_imgs = 0, 0
        seen_micro_hits = torch.FloatTensor([0, 0, 0, 0, 0])  # top 1 2 5 10 20
        seen_macro_hits = torch.FloatTensor([0, 0, 0, 0, 0])  # top 1 2 5 10 20
        seen_total_imgs = 0
        # unseen_wnids = unseen_wnids[0]
        for i, wnid in enumerate(seen_wnids, 1):
            all_label = i - 1

            hits = torch.zeros(len(top))
            tot = 0

            # load test features begin
            feat_index = wnids.index(wnid) + 1
            feat_path = os.path.join(args.seen_feat, str(feat_index) + '.mat')
            features = np.array(scio.loadmat(feat_path)['features'])
            if args.nsample and features.shape[0] > args.nsample:
                features = features[:args.nsample]

            feat = torch.from_numpy(features).float()

            fcs = pred_vectors.t()  # [2048, 883]
            table = torch.matmul(feat, fcs)
            # False: filter seen classifiers
            if not args.gzsl:
                table[:, :n] = -1e18

            # for hit@1 and hit@2
            gth_score = table[:, all_label].repeat(table.shape[1], 1).t()
            rks = (table >= gth_score).sum(dim=1)
            assert (table[:, all_label] == gth_score[:, all_label]).min() == 1
            for j, k in enumerate(top):
                hits[j] += (rks <= k).sum().item()
            tot += len(features)

            seen_micro_hits += hits
            seen_total_imgs += tot
            per_hits = hits / float(tot)
            seen_macro_hits += per_hits


        # ### Macro Acc
        print("************ Seen Macro Acc ************")
        # seen_macro_hits = [float(hit) / float(len(seen_wnids)) for hit in seen_macro_hits]
        seen_macro_hits = seen_macro_hits * 1.0 / len(seen_wnids)
        output = ['{:.2f}'.format(i * 100) for i in seen_macro_hits]
        print('results: ', output)
        # # total_hits = float(total_hits) / float(total_imgs)

        print("************* H value ************")
        acc_H = 2 * seen_macro_hits * unseen_macro_hits / (seen_macro_hits + unseen_macro_hits)
        output = ['{:.2f}'.format(i * 100) for i in acc_H]
        print('results: ', output)

        print('-----------------------------------------------')

    if args.output is not None:
        json.dump(results, open(args.output, 'w'))
