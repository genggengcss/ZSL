import argparse
import json
import os
import os.path as osp
import sys
import time
import torch
sys.path.append('../')
from gpm import utils
import scipy.io as scio
import os
import numpy as np

DATA_DIR = '/home/gyx/ZSL/data/ImageNet/Baseline/DGP'
# DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet/Baseline/DGP'
EXP_NAME = 'Exp1_Ori'
# Material_DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet'
Material_DATA_DIR = '/home/gyx/ZSL/data/ImageNet'
Material_EXP_NAME = 'Exp1'



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # python test_gpm.py --pred 500 --nsample 500
    parser.add_argument('--pred', type=str, default='', help='the predicted classifier name')
    parser.add_argument('--nsample', type=int, help='extract sub-testing set')

    parser.add_argument('--unseen_feat', default=os.path.join(Material_DATA_DIR, 'Res101_Features', 'ILSVRC2011_Res101_Features'))
    parser.add_argument('--split', default=os.path.join(DATA_DIR, EXP_NAME, 'seen-unseen-split.json'))
    parser.add_argument('--gpu', default='3')
    parser.add_argument('--consider-trains', action='store_true')

    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    pred_file = os.path.join(DATA_DIR, EXP_NAME, 'save-gpm', 'epoch-'+args.pred+'.pred')
    # set_gpu(args.gpu)
    '''
    seen-unseen-split.json:
    split[seen], split[unseen]
    '''

    split = json.load(open(args.split, 'r'))
    seen_wnids = split['seen']
    unseen_wnids = split['unseen']

    print('seen: {}, unseen: {}'.format(len(seen_wnids), len(unseen_wnids)))
    print('consider train classifiers: {}'.format(args.consider_trains))


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

    print("********* Testing **********")
    # total_hits, total_imgs = 0, 0
    total_hits = torch.FloatTensor([0, 0, 0, 0, 0])  # top 1 2 5 10 20
    macro_hits = torch.FloatTensor([0, 0, 0, 0, 0])  # top 1 2 5 10 20
    total_imgs = 0
    # unseen_wnids = unseen_wnids[0]
    for i, wnid in enumerate(unseen_wnids, 1):
        all_label = n + i - 1
        # all_label = i
        # hit = 0
        # tot = 0
        top = [1, 2, 5, 10, 20]
        hits = torch.zeros(len(top))
        tot = 0

        # load test features begin
        feat_index = wnids.index(wnid) + 1
        feat_path = os.path.join(args.unseen_feat, str(feat_index)+'.mat')
        features = np.array(scio.loadmat(feat_path)['features'])
        if args.nsample and features.shape[0] > args.nsample:
            features = features[:args.nsample]
        # print('testing feat_data shape:', features.shape)
        # load test features end
        # print(type(features[0][0]))
        feat = torch.from_numpy(features).float()
        # print(type(features[0][0]))


        # feat = torch.cat([features, torch.ones(len(features)).view(-1, 1)], dim=1)

        fcs = pred_vectors.t()  # [2048, 883]

        table = torch.matmul(feat, fcs)
        # False: filter seen classifiers
        if not args.consider_trains:
            table[:, :n] = -1e18

        # for hit@1 and hit@2
        gth_score = table[:, all_label].repeat(table.shape[1], 1).t()
        rks = (table >= gth_score).sum(dim=1)
        assert (table[:, all_label] == gth_score[:, all_label]).min() == 1
        for j, k in enumerate(top):
            hits[j] += (rks <= k).sum().item()
        tot += len(features)

        total_hits += hits
        total_imgs += tot
        per_hits = hits / float(tot)
        macro_hits += per_hits

        # print('{}/{}, {}, total: {} : '.format(i, len(unseen_wnids), wnid, tot))
        # # hits = float(hits) / float(tot)
        # hits = [float(hit) / float(tot) for hit in hits]
        # output = ['{:.2f}'.format(i * 100) for i in hits]
        # print('results: ', output)





    print('total images: ', total_imgs)

    print('-----------------------------------------------')
    print('model : ', pred_file)

    # ### Macro Acc
    print("************ Macro Acc ************")
    macro_hits = [float(hit) / float(len(unseen_wnids)) for hit in macro_hits]
    output = ['{:.2f}'.format(i * 100) for i in macro_hits]
    print('results: ', output)
    # # total_hits = float(total_hits) / float(total_imgs)
    # ### Micro Acc
    print("************ Micro Acc ************")
    total_hits = [float(hit) / float(total_imgs) for hit in total_hits]
    output = ['{:.2f}'.format(i * 100) for i in total_hits]
    print('results: ', output)

    print('-----------------------------------------------')

    if args.output is not None:
        json.dump(results, open(args.output, 'w'))
