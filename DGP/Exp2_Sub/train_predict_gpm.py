import argparse
import json
import random
import os
import sys

import torch
import torch.nn.functional as F
import scipy.io as scio
import numpy as np
sys.path.append('../../')
from DGP.gpm import gcn
from DGP.gpm import utils
# from IMAGENET_Animal.ZSL_Model.Exp1_GPM import test_in_train
'''
input: imagenet-induced-animal-graph.json, fc-weights.json
get: save prediction model file
function: train with gcn(2 layers) and predict testing features
'''


DATA_DIR = '/home/gyx/ZSL/data/ImageNet/Baseline/DGP'
# DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet/Baseline/DGP'
# Material_DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet'
Material_DATA_DIR = '/home/gyx/ZSL/data/ImageNet'

# NSample = 1000  # number of seen class samples, for computing the classifier for each seen classes

def save_checkpoint(name):
    torch.save(gcn.state_dict(), os.path.join(save_path, name + '.pth'))
    torch.save(pred_obj, os.path.join(save_path, name + '.pred'))


def mask_l2_loss(a, b, mask):
    return utils.l2_loss(a[mask], b[mask])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mtr_exp_name', default='Exp2')
    parser.add_argument('--exp_name', default='Exp2_1949')
    parser.add_argument('--proposed_split', action='store_true', default=False)


    parser.add_argument('--max_epoch', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--save_epoch', type=int, default=200)
    parser.add_argument('--evaluate_epoch', type=int, default=10)

    parser.add_argument('--seen_feat', default='')
    parser.add_argument('--gpu', default='3')
    parser.add_argument('--no-pred', action='store_true')
    args = parser.parse_args()

    if args.proposed_split:
        args.seen_feat = os.path.join(Material_DATA_DIR, 'Res101_Features', 'ProposedSplit/Seen_train')
    else:
        args.seen_feat = os.path.join(Material_DATA_DIR, 'Res101_Features', 'StandardSplit/ILSVRC2012_train')

    graph_file = os.path.join(DATA_DIR, args.exp_name, 'induced-graph.json')
    save_path = os.path.join(DATA_DIR, args.exp_name, 'save-gpm')
    utils.ensure_path(save_path)
    split_file = os.path.join(DATA_DIR, args.exp_name, 'seen-unseen-split.json')


    utils.set_gpu(args.gpu)



    graph = json.load(open(graph_file, 'r'))
    wnids = graph['wnids']
    n = len(wnids)
    edges = graph['edges']

    # + inverse edges and reflexive edges
    edges = edges + [(v, u) for (u, v) in edges]
    edges = edges + [(u, u) for u in range(n)]

    word_vectors = torch.tensor(graph['vectors'])
    word_vectors = F.normalize(word_vectors)


    # training supervision (average seen features)  start
    print("********** Processing Seen Classifier ***********")
    matcontent = scio.loadmat(os.path.join(Material_DATA_DIR, 'split.mat'))
    allwnids = matcontent['allwnids'].squeeze().tolist()
    allwords = matcontent['allwords'].squeeze()[:2549]

    split = json.load(open(split_file, 'r'))
    # print split
    train_wnids = split['seen']  # 398->398
    test_wnids = split['unseen']  # 485
    feat_set = list()

    for wnid in train_wnids:
        feat_index = allwnids.index(wnid) + 1
        feat_path = os.path.join(args.seen_feat, str(feat_index) + '.mat')
        seen_feat = np.array(scio.loadmat(feat_path)['features'])
        # if seen_feat.shape[0] > NSample:
        #     seen_feat = seen_feat[:NSample]
        feats = torch.from_numpy(seen_feat).float()  # (1000, 2048)
        feats = torch.mean(feats, dim=0)  # (2048)
        feat_set.append(feats)
    fc_vectors = np.vstack(tuple(feat_set))
    fc_vectors = torch.from_numpy(fc_vectors)
    fc_vectors = F.normalize(fc_vectors)  # shape: (398, 2049)
    # training supervision (average seen features)  end

    # construct gcn model
    hidden_layers = 'd2048,d'
    gcn = gcn.GCN(n, edges, word_vectors.shape[1], fc_vectors.shape[1], hidden_layers)

    print('{} nodes, {} edges'.format(n, len(edges)))
    print('word vectors:', word_vectors.shape)
    print('fc vectors:', fc_vectors.shape)
    print('hidden layers:', hidden_layers)

    optimizer = torch.optim.Adam(gcn.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # split seen nodes into training set and validation set
    n_train = len(fc_vectors)  # 1000, training number?
    print('num train: {}'.format(n_train))  # 1000, 0

    tlist = list(range(len(fc_vectors)))  # 1000
    random.shuffle(tlist)

    min_loss = 1e18

    trlog = {}
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['min_loss'] = 0

    # start learning ...
    for epoch in range(1, args.max_epoch + 1):
        gcn.train()
        output_vectors = gcn(word_vectors)

        # calculate the loss over training seen nodes
        loss = mask_l2_loss(output_vectors, fc_vectors, tlist[:n_train])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calculate loss on training (and validation) seen nodes
        if epoch % args.evaluate_epoch == 0:
            gcn.eval()
            output_vectors = gcn(word_vectors)
            train_loss = mask_l2_loss(output_vectors, fc_vectors, tlist[:n_train]).item()

            loss = train_loss
            print('epoch {}, train_loss={:.4f}'.format(epoch, train_loss))

            trlog['train_loss'].append(train_loss)
            trlog['min_loss'] = min_loss
            torch.save(trlog, os.path.join(save_path, 'trlog'))

        # save intermediate output_vector of each node of the graph
        if epoch % 10 == 0 and epoch >= args.save_epoch:
            if args.no_pred:
                pred_obj = None
            else:
                pred_obj = {
                    'wnids': wnids,
                    'pred': output_vectors
                }
        if epoch % 50 == 0 and epoch >= args.save_epoch:
            save_checkpoint('epoch-{}'.format(epoch))

        pred_obj = None



