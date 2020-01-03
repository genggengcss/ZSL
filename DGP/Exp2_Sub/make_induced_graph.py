import argparse
import json
import os
import sys
import torch

from nltk.corpus import wordnet as wn
import scipy.io as scio
import numpy as np
sys.path.append('../../')
from DGP.gpm import utils
'''
input: seen.txt, unseen.txt, imagenet-xml-wnids.json, w2v.mat
get:  'induced-graph.json'
function: construct graph (total nodes) and get node embedding
'''


DATA_DIR = '/home/gyx/ZSL/data/ImageNet/Baseline/DGP'
# DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet/Baseline/DGP'
# Material_DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet'
Material_DATA_DIR = '/home/gyx/ZSL/data/ImageNet'

# read txt file
def readTxtFile(file):
    lines = list()
    nodes = open(file, 'rU')
    try:
        for line in nodes:
            line = line[:-1]
            lines.append(line)
    finally:
        nodes.close()
    print(len(lines))
    return lines

def getnode(x):
    return wn.synset_from_pos_and_offset('n', int(x[1:]))

def getwnid(u):
    s = str(u.offset())
    return 'n' + (8 - len(s)) * '0' + s


# check every node's parent (not including ancestors > 2-hops)
def getedges(s):
    dic = {x: i for i, x in enumerate(s)}
    edges = []
    for i, u in enumerate(s):
        for v in u.hypernyms():
            j = dic.get(v)
            if j is not None:
                edges.append((i, j))
    return edges


# add the parents of nodes of s that are not among stop_set to s
def induce_parents(s, stop_set):
    q = s
    vis = set(s)
    l = 0
    while l < len(q):
        u = q[l]
        l += 1
        if u in stop_set:
            continue
        for p in u.hypernyms():
            if p not in vis:
                vis.add(p)
                q.append(p)

'''
Exp2: classes with standard split (seen:19, unseen 49)
Exp3: classes with re-split (proposed split, seen:14, unseen 54)
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mtr_exp_name', default='Exp2', help='the folder to store Material files')
    parser.add_argument('--exp_name', default='Exp2_1949')

    args = parser.parse_args()


    seen_file = os.path.join(Material_DATA_DIR, 'KG-GAN', args.mtr_exp_name, 'seen.txt')
    unseen_file = os.path.join(Material_DATA_DIR, 'KG-GAN', args.mtr_exp_name, 'unseen.txt')
    seen_unseen_save_file = os.path.join(DATA_DIR, args.exp_name, 'seen-unseen-split.json')
    graph_xml_file = os.path.join(DATA_DIR, 'materials', 'imagenet-xml-wnids-animal.json')
    w2v_file = os.path.join(Material_DATA_DIR, 'w2v.mat')
    output_file = os.path.join(DATA_DIR, args.exp_name, 'induced-graph.json')

    utils.ensure_path(os.path.join(DATA_DIR, args.exp_name))

    print('making graph ...')
    # animal subset, length: 3695
    xml_wnids = json.load(open(graph_xml_file, 'r'))
    print('xml_nodes:', len(xml_wnids))
    xml_nodes = list(map(getnode, xml_wnids))
    xml_set = set(xml_nodes)  # get wordnet node text

    # imagenet animal subset split
    seen_wnids = readTxtFile(seen_file)  # 398
    unseen_wnids = readTxtFile(unseen_file)  # 485

    key_wnids = seen_wnids + unseen_wnids

    # # store seen and unseen
    print("******** seen and unseen split ******")
    obj = {'seen': seen_wnids, 'unseen': unseen_wnids}
    json.dump(obj, open(seen_unseen_save_file, 'w'))

    s = list(map(getnode, key_wnids))  # get nodes' text
    print(len(s))  # 883

    '''
    Actually this does not make any changes, as all parents of nodes in s are among 'xml_set'
    s: 21842/3969
    '''
    induce_parents(s, xml_set)

    # len(s) : 3695
    # len(s_set) : 3695
    # len(s) : 3969
    s_set = set(s)
    for u in xml_nodes:
        if u not in s_set:
            s.append(u)
    '''
        new s: 32324, xml_nodes: 32295
        this means 29 nodes of train+test are not among xml_nodes

        the function of above step: construct complete graph (or node set), the total number is: 32324  
    '''
    print(len(s))  # 3695

    wnids = list(map(getwnid, s))  # get s's wnids (graph nodes)
    # print wnids
    edges = getedges(s)

    print('load word embedding ...')

    # load w2v.mat
    matcontent = scio.loadmat(w2v_file)
    w2v = matcontent['w2v']
    allwnids = matcontent['wnids'].squeeze().tolist()
    all_w_feats = list()
    for wnid in wnids:
        wnid_index = allwnids.index(wnid)
        w2v_feat = w2v[wnid_index]
        all_w_feats.append(w2v_feat)
    all_w_feats = np.array(all_w_feats)  # (3695, 500)
    print(all_w_feats.shape)


    print('done ...')

    obj = {'wnids': wnids, 'vectors': all_w_feats.tolist(), 'edges': edges}
    json.dump(obj, open(output_file, 'w'))
