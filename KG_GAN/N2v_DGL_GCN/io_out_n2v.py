import os
import json
import sys
import pickle as pkl
import numpy as np
import scipy.io as scio
'''
process the (GAE)'s output, convert to GAN's input
'''


DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet/KG-GAN'
Exp_NAME = 'Exp11-GCN'
Material_DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet'

# wnid file
seen_file = os.path.join(DATA_DIR, Exp_NAME, 'seen.txt')
unseen_file = os.path.join(DATA_DIR, Exp_NAME, 'unseen.txt')

embed_file = 'embed-GCN/100.pkl'
save_file = 'n2v-64.mat'

def readTxt(file_name):
    class_list = list()
    wnids = open(file_name, 'rU')
    try:
        for line in wnids:
            line = line[:-1]
            class_list.append(line)
    finally:
        wnids.close()
    print(len(class_list))
    return class_list

def load_class():
    seen = readTxt(seen_file)
    unseen = readTxt(unseen_file)
    return seen, unseen

###########################

def load_corresp():
    seen_corresp_file = os.path.join(DATA_DIR, Exp_NAME, 'g_seen_corresp.json')
    with open(seen_corresp_file) as fp:
        seen_corresp = json.load(fp)
    unseen_corresp_file = os.path.join(DATA_DIR, Exp_NAME, 'g_unseen_corresp.json')
    with open(unseen_corresp_file) as fp:
        unseen_corresp = json.load(fp)
    return seen_corresp, unseen_corresp

def extract_embed(seen_corresp, unseen_corresp):
    with open(os.path.join(DATA_DIR, Exp_NAME, embed_file), 'rb') as f:
        if sys.version_info > (3, 0):
            node_embed = pkl.load(f, encoding='latin1')
        else:
            node_embed = pkl.load(f)
    seen_embed = node_embed[seen_corresp]
    print("seen embed shape:", seen_embed.shape)
    unseen_embed = node_embed[unseen_corresp]
    print("unseen embed shape:", unseen_embed.shape)

    embed = np.vstack((seen_embed, unseen_embed))
    save_file = os.path.join(DATA_DIR, Exp_NAME, embed_file+'_su.pkl')
    with open(save_file, 'wb') as fp:
        pkl.dump(embed, fp)

    return seen_embed, unseen_embed


def store_embed(seen, unseen, seen_embed, unseen_embed, seen_corresp, unseen_corresp):


    # processing
    graph_nodes_file = os.path.join(DATA_DIR, Exp_NAME, 'g_cls_nodes.json')
    with open(graph_nodes_file) as fp:
        graph_nodes = json.load(fp)

    matcontent = scio.loadmat(os.path.join(Material_DATA_DIR, 'split.mat'))
    wnids = matcontent['allwnids'].squeeze().tolist()
    wnids = wnids[:2549]

    n2v = np.zeros((len(wnids), seen_embed.shape[1]), dtype=np.float)
    for i in range(len(wnids)):
        if wnids[i] in seen:
            graph_index = graph_nodes.index(wnids[i])
            corresp = seen_corresp.index(graph_index)
            n2v[i] = seen_embed[corresp]
        elif wnids[i] in unseen:
            graph_index = graph_nodes.index(wnids[i])
            corresp = unseen_corresp.index(graph_index)
            n2v[i] = unseen_embed[corresp]
        else:
            continue
    # save wnids together
    wnids_cell = np.empty((len(wnids), 1), dtype=np.object)
    # print(wnids_cell.shape)
    for i in range(len(wnids)):
        wnids_cell[i][0] = np.array(wnids[i])

    n2v_file = os.path.join(DATA_DIR, Exp_NAME, save_file)
    scio.savemat(n2v_file, {'n2v': n2v, 'wnids': wnids_cell})


if __name__ == '__main__':
    seen, unseen = load_class()
    seen_corresp, unseen_corresp = load_corresp()
    seen_embed, unseen_embed = extract_embed(seen_corresp, unseen_corresp)
    # process the format for inputting GAN
    # store_embed(seen, unseen, seen_embed, unseen_embed, seen_corresp, unseen_corresp)

