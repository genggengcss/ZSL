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
Material_DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet'

Exp_NAME = 'Exp2'

# wnid file
seen_file = os.path.join(DATA_DIR, Exp_NAME, 'seen.txt')
unseen_file = os.path.join(DATA_DIR, Exp_NAME, 'unseen.txt')



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

def load_corresp(path):
    seen_corresp_file = os.path.join(path, 'g_seen_corresp.json')
    with open(seen_corresp_file) as fp:
        seen_corresp = json.load(fp)
    unseen_corresp_file = os.path.join(path, 'g_unseen_corresp.json')
    with open(unseen_corresp_file) as fp:
        unseen_corresp = json.load(fp)
    return seen_corresp, unseen_corresp

def extract_embed(type, embed_file):
    if type == 'cls':
        path = os.path.join(DATA_DIR, Exp_NAME, 'cls')
    else:
        path = os.path.join(DATA_DIR, Exp_NAME, 'att')

    seen_corresp, unseen_corresp = load_corresp(path)

    with open(os.path.join(path, 'embed', embed_file), 'rb') as f:
        if sys.version_info > (3, 0):
            node_embed = pkl.load(f, encoding='latin1')
        else:
            node_embed = pkl.load(f)
    seen_embed = node_embed[seen_corresp]
    print("seen embed shape:", seen_embed.shape)
    unseen_embed = node_embed[unseen_corresp]
    print("unseen embed shape:", unseen_embed.shape)

    embed = np.vstack((seen_embed, unseen_embed))
    save_file = os.path.join(path, 'embed', embed_file+'_su.pkl')
    with open(save_file, 'wb') as fp:
        pkl.dump(embed, fp)



    return seen_embed, unseen_embed, seen_corresp, unseen_corresp



def prepare_n2v():
    # processing
    cls_graph_nodes_file = os.path.join(DATA_DIR, Exp_NAME, 'cls/g_nodes.json')
    with open(cls_graph_nodes_file) as fp:
        cls_graph_nodes = json.load(fp)
    att_graph_nodes_file = os.path.join(DATA_DIR, Exp_NAME, 'att/g_nodes.json')
    with open(att_graph_nodes_file) as fp:
        att_graph_nodes = json.load(fp)

    matcontent = scio.loadmat(os.path.join(Material_DATA_DIR, 'w2v.mat'))
    wnids = matcontent['wnids'].squeeze().tolist()
    wnids = wnids[:2549]

    n2v = np.zeros((len(wnids), (cls_seen_embed.shape[1] + att_seen_embed.shape[1])), dtype=np.float)
    print(n2v.shape)
    for i in range(len(wnids)):
        if wnids[i] in seen:
            att_graph_index = att_graph_nodes.index(wnids[i])
            att_corresp = att_seen_corresp.index(att_graph_index)

            cls_graph_index = cls_graph_nodes.index(wnids[i])
            cls_corresp = cls_seen_corresp.index(cls_graph_index)
            n2v[i] = np.hstack((cls_seen_embed[cls_corresp], att_seen_embed[att_corresp]))
        elif wnids[i] in unseen:
            att_graph_index = att_graph_nodes.index(wnids[i])
            att_corresp = att_unseen_corresp.index(att_graph_index)

            cls_graph_index = cls_graph_nodes.index(wnids[i])
            cls_corresp = cls_unseen_corresp.index(cls_graph_index)

            n2v[i] = np.hstack((cls_unseen_embed[cls_corresp], att_unseen_embed[att_corresp]))
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

    cls_seen_embed, cls_unseen_embed, cls_seen_corresp, cls_unseen_corresp = extract_embed('cls', '500.pkl')
    att_seen_embed, att_unseen_embed, att_seen_corresp, att_unseen_corresp= extract_embed('att', '1850.pkl')

    save_file = 'n2v.mat'
    # process the format for inputting GAN
    prepare_n2v()



