# coding='utf-8'
"""t-SNE对手写数字进行可视化"""
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE
import scipy.io as scio
import os
import sys
import pickle as pkl
import json
from nltk.corpus import wordnet as wn

def getnode(x):
    return wn.synset_from_pos_and_offset('n', int(x[1:]))
def getwnid(u):
    s = str(u.offset())
    return 'n' + (8 - len(s)) * '0' + s


'''
Exp: 550.pkl
'''
DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet/KG-GAN'
ImageNet_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet'





def readTxt(file_name):
    class_list = list()
    wnids = open(file_name, 'rU')
    try:
        for line in wnids:
            class_list.append(line[:-1])
    finally:
        wnids.close()
    return class_list

def ID2Index(wnids, class_file):
    class_wnids = readTxt(class_file)
    index_list = list()
    for wnid in class_wnids:
        idx = wnids.index(wnid)
        index_list.append(idx)
    return index_list

def readDataW():

    matcontent = scio.loadmat('/Users/geng/Desktop/ZSL_DATA/ImageNet/w2v.mat')
    wnids = matcontent['wnids'].squeeze().tolist()
    seen_index = np.array(ID2Index(wnids, os.path.join(DATA_DIR, EXP_NAME, 'seen.txt')))
    unseen_index = np.array(ID2Index(wnids, os.path.join(DATA_DIR, EXP_NAME, 'unseen.txt')))
    # print(seen_index)
    w2v = matcontent['w2v']
    seen_embed = w2v[seen_index]
    unseen_embed = w2v[unseen_index]
    embed = np.vstack((seen_embed, unseen_embed))
    seen_label = np.array(0, dtype=int).repeat(seen_embed.shape[0])
    unseen_label = np.array(1, dtype=int).repeat(unseen_embed.shape[0])
    label = np.hstack((seen_label, unseen_label))
    # print(embed.shape)
    # print(label.shape)
    # print(label)
    return embed, label


def readDataNonName(filename):
    file = os.path.join(DATA_DIR, EXP_NAME, filename)
    with open(file, 'rb') as f:
        if sys.version_info > (3, 0):
            embed = pkl.load(f, encoding='latin1')
        else:
            embed = pkl.load(f)
    print("embed shape:", embed.shape)
    seen_embed = embed[:seen_length]

    unseen_embed = embed[seen_length:]

    seen_label = np.array(0, dtype=int).repeat(seen_embed.shape[0])
    unseen_label = np.array(1, dtype=int).repeat(unseen_embed.shape[0])
    label = np.hstack((seen_label, unseen_label))
    return embed, label

def readDataWithName(filename):
    file = os.path.join(DATA_DIR, EXP_NAME, filename)
    with open(file, 'rb') as f:
        if sys.version_info > (3, 0):
            embed = pkl.load(f, encoding='latin1')
        else:
            embed = pkl.load(f)
    print("embed shape:", embed.shape)
    seen_embed = embed[:seen_length]
    name_list = list()
    for i in range(len(seen_embed)):
        node = graph_nodes[seen_corresp[i]]
        syn = getnode(node)
        syn_name = syn.lemma_names()[0]
        name_list.append(syn_name)

    unseen_embed = embed[seen_length:]
    for i in range(len(unseen_embed)):
        node = graph_nodes[unseen_corresp[i]]
        syn = getnode(node)
        syn_name = syn.lemma_names()[0]
        name_list.append(syn_name)
    seen_label = np.array(0, dtype=int).repeat(seen_embed.shape[0])
    unseen_label = np.array(1, dtype=int).repeat(unseen_embed.shape[0])
    # print(seen_label.shape)
    # print(unseen_label.shape)
    label = np.hstack((seen_label, unseen_label))


    return embed, label, name_list

def showNonName(data, label, filename):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    data_tsne = tsne.fit_transform(data)
    plt.figure(figsize=(8, 8))
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=plt.cm.Set1(label[:]))

    plt.xticks([])
    plt.yticks([])
    plt.title('t-SNE embedding of the digits (time %.2fs), %s' % ((time() - t0), filename))
    # plt.savefig('plt_bar_chart.pdf', dpi=1500, bbox_inches='tight')
    plt.show()


def showWithName(data, label, name, filename):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    data_tsne = tsne.fit_transform(data)
    plt.figure(figsize=(8, 8))
    # plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=plt.cm.Set1(label[:]))

    x_min, x_max = data_tsne.min(0), data_tsne.max(0)
    data_norm = (data_tsne - x_min) / (x_max - x_min)  # 归一化
    for i in range(data_norm.shape[0]):
        plt.text(data_norm[i, 0], data_norm[i, 1], str(name[i]), color=plt.cm.Set1(label[i]),
             fontdict={'weight': 'light', 'size': 6})
    plt.xticks([])
    plt.yticks([])
    plt.title('t-SNE embedding of the digits (time %.2fs), %s' % ((time() - t0), filename))
    out_name = filename.split('/')[-1]+'.pdf'
    # plt.savefig(out_name, dpi=1500, bbox_inches='tight')
    plt.show()


def load_data():
    seen_corresp_file = os.path.join(DATA_DIR, EXP_NAME, type_name, 'g_seen_corresp.json')
    with open(seen_corresp_file) as fp:
        seen_corresp = json.load(fp)
        seen_length = len(seen_corresp)
    unseen_corresp_file = os.path.join(DATA_DIR, EXP_NAME, type_name, 'g_unseen_corresp.json')
    with open(unseen_corresp_file) as fp:
        unseen_corresp = json.load(fp)

    inv_wordn_file = os.path.join(DATA_DIR, EXP_NAME, type_name, 'g_nodes.json')
    with open(inv_wordn_file) as fp:
        graph_nodes = json.load(fp)
    return seen_corresp, seen_length, unseen_corresp, graph_nodes

if __name__ == '__main__':


    # word2vec
    # embed, label = readDataW()
    # showNonName(embed, label, '')

    EXP_NAME = 'Exp10'
    type_name = 'att'
    seen_corresp, seen_length, unseen_corresp, graph_nodes = load_data()




    # graph embedding
    filename = os.path.join(type_name, 'embed/1000.pkl_su.pkl')
    # embed, label = readDataNonName(filename)
    # showNonName(embed, label, filename)
    embed, label, name = readDataWithName(filename)
    showWithName(embed, label, name, filename)


