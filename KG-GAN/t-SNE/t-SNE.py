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


DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet/KG-GAN'
ImageNet_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet'
seen_corresp_file = os.path.join(DATA_DIR, 'Exp2', 'g_seen_corresp.json')
with open(seen_corresp_file) as fp:
    seen_corresp = json.load(fp)
unseen_corresp_file = os.path.join(DATA_DIR, 'Exp2', 'g_unseen_corresp.json')
with open(unseen_corresp_file) as fp:
    unseen_corresp = json.load(fp)


inv_wordn_file = os.path.join(DATA_DIR, 'Exp2', 'g_nodes.json')
with open(inv_wordn_file) as fp:
    graph_nodes = json.load(fp)



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

def readData():

    matcontent = scio.loadmat('/Users/geng/Desktop/ZSL_DATA/ImageNet/w2v.mat')
    wnids = matcontent['wnids'].squeeze().tolist()
    seen_index = np.array(ID2Index(wnids, os.path.join(DATA_DIR, 'Exp2', 'seen.txt')))
    unseen_index = np.array(ID2Index(wnids, os.path.join(DATA_DIR, 'Exp2', 'unseen.txt')))
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



def example1():
    digits = datasets.load_digits(n_class=6)
    X, y = digits.data, digits.target
    print(X.shape)
    print(y)
    n_samples, n_features = X.shape
    tsne = TSNE(n_components=2, init='pca', random_state=0)

    X_tsne = tsne.fit_transform(X)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=plt.cm.Set1(y[:]))
    plt.xticks([])
    plt.yticks([])
    plt.show()


def readData2():
    file_name = os.path.join(DATA_DIR, 'Exp2', 'embed/299.pkl')
    with open(file_name, 'rb') as f:
        if sys.version_info > (3, 0):
            embed = pkl.load(f, encoding='latin1')
        else:
            embed = pkl.load(f)
    print("embed shape:", embed.shape)
    seen_embed = embed[:249]
    name_list = list()
    for i in range(len(seen_embed)):
        node = graph_nodes[seen_corresp[i]]
        syn = getnode(node)
        syn_name = syn.lemma_names()[0]
        name_list.append(syn_name)
    unseen_embed = embed[249:]
    for i in range(len(unseen_embed)):
        node = graph_nodes[unseen_corresp[i]]
        syn = getnode(node)
        syn_name = syn.lemma_names()[0]
        name_list.append(syn_name)
    seen_label = np.array(0, dtype=int).repeat(seen_embed.shape[0])
    unseen_label = np.array(1, dtype=int).repeat(unseen_embed.shape[0])
    label = np.hstack((seen_label, unseen_label))
    return embed, label, name_list

def readData3():
    file_name = os.path.join(DATA_DIR, 'Exp2', 'embed-val20-test10/199.pkl')
    with open(file_name, 'rb') as f:
        if sys.version_info > (3, 0):
            embed = pkl.load(f, encoding='latin1')
        else:
            embed = pkl.load(f)
    print("embed shape:", embed.shape)
    seen_embed = embed[:249]
    name_list = list()
    for i in range(len(seen_embed)):
        node = graph_nodes[seen_corresp[i]]
        syn = getnode(node)
        syn_name = syn.lemma_names()[0]
        name_list.append(syn_name)

    unseen_embed = embed[249:]
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


def show(data, label, name):
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
    plt.title('t-SNE embedding of the digits (time %.2fs)' % (time() - t0))
    plt.savefig('plt_bar_chart.pdf', dpi=1500, bbox_inches='tight')
    plt.show()




if __name__ == '__main__':
    # main()
    # example1()

    # word2vec
    # embed, label = readData()
    # show(embed, label)
    # embed, label, name = readData3()
    # show(embed, label, name)

    embed, label, name = readData2()
    show(embed, label, name)

