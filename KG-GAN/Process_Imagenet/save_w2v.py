import argparse
import json
import os
import sys
import torch

from nltk.corpus import wordnet as wn

from collections import Counter
import h5py
import torch
# from sklearn import preprocessing
import sys
# from sklearn.cluster import KMeans

import numpy as np

import scipy.io as scio

def readWordTxt(filename):
    wnids = open(filename, 'rU')
    words_list = list()
    try:
        for line in wnids:
            line = line[:-1]
            lines = line.split("\t")
            words_list.append(lines[1])
    finally:
        wnids.close()
    print(len(words_list))
    return words_list

def readWnidTxt(filename):
    wnids = open(filename, 'rU')
    wnid_list = list()
    try:
        for line in wnids:
            line = line[:-1]
            wnid_list.append(line)
    finally:
        wnids.close()
    print(len(wnid_list))
    return wnid_list

w2v_file = '/Users/geng/Desktop/ImageNet/w2v.mat'


wnid_file_name = '/Users/geng/Desktop/ImageNet/materials/wnids.txt'
word_file_name = '/Users/geng/Desktop/ImageNet/materials/words.txt'
ori_w2v_file = '/Users/geng/Desktop/ImageNet/w2v1.mat'


wnids = readWnidTxt(wnid_file_name)
words = readWordTxt(word_file_name)


wnids_cell = np.empty((len(wnids), 1), dtype=np.object)
for i in range(len(wnids)):
    wnids_cell[i][0] = np.array(wnids[i])

words_cell = np.empty((len(words), 1), dtype=np.object)
for i in range(len(wnids)):
    words_cell[i][0] = np.array(words[i])

# matcontent = h5py.File(w2v_file, 'r')
# w2v = np.array(matcontent['w2v']).T
# print(w2v)

matcontent = scio.loadmat(ori_w2v_file)
w2v = matcontent['w2v']
print(w2v.shape)


scio.savemat(w2v_file, {'wnids': wnids_cell, 'w2v': w2v})


