import argparse
import json
import os
import sys
import torch

from nltk.corpus import wordnet as wn

from collections import Counter
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


split_file = '/Users/geng/Desktop/ImageNet/split.mat'


wnid_file_name = '/Users/geng/Desktop/ImageNet/materials/wnids.txt'
wnid_f_file_name = '/Users/geng/Desktop/ImageNet/materials/wnids-no_w2v.txt'
word_file_name = '/Users/geng/Desktop/ImageNet/materials/words.txt'

wnids = readWnidTxt(wnid_file_name)
words = readWordTxt(word_file_name)
wnids_filter = readWnidTxt(wnid_f_file_name)


wnids_cell = np.empty((len(wnids), 1), dtype=np.object)
for i in range(len(wnids)):
    wnids_cell[i][0] = np.array(wnids[i])

words_cell = np.empty((len(words), 1), dtype=np.object)
for i in range(len(wnids)):
    words_cell[i][0] = np.array(words[i])

# wnid_1k_matrix = np.empty((len(wnids[:1000]), 1), dtype=np.int16)
# wnid_2hops_matrix = np.empty((1509, 1), dtype=np.int16)
# wnid_3hops_matrix = np.empty((6169, 1), dtype=np.int16)
# wnid_rest_matrix = np.empty((12667, 1), dtype=np.int16)

def save_matrix(wnids, leng, start, end):
    j = 0
    matrix = np.empty((leng, 1), dtype=np.int16)
    for i in range(start, end):
        wnid = wnids[i]
        if wnid in wnids_filter:
            continue
        else:
            matrix[j][0] = i + 1
            j += 1
    print("j:", j)
    return matrix

wnid_1k_matrix = save_matrix(wnids, 1000, 0, 1000)
wnid_2hops_matrix = save_matrix(wnids, 1509, 1000, 2549)
wnid_3hops_matrix = save_matrix(wnids, 6169, 2549, 8860)
wnid_rest_matrix = save_matrix(wnids, 12667, 8860, 21842)

no_list = [1041, 1055, 1061, 1093, 1131, 1239, 1259, 1288, 1346, 1358, 1440, 1456, 1480, 1493, 1506, 1524, 1566, 1608, 1677, 1728, 1736, 1813, 1920, 1951, 1967, 1998, 2034, 2169, 2175, 2237, 2296, 2301, 2340, 2352, 2375, 2383, 2464, 2470, 2496, 2539, 2594, 2595, 2630, 2849, 2911, 3060, 3082, 3087, 3161, 3285, 3321, 3331, 3344, 3367, 3403, 3414, 3434, 3441, 3442, 3464, 3473, 3474, 3541, 3542, 3545, 3571, 3638, 3669, 3677, 3696, 3718, 3897, 3905, 3968, 3982, 3989, 4130, 4197, 4216, 4331, 4368, 4381, 4426, 4433, 4484, 4630, 4663, 4688, 4706, 4730, 4738, 4753, 4763, 4814, 4845, 4866, 4872, 4959, 4968, 5196, 5230, 5258, 5307, 5391, 5436, 5440, 5490, 5494, 5526, 5536, 5541, 5555, 5598, 5692, 5751, 5803, 5813, 5878, 5916, 5923, 6108, 6124, 6169, 6170, 6206, 6211, 6234, 6250, 6353, 6379, 6428, 6442, 6513, 6547, 6582, 6617, 6618, 6625, 6649, 6655, 6755, 6775, 6831, 6875, 6888, 6920, 7002, 7006, 7139, 7286, 7298, 7361, 7399, 7431, 7433, 7521, 7555, 7582, 7602, 7640, 7740, 7741, 7815, 7863, 7921, 7949, 7952, 7962, 7972, 7999, 8155, 8235, 8301, 8420, 8554, 8557, 8779, 8803, 8809, 8819, 8835, 8848, 8866, 8883, 8914, 9025, 9082, 9152, 9158, 9169, 9174, 9183, 9195, 9246, 9304, 9311, 9324, 9390, 9394, 9430, 9524, 9560, 9561, 9570, 9584, 9596, 9599, 9622, 9631, 9645, 9708, 9795, 9798, 9848, 10049, 10074, 10084, 10112, 10129, 10194, 10197, 10206, 10237, 10286, 10304, 10426, 10453, 10454, 10476, 10478, 10511, 10569, 10655, 10697, 10754, 10758, 10823, 10851, 10888, 10933, 10975, 11049, 11093, 11105, 11112, 11177, 11239, 11334, 11377, 11389, 11472, 11485, 11494, 11496, 11561, 11591, 11640, 11643, 11710, 11745, 11748, 11763, 11776, 11808, 11814, 11865, 11876, 11901, 11947, 11985, 12025, 12058, 12100, 12203, 12257, 12295, 12385, 12421, 12445, 12454, 12561, 12586, 12589, 12631, 12671, 12697, 12700, 12708, 12795, 12850, 12883, 12886, 13021, 13211, 13267, 13342, 13347, 13387, 13422, 13453, 13606, 13645, 13651, 13761, 13815, 13821, 13829, 14073, 14121, 14126, 14137, 14155, 14213, 14251, 14255, 14265, 14297, 14431, 14468, 14506, 14511, 14588, 14606, 14633, 14648, 14710, 14731, 14744, 14758, 14787, 14816, 14896, 14908, 14929, 14943, 14993, 15027, 15061, 15073, 15076, 15100, 15130, 15148, 15156, 15193, 15211, 15301, 15331, 15342, 15345, 15374, 15385, 15389, 15417, 15600, 15674, 15707, 15827, 15836, 15846, 15959, 15965, 16224, 16307, 16338, 16395, 16396, 16456, 16534, 16539, 16557, 16619, 16645, 16772, 16810, 16850, 16905, 17015, 17027, 17129, 17135, 17220, 17317, 17338, 17458, 17484, 17507, 17551, 17581, 17589, 17615, 17720, 17757, 17758, 17786, 17798, 17803, 17813, 17911, 18076, 18143, 18153, 18251, 18252, 18318, 18371, 18428, 18483, 18520, 18541, 18555, 18562, 18625, 18635, 18639, 18651, 18715, 18836, 18857, 18938, 18946, 18973, 18980, 18997, 19000, 19019, 19020, 19133, 19164, 19206, 19212, 19262, 19270, 19272, 19277, 19309, 19331, 19341, 19458, 19467, 19554, 19566, 19728, 19750, 19788, 19806, 19849, 19860, 19900, 19921, 20000, 20015, 20017, 20061, 20100, 20132, 20136, 20236, 20272, 20277, 20290, 20313, 20329, 20349, 20432, 20467, 20564, 20603, 20609, 20687, 20701, 20793, 20877, 20924, 20932, 20950, 20966, 20967, 20976, 21010, 21019, 21063, 21104, 21167, 21199, 21233, 21314, 21334, 21406, 21445, 21449, 21529, 21601, 21695, 21780, 21785, 21802]
no_list_matrix = np.empty((len(no_list), 1), dtype=np.int16)
for i in range(len(no_list)):
    # print(wnids[i])
    index = no_list[i]
    no_list_matrix[i][0] = index

# print(wnid_1k_matrix.shape)
# print(wnid_2hops_matrix.shape)
# print(wnid_3hops_matrix.shape)
# print(wnid_rest_matrix.shape)

scio.savemat(split_file, {'seen': wnid_1k_matrix, 'hops2': wnid_2hops_matrix, 'hops3': wnid_3hops_matrix,
                          'rest': wnid_rest_matrix, 'allwnids': wnids_cell, 'allwords':words_cell, 'no_w2v_index': no_list_matrix})


