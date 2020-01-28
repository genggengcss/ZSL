import argparse
import json
import os
import sys
import torch

from nltk.corpus import wordnet as wn



'''
input: imagenet-xml-wnids.json
output: imagenet-xml-animals.json
function: extract animal subset, and keep each animal class have word embedding 
'''

# DATA_DIR = '/home/gyx/ZSL/data/ImageNet/Baseline/DGP'
DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet/Baseline/DGP'
Material_DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet'
# Material_DATA_DIR = '/home/gyx/ZSL/data/ImageNet'

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

def extractSub(wnids, subset, file_name):

    wr_fp = open(file_name, 'w')
    for wnid in wnids:
        if wnid in subset:
            wr_fp.write('%s\n' % (wnid))
    wr_fp.close()







if __name__ == '__main__':

    animal_wnids_file = os.path.join(Material_DATA_DIR, 'materials', 'wnids-artifact.txt')

    ori_seen_file = os.path.join(Material_DATA_DIR, 'materials', 'split-filter/1k.txt')
    ori_unseen_file = os.path.join(Material_DATA_DIR, 'materials', 'split-filter/2-hops.txt')


    subset_wnids = readTxtFile(animal_wnids_file)


    ori_seen_wnids = readTxtFile(ori_seen_file)
    ori_unseen_wnids = readTxtFile(ori_unseen_file)

    type_name = animal_wnids_file[animal_wnids_file.index('-'): animal_wnids_file.index('.')]
    path = os.path.join(Material_DATA_DIR, 'materials', 'split' + type_name)
    if not os.path.exists(path):
        os.makedirs(path)

    file_name = os.path.join(path, 'seen.txt')
    extractSub(ori_seen_wnids, subset_wnids, file_name)

    file_name = os.path.join(path, 'unseen.txt')
    extractSub(ori_unseen_wnids, subset_wnids, file_name)














