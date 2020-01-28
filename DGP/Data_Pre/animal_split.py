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

DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet/Baseline/DGP'
Material_DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet'

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

def extractAnimalSub(wnids, animal):
    wnids_sub = list()
    for wnid in wnids:
        if wnid in animal:
            wnids_sub.append(wnid)
    print("animal:", len(wnids_sub))
    return wnids_sub

def keepValid(valid_set, wnids):
    wnids_valid = list()
    for wnid in wnids:
        if wnid in valid_set:
            wnids_valid.append(wnid)
    print("valid set:", len(wnids_valid))
    return wnids_valid

# ### 1. extract animal subset and keep each node have w2v
# animal nodes set
animal_wnids_file = os.path.join(Material_DATA_DIR, 'materials', 'wnids-fa11misc.txt')
w2v_valid_class_file = os.path.join(Material_DATA_DIR, 'materials', 'split-filter/all.txt')
# load all imagenet wnids/nodes
imagenet_xml_file = os.path.join(DATA_DIR, 'materials', 'imagenet-xml-wnids.json')
# store all animal subset wnids
imagenet_xml_animal_file = os.path.join(DATA_DIR, 'materials', 'imagenet-xml-wnids-food.json')



if __name__ == '__main__':

    # imagenet all wnids, number: 32295, prepare for imagenet graph
    graph_wnids = json.load(open(imagenet_xml_file, 'r'))
    print(len(graph_wnids))

    # load animal wnids: 3969
    animal_wnids = readTxtFile(animal_wnids_file)
    # load w2v-valid wnids: 21345
    w2v_valid_classes = readTxtFile(w2v_valid_class_file)

    # extract imagenet animal subset and keep each nodes have word embedding
    animal_sub = extractAnimalSub(graph_wnids, animal_wnids)
    animal_sub = keepValid(w2v_valid_classes, animal_sub)
    json.dump(animal_sub, open(imagenet_xml_animal_file, 'w'))















