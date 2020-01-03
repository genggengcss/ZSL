# coding=gbk
# -*- coding: utf-8 -*-

import argparse
import re
import os
import json
import numpy as np
import pickle as pkl
import torch

"""
for extracting word embedding yourself, please download pretrained model from one of the following links.
"""
'''
original: obtain_word_embedding.py, for get embedding vector of vertives 
'''
url = {'glove': 'http://nlp.stanford.edu/data/glove.6B.zip',
       'google': 'https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing',
       'fasttext': 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip'}

WORD_VEC_LEN = 300

animal_wnid = 'n00015388'  # for extracting animal subset
DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet/KG-GAN'
Exp_NAME = 'Exp12-HGCN'
Material_DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet'

def readTxt(file_name):
    id_text = dict()
    texts = open(file_name, 'rU')
    try:
        for line in texts:
            line = line[:-1]
            lines = line.split('\t')
            id_text[lines[0]] = lines[1]
    finally:
        texts.close()
    return id_text

def embed_text_file(nodes, word_vectors, get_vector):

    all_feats = []

    has = 0
    cnt_missed = 0
    missed_list = []

    for i, vertex in enumerate(nodes):
        class_name = wnid_word[vertex].lower()
        # class_name = vertex.lower()
        if i % 500 == 0:
            print('%d / %d : %s' % (i, len(cls_nodes), class_name))
        feat = np.zeros(WORD_VEC_LEN)

        options = class_name.split(',')
        cnt_word = 0
        for option in options:
            now_feat = get_embedding(option.strip(), word_vectors, get_vector)
            if np.abs(now_feat.sum()) > 0:
                cnt_word += 1
                feat += now_feat
        if cnt_word > 0:
            feat = feat / cnt_word

        if np.abs(feat.sum()) == 0:
            # print('cannot find word ' + class_name)
            cnt_missed = cnt_missed + 1
            missed_list.append(class_name)
        else:
            has += 1
            feat = feat / (np.linalg.norm(feat) + 1e-6)

        all_feats.append(feat)

    all_feats = np.array(all_feats)


    print('does not have semantic embedding: ', cnt_missed, 'has: ', has)

    return all_feats


def get_embedding(entity_str, word_vectors, get_vector):
    try:
        feat = get_vector(word_vectors, entity_str)
        return feat
    except:
        feat = np.zeros(WORD_VEC_LEN)

    str_set = filter(None, re.split("[ \-_]+", entity_str))
    str_set = list(str_set)
    cnt_word = 0
    for i in range(len(str_set)):
        temp_str = str_set[i]
        try:
            now_feat = get_vector(word_vectors, temp_str)
            feat = feat + now_feat
            cnt_word = cnt_word + 1
        except:
            continue

    if cnt_word > 0:
        feat = feat / cnt_word
    return feat


def get_glove_dict(txt_dir):
    print('load glove word embedding')
    txt_file = os.path.join(txt_dir, 'glove.6B.300d.txt')
    word_dict = {}
    feat = np.zeros(WORD_VEC_LEN)
    with open(txt_file) as fp:
        for line in fp:
            words = line.split()
            assert len(words) - 1 == WORD_VEC_LEN
            for i in range(WORD_VEC_LEN):
                feat[i] = float(words[i+1])
            feat = np.array(feat)
            word_dict[words[0]] = feat
    print('loaded to dict!')
    return word_dict


def glove_google(word_vectors, word):
    return word_vectors[word]


def fasttext(word_vectors, word):
    return word_vectors.get_word_vector(word)


# transform the vectors of the graph (invdict_wordntext.json) to vectors
# save the vectors in e.g., glove_word2vec_wordnet.pkl
# the order of the vectors are consistent with the wordntext
if __name__ == '__main__':

    with open(os.path.join(DATA_DIR, Exp_NAME, 'g_cls_nodes.json')) as fp:
        cls_nodes = json.load(fp)
    with open(os.path.join(DATA_DIR, Exp_NAME, 'g_att_nodes.json')) as fp:
        att_nodes = json.load(fp)
    # load text
    wnid_word = readTxt(os.path.join(Material_DATA_DIR, 'materials', 'words-all.txt'))
    att_word = readTxt(os.path.join(DATA_DIR, Exp_NAME, 'attribute.txt'))

    wnid_word.update(att_word)



    save_file = os.path.join(DATA_DIR, Exp_NAME, 'g_embed.pkl')

    word_vectors = get_glove_dict('/Users/geng/Desktop/ZSL_DATA/ImageNet/Baseline/GCNZ/materials')
    get_vector = glove_google

    print('obtain semantic word embedding', save_file)
    cls_embed = embed_text_file(cls_nodes, word_vectors, get_vector)
    att_embed = embed_text_file(att_nodes, word_vectors, get_vector)

    embed_dict = {'cls': torch.Tensor(cls_embed), 'att':torch.Tensor(att_embed)}
    with open(save_file, 'wb') as fp:
        pkl.dump(embed_dict, fp)
    print('Save Graph structure to: ', save_file)