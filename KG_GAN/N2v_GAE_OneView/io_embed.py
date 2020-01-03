# coding=gbk
# -*- coding: utf-8 -*-

import argparse
import re
import os
import json
import numpy as np
import pickle as pkl


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
Exp_NAME = 'Exp7'
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

def embed_text_file(vertices_f, word_vectors, get_vector, save_file, class_list):
    with open(vertices_f) as fp:
        vertices_list = json.load(fp)

    all_feats = []

    has = 0
    cnt_missed = 0
    missed_list = []
    for i, vertex in enumerate(vertices_list):
        class_name = wnid_word[vertex].lower()
        # class_name = vertex.lower()
        if i % 500 == 0:
            print('%d / %d : %s' % (i, len(vertices_list), class_name))
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

    for each in missed_list:
        # print(each)
        if each in class_list:
            print(each)
    print('does not have semantic embedding: ', cnt_missed, 'has: ', has)

    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
        print('## Make Directory: %s' % save_file)
    with open(save_file, 'wb') as fp:
        pkl.dump(all_feats, fp)
    print('save to : %s' % save_file)


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

    vertices_file = os.path.join(DATA_DIR, Exp_NAME, 'g_nodes.json')

    # load text
    # wnid_word = dict()
    wnid_word = readTxt(os.path.join(Material_DATA_DIR, 'materials', 'words-all.txt'))
    att_word = readTxt(os.path.join(DATA_DIR, Exp_NAME, 'attribute.txt'))
    # with open(os.path.join(Material_DATA_DIR, 'materials', 'words.txt'), 'rb') as fp:
    #     for line in fp.readlines():
    #         wn, name = line.split("\t")
    #         wnid_word[wn] = name.strip()
    # att_word = dict()
    # with open(os.path.join(DATA_DIR, Exp_NAME, 'attribute.txt'), 'rb') as fp:
    #     for line in fp.readlines():
    #         aid, name = line.split("\t")
    #         att_word[aid] = name.strip()
    wnid_word.update(att_word)

    # load class
    class_list = list()
    with open(os.path.join(DATA_DIR, Exp_NAME, 'seen.txt')) as fp:
        for line in fp.readlines():
            class_list.append(line.strip())
    with open(os.path.join(DATA_DIR, Exp_NAME, 'unseen.txt')) as fp:
        for line in fp.readlines():
            class_list.append(line.strip())
    print("class num:", len(class_list))

    save_file = os.path.join(DATA_DIR, Exp_NAME, 'g_embed.pkl')

    word_vectors = get_glove_dict('/Users/geng/Desktop/ZSL_DATA/ImageNet/Baseline/GCNZ/materials')
    get_vector = glove_google

    print('obtain semantic word embedding', save_file)
    embed_text_file(vertices_file, word_vectors, get_vector, save_file, class_list)