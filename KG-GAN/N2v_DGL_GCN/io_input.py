# -*- coding: utf-8 -*-
import os
import json
import pickle as pkl
import numpy as np
from nltk.corpus import wordnet as wn
import xml.etree.ElementTree as ET

import scipy.io as scio

'''
Prepare the graph inputting for GAE
'''
animal_wnid = 'n00015388'  # for extracting animal subset
DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet/KG-GAN'
Exp_NAME = 'Exp11-GCN'
Material_DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet'

# wnid file
seen_file = os.path.join(DATA_DIR, Exp_NAME, 'seen.txt')
seen_lbl_file = os.path.join(DATA_DIR, Exp_NAME, 'seen_label.txt')

unseen_file = os.path.join(DATA_DIR, Exp_NAME, 'unseen.txt')
unseen_lbl_file = os.path.join(DATA_DIR, Exp_NAME, 'unseen_label.txt')

all_wnids_file = os.path.join(Material_DATA_DIR, 'materials', 'split-filter/all.txt')
# wordnet graph file
wordnet_structure_file = os.path.join(Material_DATA_DIR, 'materials', 'structure_released.xml')


def readTxt2(file_name):
    class_dict = dict()
    lines = open(file_name, 'rU')
    try:
        for line in lines:
            class_lal = line[:-1].split('\t')
            class_dict[class_lal[0]] = class_lal[1]
    finally:
        lines.close()
    # print(len(class_dict))
    print(class_dict)
    return class_dict

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
    seen = readTxt(seen_file)  # seen classes 16
    unseen = readTxt(unseen_file)  # unseen classes 56
    seen_label_dict = readTxt2(seen_lbl_file)  # seen classes 16
    unseen_label_dict = readTxt2(unseen_lbl_file)  # unseen classes 56
    all_valid_wnids = readTxt(all_wnids_file)  # all valid class set
    return seen, unseen, seen_label_dict, unseen_label_dict, all_valid_wnids

# filter the invalid classes/nodes
def filter_graph(vertex_list, edge_list, valid_classes):
    stop_classes = list()
    # filter vertex list
    new_vertices = list()
    for node in vertex_list:
        if node in valid_classes:
            new_vertices.append(node)
        else:
            stop_classes.append(node)
    print("vertices:", len(new_vertices))

    # filter edge list
    new_edges = list()
    for edge in edge_list:
        node1 = edge[0]
        node2 = edge[1]
        if node1 in stop_classes or node2 in stop_classes:
            continue
        else:
            new_edges.append(edge)
    print("edges:", len(new_edges))
    return new_vertices, new_edges

def add_edge_dfs(node):
    edges = []
    vertices = [node.attrib['wnid']]
    if len(node) == 0:
        return vertices, edges
    for child in node:
        if child.tag != 'synset':
            print(child.tag)

        edges.append((node.attrib['wnid'], child.attrib['wnid']))
        child_ver, child_edge = add_edge_dfs(child)
        edges.extend(child_edge)
        vertices.extend(child_ver)
    return vertices, edges

# wordNet graph: get the vertices and edges of graph
def prepare_graph(valid_classes):
    tree = ET.parse(wordnet_structure_file)
    root = tree.getroot()
    # select the animal subset start
    for sy in root.findall('synset'):  # find synset tag
        for ssy in sy.findall('synset'):  # deeper layer
            # print("wnid:", ssy.get('wnid'))
            if ssy.get('wnid') == animal_wnid:  # get tag's attribute value(wnid), 'n00015388' represents 'animal'
                vertex_list, edge_list = add_edge_dfs(ssy)  # animal node -> the root node
            else:
                continue
    # select the animal subset end
    vertex_list = list(set(vertex_list))  # remove the repeat node
    print('Unique Vertex #: %d, Edge #: %d', (len(vertex_list), len(edge_list)))

    # filter the non-existing nodes and corresponding edges, make each node have initial word embedding
    f_vertices, f_edges = filter_graph(vertex_list, edge_list, valid_classes)

    return f_vertices, f_edges




def convert_graph(vertices, edges):

    # save graph nodes/wnids/classes
    graph_nodes_file = os.path.join(DATA_DIR, Exp_NAME, 'g_nodes.json')
    with open(graph_nodes_file, 'w') as fp:
        json.dump(vertices, fp)
    print('Save graph node in wnid to %s' % graph_nodes_file)



    # save graph edge pair
    edge_pairs = np.empty((len(edges), 2), dtype=np.object)
    for i in range(len(edges)):
        edge_pairs[i][0] = edges[i][0]
        edge_pairs[i][1] = edges[i][1]

    edge_file = os.path.join(DATA_DIR, Exp_NAME, 'edges.pkl')
    with open(edge_file, 'wb') as fp:
        pkl.dump(edge_pairs, fp)
    print('Save Graph structure to: ', edge_file)
    # print(graph)




def save_corresp(vertices, seen, unseen):
    # save classes's index in graph
    seen_idx_in_graph = list()
    unseen_idx_in_graph = list()
    for i in range(len(vertices)):
        if vertices[i] in seen:
            seen_idx_in_graph.append(i)
        elif vertices[i] in unseen:
            unseen_idx_in_graph.append(i)
        else:
            continue
    seen_corresp_file = os.path.join(DATA_DIR, Exp_NAME, 'g_seen_corresp.json')
    with open(seen_corresp_file, 'w') as fp:
        json.dump(seen_idx_in_graph, fp)
    print('Save seen index in graph to %s' % seen_corresp_file)

    unseen_corresp_file = os.path.join(DATA_DIR, Exp_NAME, 'g_unseen_corresp.json')
    with open(unseen_corresp_file, 'w') as fp:
        json.dump(unseen_idx_in_graph, fp)
    print('Save unseen index in graph to %s' % unseen_corresp_file)


def save_nodes(vertices, seen_label_dict, unseen_label_dict):

    matcontent = scio.loadmat(os.path.join(Material_DATA_DIR, 'w2v.mat'))
    wnids = matcontent['wnids'].squeeze().tolist()
    w2v = matcontent['w2v']


    # nodes = np.empty((len(vertices), (w2v.shape[1]+2)), dtype=np.str)
    node_file = os.path.join(DATA_DIR, Exp_NAME, 'nodes.txt')
    wr_fp = open(node_file, 'w')
    for vertex in vertices:

        wr_fp.write('%s,' % (vertex))
        vector = w2v[wnids.index(vertex)]
        for i in range(vector.shape[0]):
            wr_fp.write('%f,' % (vector[i]))
        # print(vertices[i])
        if vertex in seen_label_dict:
            wr_fp.write('%s\n' % (seen_label_dict[vertex]))
        elif vertex in unseen_label_dict:
            wr_fp.write('%s\n' % (unseen_label_dict[vertex]))
        else:
            wr_fp.write('%s\n' % ('other'))
    wr_fp.close()

    print('Save Graph structure to: ', node_file)


if __name__ == '__main__':
    seen, unseen, seen_label_dict, unseen_label_dict, all_valid_wnids = load_class()

    vertices, edges = prepare_graph(all_valid_wnids)
    convert_graph(vertices, edges)   # save edges

    save_corresp(vertices, seen, unseen)
    save_nodes(vertices, seen_label_dict, unseen_label_dict)





