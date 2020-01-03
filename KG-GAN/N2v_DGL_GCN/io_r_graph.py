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
Exp_NAME = 'Exp12-HGCN'
Material_DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet'

# wnid files
seen_file = os.path.join(DATA_DIR, Exp_NAME, 'seen.txt')
unseen_file = os.path.join(DATA_DIR, Exp_NAME, 'unseen.txt')
all_wnids_file = os.path.join(Material_DATA_DIR, 'materials', 'split-filter/all.txt')
# wordnet graph file
wordnet_structure_file = os.path.join(Material_DATA_DIR, 'materials', 'structure_released.xml')

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
    seen = readTxt(seen_file)  # seen classes 249
    unseen = readTxt(unseen_file)  # unseen classes 361
    # all_wnids = readTxt(all_wnids_file)  # all valid class set
    return seen, unseen

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
    print(len(new_vertices))

    # filter edge list
    new_edges = list()
    for edge in edge_list:
        node1 = edge[0]
        node2 = edge[1]
        if node1 in stop_classes or node2 in stop_classes:
            continue
        else:
            new_edges.append(edge)
    print(len(new_edges))
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
def prepare_graph():
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
    # f_vertices, f_edges = filter_graph(vertex_list, edge_list, valid_classes)

    # return f_vertices, f_edges
    return vertex_list, edge_list


def ID2Index(wnids, node_wnids):
    index_list = list()
    for wnid in node_wnids:
        idx = wnids.index(wnid)
        index_list.append(idx)
    return index_list

def convert_graph(vertices, edges):
    # save graph nodes/wnids/classes
    graph_nodes_file = os.path.join(DATA_DIR, Exp_NAME, 'g_cls_nodes.json')
    with open(graph_nodes_file, 'w') as fp:
        json.dump(vertices, fp)
    print('Save graph node in wnid to %s' % graph_nodes_file)

    # save attribute nodes
    att_nodes = list()
    with open(os.path.join(DATA_DIR, Exp_NAME, 'attribute.txt')) as fp:
        for line in fp.readlines():
            att_id_name = line.split('\t')
            att_nodes.append(att_id_name[0])
    graph_nodes_file = os.path.join(DATA_DIR, Exp_NAME, 'g_att_nodes.json')
    with open(graph_nodes_file, 'w') as fp:
        json.dump(att_nodes, fp)
    print('Save graph node in wnid to %s' % graph_nodes_file)

    # save graph edge pair: class-class
    # edge_pairs_c = np.empty((len(edges), 2), dtype=np.object)
    # edge_pairs_c = list()
    # for i in range(len(edges)):
    #     edge_pairs_c[i][0] = edges[i][0]
    #     edge_pairs_c[i][1] = edges[i][1]

    edge_c_file = os.path.join(DATA_DIR, Exp_NAME, 'edges-c-c.pkl')
    with open(edge_c_file, 'wb') as fp:
        pkl.dump(edges, fp)
    print('Save Graph structure to: ', edge_c_file)
    # print(graph)

    # save graph edge pair: class-attribute
    cls_att_file = os.path.join(DATA_DIR, Exp_NAME, 'class-att.json')
    with open(cls_att_file) as fp:
        pair_list = json.load(fp)



    edge_a_file = os.path.join(DATA_DIR, Exp_NAME, 'edges-c-a.pkl')
    with open(edge_a_file, 'wb') as fp:
        pkl.dump(pair_list, fp)
    print('Save Graph structure to: ', edge_a_file)




def save_corresp_labels(vertices, seen, unseen, seen_label_dict, unseen_label_dict):
    # save classes's index in graph
    seen_idx_in_graph = list()
    unseen_idx_in_graph = list()
    label_list = list()
    for i in range(len(vertices)):
        if vertices[i] in seen:
            seen_idx_in_graph.append(i)
            label_list.append(seen_label_dict[vertices[i]])
        elif vertices[i] in unseen:
            unseen_idx_in_graph.append(i)
            label_list.append(unseen_label_dict[vertices[i]])
        else:
            label_list.append('other')
    seen_corresp_file = os.path.join(DATA_DIR, Exp_NAME, 'g_seen_corresp.json')
    with open(seen_corresp_file, 'w') as fp:
        json.dump(seen_idx_in_graph, fp)
    print('Save seen index in graph to %s' % seen_corresp_file)

    unseen_corresp_file = os.path.join(DATA_DIR, Exp_NAME, 'g_unseen_corresp.json')
    with open(unseen_corresp_file, 'w') as fp:
        json.dump(unseen_idx_in_graph, fp)
    print('Save unseen index in graph to %s' % unseen_corresp_file)

    labels_file = os.path.join(DATA_DIR, Exp_NAME, 'g_labels.json')
    with open(labels_file, 'w') as fp:
        json.dump(label_list, fp)
    print('Save unseen index in graph to %s' % labels_file)



if __name__ == '__main__':
    seen, unseen = load_class()
    vertices, edges = prepare_graph()
    convert_graph(vertices, edges)

    # if graph constructed
    # with open(os.path.join(DATA_DIR, Exp_NAME, 'g_nodes.json')) as fp:
    #     vertices = json.load(fp)

    seen_label_dict = dict()
    with open(os.path.join(DATA_DIR, Exp_NAME, 'seen_label.txt')) as fp:
        for line in fp.readlines():
            class_lbl = line[:-1].split('\t')
            seen_label_dict[class_lbl[0]] = class_lbl[1]
    unseen_label_dict = dict()
    with open(os.path.join(DATA_DIR, Exp_NAME, 'unseen_label.txt')) as fp:
        for line in fp.readlines():
            class_lbl = line[:-1].split('\t')
            unseen_label_dict[class_lbl[0]] = class_lbl[1]

    save_corresp_labels(vertices, seen, unseen, seen_label_dict, unseen_label_dict)





