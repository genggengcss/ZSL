# -*- coding: utf-8 -*-
import os
import json
import pickle as pkl

from nltk.corpus import wordnet as wn
import xml.etree.ElementTree as ET

import scipy.io as scio

'''
Prepare the graph inputting for GAE
'''
animal_wnid = 'n00015388'  # for extracting animal subset
DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet/KG-GAN'
Exp_NAME = 'Exp15'
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

    vertex_list = list()
    edge_list = list()
    # add attribute
    cls_att_file = os.path.join(DATA_DIR, Exp_NAME, 'class-att.json')
    with open(cls_att_file) as fp:
        pair_list = json.load(fp)
    for pair in pair_list:
        id1 = pair[0]
        id2 = pair[1]
        vertex_list.append(id1)
        vertex_list.append(id2)
        edge_list.append(pair)
    vertex_list = list(set(vertex_list))  # remove the repeat node
    print('Unique Vertex #: %d, Edge #: %d', (len(vertex_list), len(edge_list)))


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
    graph_nodes_file = os.path.join(DATA_DIR, Exp_NAME, 'att/g_nodes.json')
    with open(graph_nodes_file, 'w') as fp:
        json.dump(vertices, fp)
    print('Save graph node in wnid to %s' % graph_nodes_file)



    # save graph
    ver_dict = {}
    graph = {}
    for i, vertex in enumerate(vertices):
        ver_dict[vertex] = i
        graph[i] = []
    for edge in edges:
        id1 = ver_dict[edge[0]]
        id2 = ver_dict[edge[1]]
        graph[id1].append(id2)
        graph[id2].append(id1)
    graph_file = os.path.join(DATA_DIR, Exp_NAME, 'att/graph.pkl')
    with open(graph_file, 'wb') as fp:
        pkl.dump(graph, fp)
    print('Save Graph structure to: ', graph_file)
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
    seen_corresp_file = os.path.join(DATA_DIR, Exp_NAME, 'att/g_seen_corresp.json')
    with open(seen_corresp_file, 'w') as fp:
        json.dump(seen_idx_in_graph, fp)
    print('Save seen index in graph to %s' % seen_corresp_file)

    unseen_corresp_file = os.path.join(DATA_DIR, Exp_NAME, 'att/g_unseen_corresp.json')
    with open(unseen_corresp_file, 'w') as fp:
        json.dump(unseen_idx_in_graph, fp)
    print('Save unseen index in graph to %s' % unseen_corresp_file)


if __name__ == '__main__':
    seen, unseen = load_class()
    vertices, edges = prepare_graph()
    convert_graph(vertices, edges)

    # if graph constructed
    # with open(os.path.join(DATA_DIR, Exp_NAME, 'g_nodes.json')) as fp:
    #     vertices = json.load(fp)
    save_corresp(vertices, seen, unseen)





