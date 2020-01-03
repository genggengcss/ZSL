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

# wnid file
seen_file = os.path.join(DATA_DIR, Exp_NAME, 'seen.txt')
unseen_file = os.path.join(DATA_DIR, Exp_NAME, 'unseen.txt')
tool_file = os.path.join(DATA_DIR, Exp_NAME, 'tool_class.txt')
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
    tool_class = readTxt(tool_file)
    return seen, unseen, tool_class

# filter the invalid classes/nodes
def filter_graph(vertex_list, edge_list, seen, unseen, tool_classes):
    stop_classes = list()
    # filter vertex list
    new_vertices = list()
    for node in vertex_list:
        if node in seen or node in unseen or node in tool_classes:
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
def prepare_graph(seen, unseen, tool_class):
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
    f_vertices, f_edges = filter_graph(vertex_list, edge_list, seen, unseen, tool_class)

    return f_vertices, f_edges


def ID2Index(wnids, node_wnids):
    index_list = list()
    for wnid in node_wnids:
        idx = wnids.index(wnid)
        index_list.append(idx)
    return index_list

def convert_graph(vertices, edges):
    # save graph nodes/wnids/classes
    graph_nodes_file = os.path.join(DATA_DIR, Exp_NAME, 'cls/g_nodes.json')
    with open(graph_nodes_file, 'w') as fp:
        json.dump(vertices, fp)
    print('Save graph node in wnid to %s' % graph_nodes_file)

    # save graph initial word embedding
    node_embed_file = os.path.join(DATA_DIR, Exp_NAME, 'cls/g_embed.mat')
    matcontent = scio.loadmat(os.path.join(Material_DATA_DIR, 'w2v.mat'))
    wnids = matcontent['wnids'].squeeze().tolist()
    vertices_index = ID2Index(wnids, vertices)
    # print(vertices_index)
    w2v = matcontent['w2v']
    word_embed = w2v[vertices_index]  # (3695, 500)
    scio.savemat(node_embed_file, {'w2v': word_embed})
    print('Save graph node initial embedding to %s' % node_embed_file)

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
    graph_file = os.path.join(DATA_DIR, Exp_NAME, 'cls/graph.pkl')
    with open(graph_file, 'wb') as fp:
        pkl.dump(graph, fp)
    print('Save Graph structure to: ', graph_file)
    # print(graph)

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
    seen_corresp_file = os.path.join(DATA_DIR, Exp_NAME, 'cls/g_seen_corresp.json')
    with open(seen_corresp_file, 'w') as fp:
        json.dump(seen_idx_in_graph, fp)
    print('Save seen index in graph to %s' % seen_corresp_file)

    unseen_corresp_file = os.path.join(DATA_DIR, Exp_NAME, 'cls/g_unseen_corresp.json')
    with open(unseen_corresp_file, 'w') as fp:
        json.dump(unseen_idx_in_graph, fp)
    print('Save unseen index in graph to %s' % unseen_corresp_file)




if __name__ == '__main__':
    seen, unseen, tool_class = load_class()
    vertices, edges = prepare_graph(seen, unseen, tool_class)
    convert_graph(vertices, edges)





