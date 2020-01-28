# -*- coding: utf-8 -*-
import os
import json
import pickle as pkl

from nltk.corpus import wordnet as wn
import xml.etree.ElementTree as ET

from py2neo import Graph, Node, Relationship, NodeMatcher
from py2neo.matching import RelationshipMatcher
from nltk.corpus import wordnet as wn
from py2neo import Graph, Node, Relationship, NodeMatcher
from py2neo.matching import RelationshipMatcher



def getnode(x):
    return wn.synset_from_pos_and_offset('n', int(x[1:]))
def getwnid(u):
    s = str(u.offset())
    return 'n' + (8 - len(s)) * '0' + s

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


# subset_wnid = 'n00015388'  # extract animal subset
# subset_wnid = 'n00021939'  # extract artifact subset
subset_wnid = 'fa11misc'  # extract fa11misc subset
# add the edges between the input node and its children
# with deep first search
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

wordnet = '/Users/geng/Desktop/ZSL_DATA/ImageNet/materials/structure_released.xml'

# get WordNet graph
# save vertices, words of each vertex, adjacency matrix of the graph
def prepare_graph():
    structure_file = wordnet
    tree = ET.parse(structure_file)
    root = tree.getroot()
    # move to the animal nodes start
    for sy in root.findall('synset'):  # find synset tag
        for ssy in sy.findall('synset'):  # deeper layer
            # print("wnid:", ssy.get('wnid'))
            if ssy.get('wnid') == subset_wnid:  # get tag's attribute value(wnid), 'n00015388' represents 'animal'
                vertex_list, edge_list = add_edge_dfs(ssy)  # animal node -> the root node
            else:
                continue
    # move to the animal nodes end

    return edge_list


# MATCH (GraphT) RETURN (GraphT);
# MATCH (GraphT) DETACH DELETE GraphT;

def prepare_neo4j():

    # 连接数据库
    GraphT = Graph("http://localhost:7474", username="neo4j", password='gyx43')

    # 创建节点
    for node in seen:
        syn = getnode(node)
        syn_name = syn.lemma_names()[0]
        p = Node("Seen", id=node, name=syn_name)
        GraphT.create(p)

    for node in unseen:
        syn = getnode(node)
        # print(node)
        syn_name = syn.lemma_names()[0]
        p = Node("Unseen", id=node, name=syn_name)
        GraphT.create(p)



    matcher = NodeMatcher(GraphT)

    for edge in edge_list:
        ed = ['', '']
        flag = 0
        for i, node in enumerate(edge):
            if node in seen:
                ed[i] = matcher.match("Seen", id=node).first()
                flag += 1
            elif node in unseen:
                ed[i] = matcher.match("Unseen", id=node).first()
                flag += 1


        if flag == 2:
            r = Relationship(ed[0], " ", ed[1])
            GraphT.create(r)
        else:
            continue


if __name__ == '__main__':

    seen_file = '/Users/geng/Desktop/ZSL_DATA/ImageNet/materials/split-fa11misc/seen.txt'
    unseen_file = '/Users/geng/Desktop/ZSL_DATA/ImageNet/materials/split-fa11misc/unseen.txt'

    # seen_file = '/Users/geng/Desktop/ZSL_DATA/ImageNet/materials/split-artifact/seen.txt'
    # unseen_file = '/Users/geng/Desktop/ZSL_DATA/ImageNet/materials/split-artifact/unseen.txt'


    # loal classes
    seen = list()
    unseen = list()

    if subset_wnid == 'fa11misc':
        seen_ = readTxt(seen_file)
        unseen_ = readTxt(unseen_file)

        animal_wnids_file = os.path.join('/Users/geng/Desktop/ZSL_DATA/ImageNet/materials/wnids-animal.txt')
        animal_wnids = readTxt(animal_wnids_file)
        for node in seen_:
            if node in animal_wnids:
                continue
            else:
                seen.append(node)

        for node in unseen_:
            if node in animal_wnids:
                continue
            else:
                unseen.append(node)

        print(len(seen))
        print(len(unseen))
    else:
        seen = readTxt(seen_file)
        unseen = readTxt(unseen_file)
    # extract background graph
    edge_list = prepare_graph()

    prepare_neo4j()





