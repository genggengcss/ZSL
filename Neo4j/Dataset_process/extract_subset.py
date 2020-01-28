import os
import json
import pickle as pkl
import argparse
import xml.etree.ElementTree as ET
from nltk.corpus import wordnet as wn





Material_DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet'
# Material_DATA_DIR = '/home/gyx/ZSL/data/ImageNet'




def getnode(x):
    return wn.synset_from_pos_and_offset('n', int(x[1:]))

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


def writeToTxt(wnids, type):
    path = os.path.join(Material_DATA_DIR, 'materials', 'wnids-'+type+'.txt')
    wr_fp = open(path, 'w')
    for wnid in wnids:
        wr_fp.write('%s\n' % (wnid))

    wr_fp.close()

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


# get WordNet graph
# save vertices, words of each vertex, adjacency matrix of the graph
def extract_graph():
    structure_file = os.path.join(Material_DATA_DIR, 'materials', 'structure_released.xml')
    tree = ET.parse(structure_file)
    root = tree.getroot()
    # move to the animal nodes start
    for sy in root.findall('synset'):  # find synset tag
        for ssy in sy.findall('synset'):  # deeper layer

            print("name:", getnode(ssy.get('wnid')).lemma_names()[0], " wnid:", ssy.get('wnid'))

            # vertex_list, edge_list = add_edge_dfs(ssy)  # animal node -> the root node
            # vertex_list = list(set(vertex_list))
            #
            # seen_inter_class = list(set(vertex_list).intersection(set(seen)))
            # unseen_inter_class = list(set(vertex_list).intersection(set(unseen)))
            #
            # if ssy.get('wnid') == 'fa11misc':
            #     print("fa11misc :", len(vertex_list), "-", len(seen_inter_class), "-", len(unseen_inter_class))
            #     continue
            #
            # # vertex_list, edge_list = add_edge_dfs(ssy)  # animal node -> the root node
            # # vertex_list = list(set(vertex_list))
            # name = getnode(ssy.get('wnid')).lemma_names()[0]
            # # print(name, ":", len(vertex_list))
            # print(name, ":", len(vertex_list), "-", len(seen_inter_class), "-", len(unseen_inter_class))


            # writeToTxt(vertex_list, name)


    # move to the animal nodes end




if __name__ == '__main__':
    ori_seen_file = os.path.join(Material_DATA_DIR, 'materials', 'split-filter/1k.txt')
    ori_unseen_file = os.path.join(Material_DATA_DIR, 'materials', 'split-filter/2-hops.txt')

    seen = readTxtFile(ori_seen_file)
    unseen = readTxtFile(ori_unseen_file)

    extract_graph()

