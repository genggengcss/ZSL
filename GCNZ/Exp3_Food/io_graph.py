import os
import json
import pickle as pkl
import argparse
import xml.etree.ElementTree as ET

'''
prepare_graph(), make_corresp_awa()
prepare the graph data;
original graph file: wordnet_tree_structure.xml
'''


DATA_DIR = '/home/gyx/ZSL/data/ImageNet/Baseline/GCNZ'
# DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet/Baseline/GCNZ'
# Material_DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet'
Material_DATA_DIR = '/home/gyx/ZSL/data/ImageNet'

# animal_wnid = 'n00015388'  # extract animal subset
animal_wnid = 'fa11misc'  # extract fa11misc subset

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


def convert_to_graph(vertices, edges):
    # save vertices
    inv_wordn_file = os.path.join(DATA_DIR, args.exp_name, 'invdict_wordn.json')
    with open(inv_wordn_file, 'w') as fp:
        json.dump(vertices, fp)
        print('Save graph node in wnid to %s' % inv_wordn_file)

    # save the graph as adjacency matrix
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

    graph_file = os.path.join(DATA_DIR, args.exp_name, 'graph.pkl')
    with open(graph_file, 'wb') as fp:
        pkl.dump(graph, fp)
        print('Save Graph structure to: ', graph_file)


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

# filter the invalid classes/nodes
def filter_graph(vertex_list, edge_list):
    valid_classes_file = os.path.join(Material_DATA_DIR, 'materials', 'split-filter/all.txt')
    valid_classes = readTxt(valid_classes_file)

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

# get WordNet graph
# save vertices, words of each vertex, adjacency matrix of the graph
def prepare_graph():
    structure_file = os.path.join(DATA_DIR, 'materials', 'wordnet_tree_structure.xml')
    tree = ET.parse(structure_file)
    root = tree.getroot()
    # move to the animal nodes start
    for sy in root.findall('synset'):  # find synset tag
        for ssy in sy.findall('synset'):  # deeper layer
            print("wnid:", ssy.get('wnid'))
            if ssy.get('wnid') == animal_wnid:  # get tag's attribute value(wnid), 'n00015388' represents 'animal'
                vertex_list, edge_list = add_edge_dfs(ssy)  # animal node -> the root node
            else:
                continue
    # move to the animal nodes end

    vertex_list = list(set(vertex_list))  # remove the repeat node
    print('Unique Vertex #: %d, Edge #: %d', (len(vertex_list), len(edge_list)))
    f_vertices, f_edges = filter_graph(vertex_list, edge_list)
    print('Filtered Vertex #: %d, Edge #: %d', (len(f_vertices), len(f_edges)))

    convert_to_graph(f_vertices, f_edges)


# Label each seen and unseen class with an order number e.g., 0-397, 398-407
# if a graph vertex is a seen or unseen class
#   set the vertex with the class id and label of seen (value: 0) or unseen (1)
def make_corresp():
    seen_file = os.path.join(Material_DATA_DIR, 'KG-GAN', args.mtr_exp_name, 'seen.txt')  # nun:398
    unseen_file = os.path.join(Material_DATA_DIR, 'KG-GAN', args.mtr_exp_name, 'unseen.txt')  # num: 485
    seen_dict = {}
    unseen_dict = {}
    cnt = 0
    with open(seen_file) as fp:
        for line in fp.readlines():
            seen_dict[line.strip()] = cnt
            cnt += 1

    with open(unseen_file) as fp:
        for line in fp.readlines():
            unseen_dict[line.strip()] = cnt
            cnt += 1

    inv_wordn_file = os.path.join(DATA_DIR, args.exp_name, 'invdict_wordn.json')
    with open(inv_wordn_file) as fp:
        wnids = json.load(fp)

    corresp_list = []
    for wnid in wnids:
        # this is a seen class, label: 0
        if wnid in seen_dict:
            corresp_id = seen_dict[wnid]
            corresp_list.append([corresp_id, 0])
        # this is an unseen class, label: 1
        elif wnid in unseen_dict:
            corresp_id = unseen_dict[wnid]
            corresp_list.append([corresp_id, 1])
        else:
            corresp_list.append([-1, -1])

    check_train, check_test = 0, 0
    for corresp in corresp_list:
        if corresp[1] == 1:
            check_test += 1
        elif corresp[1] == 0:
            check_train += 1
        else:
            assert corresp[0] == -1
    print('unseen classes #: %d, seen classes #: %d' % (check_test, check_train))

    save_file = os.path.join(DATA_DIR, args.exp_name, 'corresp.json')
    with open(save_file, 'w') as fp:
        json.dump(corresp_list, fp)

'''
Exp2_1949: 19 seen, 49 unseen
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mtr_exp_name', default='Exp2', help='the folder to store Material files')
    parser.add_argument('--exp_name', default='Exp2_1949', help='the folder to store experiment files')
    args = parser.parse_args()

    save_dir = os.path.join(DATA_DIR, args.exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    prepare_graph()
    make_corresp()

