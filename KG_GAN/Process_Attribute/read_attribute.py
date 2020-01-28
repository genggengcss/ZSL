from collections import OrderedDict
import csv
from itertools import islice
import os
from nltk.corpus import wordnet as wn
import json



def getnode(x):
    return wn.synset_from_pos_and_offset('n', int(x[1:]))
def getwnid(u):
    s = str(u.offset())
    return 'n' + (8 - len(s)) * '0' + s

class_attribute_csv = "/Users/geng/Desktop/IJCAI/attribute-annotation/class-attribute-0120.csv"

DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet/KG-GAN'
EXP_NAME = 'Exp10'

cls_att_csv = open(class_attribute_csv, "r")
reader = csv.reader(cls_att_csv)

seen_classes = list()
seen_cls_att = OrderedDict()
unseen_classes = list()
unseen_cls_att = OrderedDict()
# skip the first line
for item in islice(reader, 1, None):
    # line = item[0].split('\t')[2]

    # seen class
    seen_class_id = item[0]
    if seen_class_id != '':
        seen_classes.append(seen_class_id)
        name = getnode(seen_class_id).lemma_names()[0]
        if name == item[1]:
            seen_cls_att[seen_class_id] = item[2]
        else:
            print(item)

    unseen_class_id = item[3]
    if unseen_class_id != '':
        unseen_classes.append(unseen_class_id)
        name = getnode(unseen_class_id).lemma_names()[0]

        if name == item[4]:
            unseen_cls_att[unseen_class_id] = item[5]
        else:
            print(item)


cls_att_csv.close()


print("seen classes:", len(seen_classes))
print("unseen classes:", len(unseen_classes))


# extract attribute
att_list = list()
for wnid, att in seen_cls_att.items():
    atts = att.split(',')

    atts1 = [at.strip() for at in atts]
    if '' in atts1:
        atts1.remove('')  # remove the empty value
    att_list.extend(atts1)
    # print(atts1)
# print(len(att_list))
# print(len(set(att_list)))
for wnid, att in unseen_cls_att.items():
    atts = att.split(',')

    atts1 = [at.strip() for at in atts]
    if '' in atts1:
        atts1.remove('')  # remove the empty value
    att_list.extend(atts1)
    # print(atts1)
print(len(att_list))
print(att_list)
att_list2 = sorted(set(att_list), key=att_list.index)
print(len(att_list2))
print(att_list2)

def saveClass(item_list, filename):
    file = os.path.join(DATA_DIR, EXP_NAME, filename)
    wr_fp = open(file, 'w')
    for item in item_list:
        wr_fp.write('%s\n' % item)
    wr_fp.close()

def saveAttr(att_list, filename):
    att_id_dict = dict()
    file = os.path.join(DATA_DIR, EXP_NAME, filename)
    wr_fp = open(file, 'w')
    for i in range(len(att_list)):
        index = 'a' + str("%03d" % (i+1))
        # print(index)
        att_id_dict[att_list[i]] = index
        wr_fp.write('%s\t%s\n' % (index, att_list[i]))
    wr_fp.close()
    return att_id_dict

# save seen classes, unseen classes, attributes
# saveClass(seen_classes, 'seen.txt')
# saveClass(unseen_classes, 'unseen.txt')



att_id_dict = saveAttr(att_list2, 'attribute.txt')
# construct wnid&att pair
edges = list()
for wnid, att in seen_cls_att.items():
    atts = att.split(',')
    for at in atts:
        at = at.strip()
        if at == '':
            continue
        else:
            edges.append((wnid, att_id_dict[at]))

for wnid, att in unseen_cls_att.items():
    atts = att.split(',')
    for at in atts:
        at = at.strip()
        if at == '':
            continue
        else:
            edges.append((wnid, att_id_dict[at]))
json.dump(edges, open(os.path.join(DATA_DIR, EXP_NAME, 'class-att.json'), 'w'))






# print(result)

# wr_fp = open(new_file_name, 'w')
#
# alarm_ids_11 = list()
# id_names = open(file_name, 'rU')
# try:
#     for line in id_names:
#         line = line[:-1]
#         id, name_cn, name_en = line.split("=")
#         id = id.strip()
#         name_cn = name_cn.strip()
#         name_en = name_en.strip()
#         # print(id, " ", name_cn, " ", name_en)
#         new_line = id + "\t" + name_cn + "\t" + name_en
#
#         alarm_ids_11.append(id)
#
# finally:
#     id_names.close()
#
#
# print(len(alarm_ids_11))
#
# alarm_ids_10 = dict()
# id_names = open(file_name_o, 'rU')
# try:
#     for line in id_names:
#         line = line[:-1]
#         id, name_cn = line.split("\t")
#         id = id.strip()
#         name_cn = name_cn.strip()
#
#
#         new_line = id + "\t" + name_cn
#         # print(new_line)
#         alarm_ids_10[id] = name_cn
#
# finally:
#     id_names.close()
#
#
#
# print(len(set(alarm_ids_11)))
# print(len(set(alarm_ids_10)))
# overlap = list()
#
#
# for key, value in alarm_ids_10.items():
#     if key not in alarm_ids_11:
#         print(key, '=', value, '= ')
#         line = key + "\t" + value
#         wr_fp.write('%s\n' % line)
#         overlap.append(key)
# print(len(overlap))
#
# wr_fp.close()


