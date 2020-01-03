import numpy as np
import scipy.io as sio
import h5py
import torch
# from sklearn import preprocessing
import sys
# from sklearn.cluster import KMeans


file_name = '/Users/geng/Desktop/xlsa17-xian/data/ImageNet/ImageNet_w2v/wnids.txt'

wnids = open(file_name, 'rU')
wnid_list = list()
try:
    for line in wnids:
        line = line[:-1]
        wnid_list.append(line)
finally:
    wnids.close()

print(len(wnid_list))

file_name = '/Users/geng/Desktop/xlsa17-xian/data/ImageNet/ImageNet_w2v/test.mat'
# file_name = '/Users/geng/Desktop/xlsa17-xian/data/ImageNet/ImageNet_w2v/ImageNet_w2v.mat'
matcontent = sio.loadmat(file_name)
# print(matcontent['w2v'])

words = np.array(matcontent['data']).squeeze()
print(words.shape)

# print(words[19])

wnid_file = '/Users/geng/Desktop/xlsa17-xian/data/ImageNet/ImageNet_w2v/words.txt'
wr_fp = open(wnid_file, 'w')

for i in range(len(words)):
    wnid = wnid_list[i]  # wnid
    word_set = words[i][0]
    line = wnid + "\t"
    for v in word_set:
        item = "|"+v[0]
        line += item
    print(line)
    wr_fp.write('%s\n' % line)
wr_fp.close()

# print(words)
# for v in words[19][0]:
#     print(v[0])
#
# for i in range(len(words)):



'''
# save non-word2vec wnids
matcontent = h5py.File(file_name, 'r')
no_w2v_loc = np.array(matcontent['no_w2v_loc']).squeeze()
# print(no_w2v_loc)
no_list = list()
for v in no_w2v_loc:
    v = int(v)
    no_list.append(v)
print("no length:", len(no_list))
print(no_list)

wnids = np.array(matcontent['wnids']).squeeze()
wnid_file = '/Users/geng/Desktop/xlsa17-xian/data/ImageNet/ImageNet_w2v/wnids-no-w2v.txt'
wr_fp = open(wnid_file, 'w')
wnid_list = list()
for i in range(len(wnids)):
    wnid = ''.join([chr(v[0]) for v in matcontent[(wnids[i])]])
    if (i+1) in no_list:
        print(wnid)
        wnid_list.append(wnid)
        wr_fp.write('%s\n' % wnid)
print(len(wnid_list))
wr_fp.close()
'''

# matcontent = h5py.File(file_name, 'r')
# feature = np.array(matcontent['w2v']).T
# print(feature.shape)
# print(feature[0])
#
# words = matcontent['words']
# st = words[0][0]

# obj = matcontent[st]
# str1 = ''.join(chr(i) for i in obj[:])
# print( str1 )

# image_file = np.array(matcontent['image_files']).squeeze()
# # print(type(image_file))
# print(image_file[0])

# words = matcontent['words'][:]
# print(words)
# words = np.array(matcontent['words']).squeeze()
# print(words.shape)



# print(type(matcontent[(words[0])][0]))




# matcontent = h5py.File(file_name, 'r')
# words = np.array(matcontent['words']).squeeze()
# for v in matcontent[(words[0])]:
#     print(v)
#     for i in v:
#         i = str(i)
#         print(i)
#         print(type(i))




# wnid_file = '/Users/geng/Desktop/xlsa17-xian/data/ImageNet/ImageNet_w2v/wnids.txt'
# wr_fp = open(wnid_file, 'w')
# wnid_list = list()
# for i in range(len(wnids)):
#     wnid = ''.join([chr(v[0]) for v in matcontent[(wnids[i])]])
#     print(wnid)
#     wnid_list.append(wnid)
    # wr_fp.write('%s\n' % wnid)
# print(len(wnid_list))
# wr_fp.close()



