import numpy as np
import scipy.io as scio
import h5py
import torch
from sklearn import preprocessing
import sys
from sklearn.cluster import KMeans

file_name = '/Users/geng/Desktop/xlsa17-xian/data/ImageNet/ILSVRC2012_res101_feature.mat'

file_save_name = '/Users/geng/Desktop/xlsa17-xian/data/ImageNet/test.mat'
# file = '/private/tmp/448.mat'
# matcontent = scio.loadmat(file)
# data = matcontent['features']
# print(data.shape)
# print(data[0])
# print(type(data[0][0]))


# <KeysViewHDF5 ['#refs#', 'features', 'features_val', 'image_files', 'image_files_val', 'labels', 'labels_val']>
# with h5py.File(file_name, 'r') as f:
#     item = f['#refs#']
#     print(item)
#     print(len(item))

# matcontent = h5py.File(file_name, 'r')
# feature = np.array(matcontent['features'])
# print(feature.shape)
matcontent = h5py.File(file_name, 'r')
label = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
print(label)
# i = 0
# while (i < 50000):
#     print(label[i])
#     i += 1

# classes = np.unique(label)
# print(classes.shape)
# train features
# label = np.array(matcontent['labels']).astype(int).squeeze() - 1
# print(label[0])
# print(label[1299])
#
# print(label[1300])
# print(label[2599])
#
# print(label[2600])
# classes = np.unique(label)
# print(classes)

# class_num = dict()
# for i in range(len(classes)):
#     class_num[classes[i]] = np.sum(label == classes[i])
#
# print(class_num)

# features = np.array(matcontent['features'])
#
# scio.savemat(file_save_name, {'features': features})
# print(type(features))
# print(features.shape)

# for


# val features
image_file = np.array(matcontent['image_files']).squeeze()
# print(type(image_file))
print(image_file[0])
name = ''.join([chr(v[0]) for v in matcontent[(image_file[0])]])
print(name)

# file_list = list()
# for i in range(len(image_file)):
#     name = ''.join([chr(v[0]) for v in matcontent[(image_file[i])]])
#     print(name)
#     file_list.append(name)
# print(len(file_list))


# label = np.array(matcontent['labels']).astype(int).squeeze() - 1
#
# label = torch.from_numpy(label).long()
# print(label)
# print(len(label))
#
# label = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
#
# label = torch.from_numpy(label).long()
# print(label)
# print(len(label))
#
# feature = matcontent['features']
# print(feature.shape)
#
# feature_val = matcontent['features_val']
# print(feature_val.shape)

# classes = torch.from_numpy(np.unique(label.numpy()))
# print(classes)
# print(len(classes))