import numpy as np
import scipy
import scipy.io
import argparse
import os
from src.data_reader import get_vec_mat
import scipy.io as sio
import re
from src.count_tools import macro_acc
def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--ld', type=float, default=5) # lambda 500000?
	return parser.parse_args()


def normalizeFeature(x):
	# x = d x N dims (d: feature dimension, N: the number of features)
	x = x + 1e-10 # for avoid RuntimeWarning: invalid value encountered in divide
	feature_norm = np.sum(x**2, axis=1)**0.5 # l2-norm
	feat = x / feature_norm[:, np.newaxis]
	return feat

def SAE(x, s, ld):
	# SAE is Semantic Autoencoder
	# INPUTS:
	# 	x: d x N data matrix
	#	s: k x N semantic matrix
	#	ld: lambda for regularization parameter
	#
	# OUTPUT:
	#	w: kxd projection matrix

	A = np.dot(s, s.transpose())
	B = ld * np.dot(x, x.transpose())
	C = (1+ld) * np.dot(s, x.transpose())
	w = scipy.linalg.solve_sylvester(A,B,C)
	return w

def distCosine(x, y):
	xx = np.sum(x**2, axis=1)**0.5
	x = x / xx[:, np.newaxis]
	yy = np.sum(y**2, axis=1)**0.5
	y = y / yy[:, np.newaxis]
	dist = 1 - np.dot(x, y.transpose())
	return dist



def zsl_acc(semantic_predicted, semantic_gt, opts):
	# zsl_acc calculates zero-shot classification accruacy
	#
	# INPUTS:
	#	semantic_prediced: predicted semantic labels
	# 	semantic_gt: ground truth semantic labels
	# 	opts: other parameters
	#
	# OUTPUT:
	# 	zsl_accuracy: zero-shot classification accuracy (per-sample)
	pre_label=[]
	dist = 1 - distCosine(semantic_predicted, normalizeFeature(semantic_gt.transpose()).transpose())
	y_hit_k = np.zeros((dist.shape[0], opts.HITK))
	for idx in range(0, dist.shape[0]):
		sorted_id = sorted(range(len(dist[idx,:])), key=lambda k: dist[idx,:][k], reverse=True)
		y_hit_k[idx,:] = opts.test_classes_id[sorted_id[0:opts.HITK]]
		
	n = 0
	true_label=opts.test_labels.tolist()
	true_label=[int(i) for i in true_label]
	# for idx in range(0, dist.shape[0]):
	# 	if opts.test_labels[idx] in y_hit_k[idx,:]:
	# 		n = n + 1
	# zsl_accuracy = float(n) / dist.shape[0] * 100
	# return zsl_accuracy, y_hit_k
	for idx in range(0, dist.shape[0]):
		if opts.test_labels[idx] in y_hit_k[idx, :]:
			pre_label.append(int(true_label[idx]))

		else:
			pre_label.append(int(y_hit_k[idx, 0]))
	zsl_accuracy = macro_acc(true_label,pre_label)
	return zsl_accuracy*100, y_hit_k



def main():
	# for AwA dataset: Perfectly works.
	opts = parse_args()

	vec_mat = get_vec_mat()
	folder_dir_train = "./ZSL_DATA/train_seen"



	#part1
	#get the training data

	pth = os.path.join(folder_dir_train)
	dirs = os.listdir(pth)
	x = []
	#y_tag = []
	y_vec = []  # vec
	for f in dirs:
		tmp_dir = os.path.join(pth, f)
		mat = sio.loadmat(tmp_dir)
		features = mat['features'].astype(np.float64)
		# get id
		id = int(re.sub("\D", "", f))
		# id tag build
		# for i in range(features.shape[0]):
		# 	y_tag.append(id)
		# vec build
		idx = id - 1  # must -1
		for i in range(features.shape[0]):
			y_vec.append(vec_mat[idx])
		# features build
		if len(x) == 0:
			x = features
		else:
			x = np.concatenate((x, features), axis=0)
	train_data = x  # seen	19* 1300  *   2048
	train_class_attributes_labels_continuous_allset = y_vec  # w2v 19*   1300*500   to get w

	#part2
	#get the testing data
	folder_dir_test = "./ZSL_DATA/test/unseen"  #seen/unseen
	folder_dir_combine= "./ZSL_DATA/test/combine"#seen/unseen/combine
	pth = os.path.join(folder_dir_test)
	pth2=os.path.join(folder_dir_combine)
	dirs = os.listdir(pth)
	dirs2 = os.listdir(pth2)
	opts.test_classes_id = []  #  id 49*1  /68*1
	test_class_attributes_labels_continuous = []  #  w2v  68*500

	for f in dirs2:
		id = int(re.sub("\D", "", f))
		opts.test_classes_id.append(id)
		idx = id - 1  # must -1
		test_class_attributes_labels_continuous.append(vec_mat[idx])


	x = []
	y_tag = []
	for f in dirs:
		tmp_dir = os.path.join(pth, f)
		mat = sio.loadmat(tmp_dir)
		features = mat['features'].astype(np.float64)
		# get id
		id = int(re.sub("\D", "", f))
		#opts.test_classes_id.append(id)
		# id tag build

		for i in range(features.shape[0]):  # 50 or //features.shape[0]
			y_tag.append(id)
		# vec build
		idx = id - 1  # must -1

		# features build

		if len(x) == 0:
			x = features
		else:
			x = np.concatenate((x, features), axis=0)

	test_data = x  # unseen 49* <=50  *   2048
	opts.test_labels = y_tag  #49*   <=50*1




	test_data = np.array(test_data).astype(np.float64)
	opts.test_labels = np.array(opts.test_labels).astype(np.float64)
	opts.test_classes_id = np.array(opts.test_classes_id).astype(np.float64)
	test_class_attributes_labels_continuous = np.array(test_class_attributes_labels_continuous).astype(np.float64)
	train_data = np.array(train_data).astype(np.float64)
	train_class_attributes_labels_continuous_allset = np.array(train_class_attributes_labels_continuous_allset).astype(np.float64)


	print("train_data size: ", train_data.shape)  # (24700, 2048)
	print("train_class_attributes data size: ", train_class_attributes_labels_continuous_allset.shape)  # (24700, 500)
	print("test_data size: ", test_data.shape)  # (2325, 2048)
	print("opts.test_labels data size: ", opts.test_labels.shape)  # (2325,)
	print("opts.test_classes_id data size: ", opts.test_classes_id.shape)  # (49,)
	print("test_class_attributes_labels_continuous data size: ", test_class_attributes_labels_continuous.shape)  # (49, 500)




	
	##### Normalize the data
	#train_data = normalizeFeature(train_data.transpose()).transpose()

	##### Training
	# SAE
	W = SAE(train_data.transpose(), train_class_attributes_labels_continuous_allset.transpose(), opts.ld)
	#np.savetxt("W2v.txt", W)
	# W=np.loadtxt("W5.txt")
	#
	# print("模型载入成功")
	##### Test
	semantic_predicted = np.dot(test_data, normalizeFeature(W).transpose())
	opts.HITK = 1
	# [F --> S], projecting data from feature space to semantic space: 84.68% for AwA dataset
	[zsl_accuracy, y_hit_k] = zsl_acc(semantic_predicted, test_class_attributes_labels_continuous, opts)
	print('[1] zsl macro_accuracy [F >>> S]: {:.5f}%'.format(zsl_accuracy))
	opts.HITK = 2
	# [F --> S], projecting data from feature space to semantic space: 84.68% for AwA dataset
	[zsl_accuracy, y_hit_k] = zsl_acc(semantic_predicted, test_class_attributes_labels_continuous, opts)
	print('[2] zsl macro_accuracy [F >>> S]: {:.5f}%'.format(zsl_accuracy))
	opts.HITK = 5
	# [F --> S], projecting data from feature space to semantic space: 84.68% for AwA dataset
	[zsl_accuracy, y_hit_k] = zsl_acc(semantic_predicted, test_class_attributes_labels_continuous, opts)
	print('[2] zsl macro_accuracy [F >>> S]: {:.2f}%'.format(zsl_accuracy))
	# # [S --> F], projecting from semantic to visual space: 84.00% for AwA dataset
	# test_predicted = np.dot(normalizeFeature(test_class_attributes_labels_continuous.transpose()).transpose(), normalizeFeature(W))
	# [zsl_accuracy, y_hit_k] = zsl_acc(test_data, test_predicted, opts)
	# print('[2] zsl macro_accuracy for 19class dataset [S >>> F]: {:.2f}%'.format(zsl_accuracy))
	
if __name__ == '__main__':
	main()
