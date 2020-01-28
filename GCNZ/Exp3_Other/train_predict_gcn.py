# coding=gbk
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import argparse
import time
import os
import tensorflow as tf
import numpy as np
import pickle as pkl

import sys

sys.path.append('../')

from gcn.utils import load_data_vis_multi
from gcn.utils import preprocess_features_dense2
from gcn.utils import preprocess_adj
from gcn.utils import create_config_proto
from gcn.utils import construct_feed_dict
from gcn.models import GCN_dense_mse

# from IMAGENET_Animal.Exp_Test.test_in_train import test_imagenet_zero

'''
train gcn
'''

DATA_DIR = '/home/gyx/ZSL/data/ImageNet/Baseline/GCNZ'
# DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet/Baseline/GCNZ'
# Material_DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet'
Material_DATA_DIR = '/home/gyx/ZSL/data/ImageNet'

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
parser = argparse.ArgumentParser()

parser.add_argument('--mtr_exp_name', default='Exp2', help='the folder to store Material files')
parser.add_argument('--exp_name', default='Exp2_1949', help='the folder to store experiment files')

parser.add_argument('--dataset', default='w2v_res101', help='Dataset string.')
parser.add_argument('--model', default='dense', help='Model string.')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--out_path', default='w2v_res101/output', help='save dir')
parser.add_argument('--epochs', default=1000, help='Number of epochs to train.')
parser.add_argument('--save_epoch', default=300, help='Number of epochs to train.')

parser.add_argument('--hidden1', type=int, default=2048, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=2048, help='Number of units in hidden layer 2.')
parser.add_argument('--hidden3', type=int, default=1024, help='Number of units in hidden layer 3.')
parser.add_argument('--hidden4', type=int, default=1024, help='Number of units in hidden layer 4.')
parser.add_argument('--hidden5', type=int, default=512, help='Number of units in hidden layer 5.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=5e-3, help='Weight for L2 loss on embedding matrix.')
parser.add_argument('--early_stopping', type=int, default=10, help='Tolerance for early stopping (# of epochs).')
parser.add_argument('--max_degree', type=int, default=3, help='Maximum Chebyshev polynomial degrees.')
parser.add_argument('--gpu', default='2', help='gpu id')


args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

args.dataset = os.path.join(DATA_DIR, args.exp_name, 'w2v_res101')
args.out_path = os.path.join(DATA_DIR, args.exp_name, 'w2v_res101/output')



use_trainval = True
X_dense_file = 'all_x_dense.pkl'
train_y_file = 'train_y.pkl'
graph_file = 'graph.pkl'
test_index_file = 'test_index.pkl'



# train_adj_mask, val_adj_mask is mainly for mask the attention weights matrices
adj, X, y_train, train_mask, train_adj_mask, val_mask, val_adj_mask, trainval_mask, trainval_adj_mask = \
    load_data_vis_multi(args.dataset, use_trainval, X_dense_file, train_y_file,
                        graph_file, test_index_file)

# Some preprocessing
X, div_mat = preprocess_features_dense2(X)

if args.model == 'dense':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN_dense_mse
else:
    raise ValueError('Invalid argument for model: ' + str(args.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],  # adj
    'features': tf.placeholder(tf.float32, shape=(X.shape[0], X.shape[1])),  # sparse_placeholder
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'labels_adj_mask': tf.placeholder(tf.int32),
    'val_mask': tf.placeholder(tf.int32),
    'val_adj_mask': tf.placeholder(tf.int32),
    'trainval_mask': tf.placeholder(tf.int32),
    'trainval_adj_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
    'learning_rate': tf.placeholder(tf.float32, shape=())
}

# Create model
model = model_func(args, placeholders, input_dim=X.shape[1], logging=True)

sess = tf.Session(config=create_config_proto())

# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []


if not os.path.exists(args.out_path):
    os.makedirs(args.out_path)
    print('!!! Make directory %s' % args.out_path)
else:
    print('### save to: %s' % args.out_path)

# Train model
now_lr = args.learning_rate
for epoch in range(args.epochs):
    t = time.time()

    # Construct feed dictionary
    # train_mask: point out which vertices are used as seen classes
    # feed_dict = construct_feed_dict(X, support, y_train, train_mask, placeholders)
    # feed_dict.update({placeholders['learning_rate']: now_lr})
    feed_dict = construct_feed_dict(X, support, y_train, train_mask, train_adj_mask,
                                    val_mask, val_adj_mask, trainval_mask, trainval_adj_mask, placeholders)
    feed_dict.update({placeholders['learning_rate']: now_lr})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy, model.optimizer._lr], feed_dict=feed_dict)

    if epoch % 20 == 0:
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_loss_nol2=", "{:.5f}".format(outs[2]),
              "time=", "{:.5f}".format(time.time() - t),
              "lr=", "{:.5f}".format(float(outs[3])))
        # print(sess.run(model.outputs, feed_dict=feed_dict))

    # Predicting step
    # --save outputs
    # if (epoch + 1) in save_epochs:
    if (epoch + 1) % 50 == 0 and (epoch + 1) >= args.save_epoch:
        outs = sess.run(model.outputs, feed_dict=feed_dict)
        filename = os.path.join(args.out_path, ('feat_%d' % (epoch + 1)))
        print(time.strftime('[%X %x %Z]\t') + 'save to: ' + filename)

        filehandler = open(filename, 'wb')
        pkl.dump(outs, filehandler)
        filehandler.close()


print("Optimization Finished!")

sess.close()
