from __future__ import division
from __future__ import print_function
import pickle as pkl
import time
import argparse
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from N2v_GCN.utils import load_data, accuracy, ensure_path
from N2v_GCN.models import GCN

DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet/KG-GAN'
Exp_NAME = 'Exp13'
path = os.path.join(DATA_DIR, Exp_NAME)
save_path = os.path.join(DATA_DIR, Exp_NAME, 'embed')
node_file = 'nodes.txt'
edge_file = 'edges.pkl'



# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=6918, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden1', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--hidden2', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.seed is None:
    ManualSeed = random.randint(1, 10000)
else:
    ManualSeed = args.seed
print("Random Seed: ", ManualSeed)



ensure_path(save_path)

random.seed(ManualSeed)
np.random.seed(ManualSeed)
torch.manual_seed(ManualSeed)
if args.cuda:
    torch.cuda.manual_seed(ManualSeed)

# Load data
adj, features, labels, idx_train = load_data(path, node_file, edge_file)

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid1=args.hidden1,
            nhid2=args.hidden2,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()



def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    hidden, output = model(features, adj)
    # output = model(features, adj)
    pred = F.log_softmax(output, dim=1)

    loss_train = F.nll_loss(pred[idx_train], labels[idx_train])
    acc_train = accuracy(pred[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()


    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'time: {:.4f}s'.format(time.time() - t))

    if (epoch+1) >= 20 and (epoch+1) % 5 == 0:
        filename = str(epoch + 1) + '.pkl'
        save_file = os.path.join(save_path, filename)
        with open(save_file, 'wb') as fp:
            pkl.dump(hidden.data.numpy(), fp)




# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


