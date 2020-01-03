import dgl
import pickle as pkl
import numpy as np
import os
import torch
import sys
import json
import random
import torch.nn.functional as F
from N2v_DGL_GCN.models import GCN
from N2v_DGL_GCN.utils import *




DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet/KG-GAN'
Exp_NAME = 'Exp11-GCN'
path = os.path.join(DATA_DIR, Exp_NAME)
save_path = os.path.join(DATA_DIR, Exp_NAME, 'embed-GCN')
node_file = 'nodes.txt'
edge_file = 'edges.pkl'

Seed = None
if Seed is None:
    ManualSeed = random.randint(1, 10000)
else:
    ManualSeed = Seed
print("Random Seed: ", ManualSeed)



ensure_path(save_path)

random.seed(ManualSeed)
np.random.seed(ManualSeed)
torch.manual_seed(ManualSeed)

torch.cuda.manual_seed(ManualSeed)

# Load data
G, features, labels, labeled_nodes = load_data(path, node_file, edge_file)


print('We have %d nodes.' % G.number_of_nodes())
print('We have %d edges.' % G.number_of_edges())

# The first layer transforms input features of size of 34 to a hidden size of 5.
# The second layer transforms the hidden layer and produces output features of
# size 2, corresponding to the two groups of the karate club.
# net = GCN(features.shape[1], 128, 64, labels.max().item() + 1)
net = GCN(features.shape[1], 64, labels.max().item() + 1)

print(labels.max().item() + 1)


optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
all_logits = []
for epoch in range(100):
    # output, logits = net(G, features)
    logits = net(G, features)
    # we save the logits for visualization later
    all_logits.append(logits.detach())
    logp = F.log_softmax(logits, 1)
    # we only compute loss for labeled nodes
    loss = F.nll_loss(logp[labeled_nodes], labels[labeled_nodes])
    acc = accuracy(logp[labeled_nodes], labels[labeled_nodes])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch %d | Loss: %.4f | Acc: %.4f' % (epoch+1, loss.item(), acc.item()))

    if (epoch+1) >= 30 and (epoch+1) % 5 == 0:
        filename = str(epoch + 1) + '.pkl'
        save_file = os.path.join(save_path, filename)
        # seen_embed = hidden_emb[seen_corresp]
        # unseen_embed = hidden_emb[unseen_corresp]
        # embed = np.vstack((seen_embed, unseen_embed))
        # print("embed shape:", embed.shape)
        with open(save_file, 'wb') as fp:
            pkl.dump(logits.data.numpy(), fp)


