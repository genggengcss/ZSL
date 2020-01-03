import dgl
import pickle as pkl
import numpy as np
import os
import torch
import sys
import json
import random
import torch.nn.functional as F
from N2v_DGL_GCN.models import HeteroRGCN
from N2v_DGL_GCN.utils import *



DATA_DIR = '/Users/geng/Desktop/ZSL_DATA/ImageNet/KG-GAN'
Exp_NAME = 'Exp12-HGCN'
path = os.path.join(DATA_DIR, Exp_NAME)
save_path = os.path.join(DATA_DIR, Exp_NAME, 'embed-HGCN')
# node_file = 'nodes.txt'
# edge_file = 'edges.pkl'

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
G, features, labels, labeled_nodes = load_Hete_Data(path)





# Create the model. The output has three logits for three classes.
model = HeteroRGCN(G, 300, 128, 64, labels.max().item() + 1)
print(labels.max().item() + 1)
opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)



for epoch in range(60):
    output, logits = model(G, features)
    # The loss is computed only for labeled nodes.
    loss = F.cross_entropy(logits[labeled_nodes], labels[labeled_nodes])

    pred = logits.argmax(1)
    train_acc = (pred[labeled_nodes] == labels[labeled_nodes]).float().mean()


    opt.zero_grad()
    loss.backward()
    opt.step()


    print('Epoch %d, Loss %.4f, Train Acc %.4f' % (
        epoch+1,
        loss.item(),
        train_acc.item(),
    ))

    if (epoch+1) >= 30 and (epoch+1) % 5 == 0:
        filename = str(epoch + 1) + '.pkl'
        save_file = os.path.join(save_path, filename)
        with open(save_file, 'wb') as fp:
            pkl.dump(output.data.numpy(), fp)