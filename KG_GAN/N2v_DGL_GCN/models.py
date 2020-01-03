import torch
import torch.nn as nn
import torch.nn.functional as F
from N2v_DGL_GCN.layers import GCNLayer
from N2v_DGL_GCN.layers import HeteroRGCNLayer


# Define a 2-layer GCN model
class GCN(nn.Module):
    def __init__(self, in_feats, hidden1_size, num_classes):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_feats, hidden1_size)
        # self.gcn2 = GCNLayer(hidden1_size, hidden2_size)
        # self.fc = nn.Linear(hidden2_size, num_classes)
        self.gcn2 = GCNLayer(hidden1_size, num_classes)

    def forward(self, g, inputs):
        h = self.gcn1(g, inputs)
        h = torch.relu(h)
        h = self.gcn2(g, h)  # output
        # x = self.fc(h)  # predict
        return h
        # return h, x



class HeteroRGCN(nn.Module):
    def __init__(self, G, in_size, hidden1_size, hidden2_size , out_size):
        super(HeteroRGCN, self).__init__()

        # create layers
        self.layer1 = HeteroRGCNLayer(in_size, hidden1_size, G.etypes)
        self.layer2 = HeteroRGCNLayer(hidden1_size, hidden2_size, G.etypes)
        self.fc = nn.Linear(hidden2_size, out_size)

    def forward(self, G, features):
        h_dict = self.layer1(G, features)
        h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(G, h_dict)
        # get cls logits
        # return h_dict['cls']

        output = h_dict['cls']
        pred = self.fc(output)
        return output, pred