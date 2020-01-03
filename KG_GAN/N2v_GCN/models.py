import torch.nn as nn
import torch.nn.functional as F
from N2v_GCN.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid2)
        self.fc = nn.Linear(nhid2, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        p = self.fc(x)
        # return F.log_softmax(x, dim=1)
        return x, p
        # return x

