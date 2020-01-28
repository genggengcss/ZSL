import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class MLP_AC_D(nn.Module):
    def __init__(self, opt):
        super(MLP_AC_D, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.ndh)
        self.disc_linear = nn.Linear(opt.ndh, 1)
        self.aux_linear = nn.Linear(opt.ndh, opt.attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, x):
        h = self.lrelu(self.fc1(x))
        s = self.sigmoid(self.disc_linear(h))
        a = self.aux_linear(h)
        return s, a


class MLP_AC_2HL_D(nn.Module):
    def __init__(self, opt):
        super(MLP_AC_2HL_D, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.disc_linear = nn.Linear(opt.ndh, 1)
        self.aux_linear = nn.Linear(opt.ndh, opt.attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)

        self.apply(weights_init)

    def forward(self, x):
        h = self.dropout(self.lrelu(self.fc1(x)))
        h = self.dropout(self.lrelu(self.fc2(h)))
        s = self.sigmoid(self.disc_linear(h))
        a = self.aux_linear(h)
        return s, a


class MLP_3HL_CRITIC(nn.Module):
    def __init__(self, opt):
        super(MLP_3HL_CRITIC, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc3 = nn.Linear(opt.ndh, opt.ndh)
        self.fc4 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.lrelu(self.fc2(h))
        h = self.lrelu(self.fc3(h))
        h = self.fc4(h)
        return h


class MLP_2HL_CRITIC(nn.Module):
    def __init__(self, opt):
        super(MLP_2HL_CRITIC, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc3 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.lrelu(self.fc2(h))
        h = self.fc3(h)
        return h


class MLP_2HL_Dropout_CRITIC(nn.Module):
    def __init__(self, opt):
        super(MLP_2HL_Dropout_CRITIC, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc3 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.dropout = nn.Dropout(p=0.5)
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        h = self.dropout(self.lrelu(self.fc1(h)))
        h = self.dropout(self.lrelu(self.fc2(h)))
        h = self.fc3(h)
        return h


class MLP_CRITIC(nn.Module):
    def __init__(self, args):
        super(MLP_CRITIC, self).__init__()
        self.fc1 = nn.Linear(args.FeaSize + args.SemSize, args.NDH)
        # self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(args.NDH, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.apply(weights_init)

    def forward(self, x, sem):
        h = torch.cat((x, sem), 1)
        h = self.lrelu(self.fc1(h))
        h = self.fc2(h)
        return h


class MLP_D(nn.Module):
    def __init__(self, opt):
        super(MLP_D, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.sigmoid(self.fc2(h))
        return h


class MLP_2HL_Dropout_G(nn.Module):
    def __init__(self, opt):
        super(MLP_2HL_Dropout_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        # self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.5)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.dropout(self.lrelu(self.fc1(h)))
        h = self.dropout(self.lrelu(self.fc2(h)))
        h = self.relu(self.fc3(h))
        return h


class MLP_3HL_G(nn.Module):
    def __init__(self, opt):
        super(MLP_3HL_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, opt.ngh)
        self.fc4 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        # self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.lrelu(self.fc2(h))
        h = self.lrelu(self.fc3(h))
        h = self.relu(self.fc4(h))
        return h


class MLP_2HL_G(nn.Module):
    def __init__(self, opt):
        super(MLP_2HL_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        # self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.lrelu(self.fc2(h))
        h = self.relu(self.fc3(h))
        return h


class MLP_Dropout_G(nn.Module):
    def __init__(self, opt):
        super(MLP_Dropout_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.2)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.dropout(self.lrelu(self.fc1(h)))
        h = self.relu(self.fc2(h))
        return h


class MLP_G_GCN(nn.Module):
    def __init__(self, args):
        super(MLP_G_GCN, self).__init__()

        self.gcn1 = GraphConvolution(300, 100, act=F.relu)
        self.gcn2 = GraphConvolution(100, 50, act=lambda x: x)

        self.fc1 = nn.Linear(args.SemSize + args.NoiseSize, args.NGH)
        self.fc2 = nn.Linear(args.NGH, args.FeaSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        # self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)

        self.apply(weights_init)

    def encode(self, x, adj):
        hidden1 = self.gcn1(x, adj)
        output = self.gcn2(hidden1, adj)
        return output

    def forward(self, noise, data, input_index, input_cls_feat, input_cls_adj, n2v):

        cls_output = self.encode(input_cls_feat, input_cls_adj)
        # att_output = self.encode(input_att_feat, input_att_adj)

        wnids = data.wnids
        seen = data.seen
        unseen = data.unseen


        # n2v = torch.FloatTensor(len(wnids), (cls_output.shape[1] + att_output.shape[1]))
        for i in range(len(wnids)):
            if wnids[i] in seen:
                # att_graph_index = data.att_nodes.index(wnids[i])
                # att_corresp = data.att_seen_corresp.index(att_graph_index)

                cls_graph_index = data.cls_nodes.index(wnids[i])
                cls_corresp = data.cls_seen_corresp.index(cls_graph_index)
                # test = torch.cat((cls_output[cls_corresp], att_output[att_corresp]), -1)

                # print("test shape:", test.shape)
                # print(n2v[i].shape)
                n2v[i] = cls_output[cls_corresp]
            elif wnids[i] in unseen:
                # att_graph_index = data.att_nodes.index(wnids[i])
                # att_corresp = data.att_unseen_corresp.index(att_graph_index)

                cls_graph_index = data.cls_nodes.index(wnids[i])
                cls_corresp = data.cls_unseen_corresp.index(cls_graph_index)
                # n2v[i] = torch.cat((cls_output[cls_corresp], att_output[att_corresp]), -1)
                n2v[i] = cls_output[cls_corresp]
            else:
                continue

        # print(n2v.shape)
        cls_embed = n2v[input_index]
        # print("cls embed shape:", cls_embed.shape)

        h = torch.cat((noise, cls_embed), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h, cls_embed

    # def forward(self, noise, data, input_index, input_cls_feat, input_cls_adj, input_att_feat, input_att_adj, n2v):
    #
    #     cls_output = self.encode(input_cls_feat, input_cls_adj)
    #     att_output = self.encode(input_att_feat, input_att_adj)
    #
    #     wnids = data.wnids
    #     seen = data.seen
    #     unseen = data.unseen
    #
    #
    #     # n2v = torch.FloatTensor(len(wnids), (cls_output.shape[1] + att_output.shape[1]))
    #     for i in range(len(wnids)):
    #         if wnids[i] in seen:
    #             att_graph_index = data.att_nodes.index(wnids[i])
    #             att_corresp = data.att_seen_corresp.index(att_graph_index)
    #
    #             cls_graph_index = data.cls_nodes.index(wnids[i])
    #             cls_corresp = data.cls_seen_corresp.index(cls_graph_index)
    #             test = torch.cat((cls_output[cls_corresp], att_output[att_corresp]), -1)
    #
    #             # print("test shape:", test.shape)
    #             # print(n2v[i].shape)
    #             n2v[i] = test
    #         elif wnids[i] in unseen:
    #             att_graph_index = data.att_nodes.index(wnids[i])
    #             att_corresp = data.att_unseen_corresp.index(att_graph_index)
    #
    #             cls_graph_index = data.cls_nodes.index(wnids[i])
    #             cls_corresp = data.cls_unseen_corresp.index(cls_graph_index)
    #             n2v[i] = torch.cat((cls_output[cls_corresp], att_output[att_corresp]), -1)
    #         else:
    #             continue
    #
    #     # print(n2v.shape)
    #     cls_embed = n2v[input_index]
    #     # print("cls embed shape:", cls_embed.shape)
    #
    #     h = torch.cat((noise, cls_embed), 1)
    #     h = self.lrelu(self.fc1(h))
    #     h = self.relu(self.fc2(h))
    #     return h, cls_embed


class MLP_G(nn.Module):
    def __init__(self, args):
        super(MLP_G, self).__init__()
        self.fc1 = nn.Linear(args.SemSize + args.NoiseSize, args.NGH)
        self.fc2 = nn.Linear(args.NGH, args.FeaSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        # self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)

        self.apply(weights_init)

    def forward(self, noise, sem):
        h = torch.cat((noise, sem), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h


class MLP_2048_1024_Dropout_G(nn.Module):
    def __init__(self, opt):
        super(MLP_2048_1024_Dropout_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        # self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, 1024)
        self.fc3 = nn.Linear(1024, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        # self.prelu = nn.PReLU()
        # self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.5)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.dropout(self.lrelu(self.fc1(h)))
        h = self.dropout(self.lrelu(self.fc2(h)))
        h = self.fc3(h)
        return h


class MLP_SKIP_G(nn.Module):
    def __init__(self, opt):
        super(MLP_SKIP_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        # self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        # self.fc2 = nn.Linear(opt.ngh, 1024)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.fc_skip = nn.Linear(opt.attSize, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        # self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        # h = self.lrelu(self.fc2(h))
        h = self.relu(self.fc2(h))
        h2 = self.fc_skip(att)
        return h + h2


class MLP_SKIP_D(nn.Module):
    def __init__(self, opt):
        super(MLP_SKIP_D, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.fc_skip = nn.Linear(opt.attSize, opt.ndh)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        h = self.lrelu(self.fc1(h))
        h2 = self.lrelu(self.fc_skip(att))
        h = self.sigmoid(self.fc2(h + h2))
        return h

