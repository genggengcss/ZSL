from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import util
import classifier_pretrain
import classifier_cls
import sys
import model
import numpy as np
import time
import config_args


# parser = argparse.ArgumentParser()
# '''
# Data loading
# '''
# parser.add_argument('--DATADIR', default='/home/gyx/ZSL/data', help='path to dataset')
# parser.add_argument('--Workers', default=2, help='number of data loading workers')
# parser.add_argument('--DATASET', default='ImageNet', help='for imagenet')
#
# parser.add_argument('--ProposedSplit', action='store_true', default=False, help='the class split doesnot follow standard split')
# parser.add_argument('--SeenFeaFile', default='Res101_Features/StandardSplit/ILSVRC2012_train', help='')
# parser.add_argument('--SeenTestFeaFile', default='Res101_Features/StandardSplit/ILSVRC2012_val', help='')
# parser.add_argument('--UnseenFeaFile', default='Res101_Features/StandardSplit/ILSVRC2011', help='')
#
# parser.add_argument('--SemEmbed', default='n2v', help='the type of class embedding to input')
# parser.add_argument('--SemFile', default='n2v-550.mat', help='the file to store class embedding')
# parser.add_argument('--SemSize', default=100, help='size of semantic features')
# parser.add_argument('--NoiseSize', default=100, help='size of semantic features')
# parser.add_argument('--FeaSize', default=2048, help='size of visual features')
#
# parser.add_argument('--ExpName', default='Exp5', help='the folder to store class file and class embedding file')
#
# parser.add_argument('--Unseen_NSample', default=50, help='extract the subset of unseen samples')
# parser.add_argument('--PerClassAcc', action='store_true', default=False, help='testing the accuracy of each class')
#
# '''
# Generator and Discriminator
# '''
# parser.add_argument('--NetG_Path', default='', help='path to netG (to continue training)')
# parser.add_argument('--NetD_Path', default='', help='path to netD (to continue training)')
# parser.add_argument('--Pretrained_Classifier', default='', help='path to pretrain classifier (to continue training)')
# parser.add_argument('--NetG_Name', default='MLP_G', help='')
# parser.add_argument('--NetD_Name', default='MLP_CRITIC', help='')
# parser.add_argument('--NGH', default=4096, help='size of the hidden units in generator')
# parser.add_argument('--NDH', default=4096, help='size of the hidden units in discriminator')
# parser.add_argument('--Critic_Iter', default=5, help='critic iteration of discriminator, default=5, following WGAN-GP setting')
# parser.add_argument('--GP_Weight', type=float, default=10, help='gradient penalty regularizer, default=10, the completion of Lipschitz Constraint in WGAN-GP')
# parser.add_argument('--Cls_Weight', default=0.01, help='loss weight for the supervised classification loss')
# parser.add_argument('--Proto_Weight1', default=0.01, help='# para for soul samples\'s regularization1')
# parser.add_argument('--Proto_Weight2', default=0.01, help='para for soul samples\'s regularization2')
# parser.add_argument('--NClusters', default=5, help='number of real sample clusters')
# parser.add_argument('--Cluster_Save_Dir', default='save_cluster_5', help='')
# parser.add_argument('--NSynClusters', default=20, help='number of fake clusters?')
# parser.add_argument('--SynNum', default=500, help='number of features generating for each unseen class; awa_default = 300')
# parser.add_argument('--SeenTestNum', default=500, help='number of features for testing seen classes')
#
# '''
# Training Parameter
# '''
# parser.add_argument('--GZSL', action='store_true', default=False, help='enable generalized zero-shot learning')
# parser.add_argument('--PreProcess', default=True, help='enbale MinMaxScaler on visual features, default=True')
# parser.add_argument('--Standardization', default=False, help='')
# parser.add_argument('--Cross_Validation', default=False, help='enable cross validation mode')
# parser.add_argument('--Cuda', default=True, help='')
# parser.add_argument('--NGPU', default=1, help='number of GPUs to use')
# parser.add_argument('--CUDA_DEVISE', default='3', help='')
# parser.add_argument('--ManualSeed', type=int, default=9416, help='')
# parser.add_argument('--BatchSize', default=4096, help='')
# parser.add_argument('--Epoch', default=100, help='')
# parser.add_argument('--LR', default=0.0001, help='learning rate to train GAN')
# parser.add_argument('--Cls_LR', default=0.001, help='after generating unseen features, the learning rate for training softmax classifier')
# parser.add_argument('--Ratio', default=0.1, help='ratio of easy samples')
# parser.add_argument('--Beta', default=0.5, help='beta for adam, default=0.5')
#
# parser.add_argument('--OutFolder', default=, help='')
# parser.add_argument('--OutName', default='imagenet', help='')
# parser.add_argument('--SaveEvery', default=100, help='')
# parser.add_argument('--PrintEvery', default=1, help='')
# parser.add_argument('--ValEvery', default=1, help='')
# parser.add_argument('--StartEvery', default=0, help='')
#
# args = parser.parse_args()
#
# if args.ProposedSplit is None:
#     args.SeenFeaFile = 'Res101_Features/StandardSplit/ILSVRC2012_train'
#     args.SeenTestFeaFile = 'Res101_Features/StandardSplit/ILSVRC2012_val'
#     args.UnseenFeaFile = 'Res101_Features/StandardSplit/ILSVRC2011'
# else:
#     args.SeenFeaFile = 'Res101_Features/ProposedSplit/Seen_train'
#     args.SeenTestFeaFile = 'Res101_Features/ProposedSplit/Seen_val'
#     args.UnseenFeaFile = 'Res101_Features/ProposedSplit/Unseen'
#
# if args.SemEmbed != 'n2v':
#     args.SemEmbed = 'w2v'
#     args.SemFile = 'w2v.mat'
#     args.SemSize = 500
#     args.NoiseSize = 500
# else:
#     args.SemFile = os.path.join('KG-GAN', args.ExpName, args.SemFile)

# load seen classes and unseen classes

def GetNowTime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

print(GetNowTime())
print('Begin run!!!')
since = time.time()

args = config_args.loadArgums()
print("Params:")
print('ProposedSplit:{:s}, SemEmbed:{:s}, ExpName: {:s}, Proto_Weight1={:.5f}, Proto_Weight2={:.5f}, NClusters={:d}, SynNum={:d}, GZSL:{:s}, ManualSeed:{:d}'.format(
        str(args.ProposedSplit), args.SemEmbed, args.ExpName, args.Proto_Weight1, args.Proto_Weight2, args.NClusters, args.SynNum, str(args.GZSL), args.ManualSeed))
# parser.add_argument('--dataset', default=args.DATASET, help='ImageNet')
# parser.add_argument('--dataroot', default=args.DATADIR, help='path to dataset')

# parser.add_argument('--matdataset', default=True, help='Data in matlab format')
# parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
# parser.add_argument('--preprocessing', action='store_true', default=True,
#                     help='enbale MinMaxScaler on visual features')
# parser.add_argument('--standardization', action='store_true', default=False)
# parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
# parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')

# parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)


# parser.add_argument('--image_embedding', default='res101')
# parser.add_argument('--class_embedding', default='att')

# parser.add_argument('--syn_num', type=int, default=300, help='number features to generate per class')
# parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
# parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
# parser.add_argument('--attSize', type=int, default=103, help='size of semantic features')
# parser.add_argument('--nz', type=int, default=103, help='size of the latent z vector')
# parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
# parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator')
# parser.add_argument('--nepoch', type=int, default=30, help='number of epochs to train for')
# parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
# parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
# parser.add_argument('--cls_weight', type=float, default=0.01, help='weight of the classification loss')
# parser.add_argument('--lr', type=float, default=0.00001, help='learning rate to train GANs ')
# parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
# parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

# parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')


# parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
# parser.add_argument('--netG', default='', help="")
# parser.add_argument('--netD', default='', help="path to netD (to continue training)")
# parser.add_argument('--netG_name', default='MLP_G')
# parser.add_argument('--netD_name', default='MLP_CRITIC')
# parser.add_argument('--outf', default='./checkpoint/', help='folder to output data and model checkpoints')
# parser.add_argument('--outname', default='awa', help='folder to output data and model checkpoints')
# parser.add_argument('--save_every', type=int, default=100)
# parser.add_argument('--print_every', type=int, default=1)
# parser.add_argument('--val_every', type=int, default=1)
# parser.add_argument('--start_epoch', type=int, default=0)

# parser.add_argument('--manualSeed', type=int, default=9182, help='manual seed')
# parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
# parser.add_argument('--ratio', type=float, default=0.1, help='ratio of easy samples')
# parser.add_argument('--proto_param1', type=float, default=3e-2, help='proto param 1')
# parser.add_argument('--proto_param2', type=float, default=3e-5, help='proto param 2')
# parser.add_argument('--loss_syn_num', type=int, default=20, help='number of real clusters')
# parser.add_argument('--n_clusters', type=int, default=3, help='number of real clusters')




# config = parser.parse_args()
# print('Params: GZSL={:s}, ratio={:.1f}, cls_weight={:.4f}, proto_weight1={:.5f}, proto_weight2={:.5f}'.format(
#      str(args.GZSL), args.Ratio, args.Cls_Weight, args.Proto_Weight1, args.Proto_Weight2))
sys.stdout.flush()

os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_DEVISE

if args.ManualSeed is None:
    ManualSeed = random.randint(1, 10000)
else:
    ManualSeed = args.ManualSeed
print("Random Seed: ", ManualSeed)

# set random seed
random.seed(ManualSeed)
torch.manual_seed(ManualSeed)
if args.Cuda:
    torch.cuda.manual_seed_all(ManualSeed)

cudnn.benchmark = True


if torch.cuda.is_available() and not args.Cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data
data = util.DATA_LOADER(args)
print("Training samples: ", data.ntrain)  # number of training samples

# initialize generator and discriminator
netG = model.MLP_G(args)
if args.NetG_Path != '':
    netG.load_state_dict(torch.load(args.NetG_Path))  # load the trained model: model.load_state_dict(torch.load(PATH))
# print(netG)

netD = model.MLP_CRITIC(args)
if args.NetD_Path != '':
    netD.load_state_dict(torch.load(args.NetD_Path))
# print(netD)

# classification loss, Equation (4) of the paper
cls_criterion = nn.NLLLoss()  # cross entropy loss

input_fea = torch.FloatTensor(args.BatchSize, args.FeaSize)  # (64, 2048)
input_sem = torch.FloatTensor(args.BatchSize, args.SemSize)  # (64, 500)
noise = torch.FloatTensor(args.BatchSize, args.NoiseSize)  # (64, 500)
one = torch.FloatTensor([1])
mone = one * -1
input_label = torch.LongTensor(args.BatchSize)

if args.Cuda:
    netD.cuda()
    netG.cuda()
    input_fea = input_fea.cuda()
    noise, input_sem = noise.cuda(), input_sem.cuda()
    one = one.cuda()
    mone = mone.cuda()
    cls_criterion.cuda()
    input_label = input_label.cuda()


def sample():
    batch_feature, batch_label, batch_sem = data.next_batch(args.BatchSize)

    input_fea.copy_(batch_feature)
    input_sem.copy_(batch_sem)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))


def generate_syn_feature(netG, classes, semantic, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, args.FeaSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_sem = torch.FloatTensor(num, args.SemSize)
    syn_noise = torch.FloatTensor(num, args.NoiseSize)
    if args.Cuda:
        syn_sem = syn_sem.cuda()
        syn_noise = syn_noise.cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_sem = semantic[iclass]
        syn_sem.copy_(iclass_sem.repeat(num, 1))
        syn_noise.normal_(0, 1)
        output = netG(Variable(syn_noise, volatile=True), Variable(syn_sem, volatile=True))
        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)

    return syn_feature, syn_label

def generate_syn_feature_with_grad(netG, classes, semantic, num):
    nclass = classes.size(0)
    # syn_feature = torch.FloatTensor(nclass*num, args.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_sem = torch.FloatTensor(nclass * num, args.SemSize)
    syn_noise = torch.FloatTensor(nclass * num, args.NoiseSize)
    if args.Cuda:
        syn_sem = syn_sem.cuda()
        syn_noise = syn_noise.cuda()
        syn_label = syn_label.cuda()
    syn_noise.normal_(0, 1)
    for i in range(nclass):
        iclass = classes[i]
        iclass_sem = semantic[iclass]
        syn_sem.narrow(0, i * num, num).copy_(iclass_sem.repeat(num, 1))
        syn_label.narrow(0, i * num, num).fill_(iclass)
    syn_feature = netG(Variable(syn_noise), Variable(syn_sem))
    return syn_feature, syn_label.cpu()

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i

    return mapped_label


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an configional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    if y is None:
        dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=args.LR, betas=(args.Beta, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=args.LR, betas=(args.Beta, 0.999))

# the last item of equation (2)
def calc_gradient_penalty(netD, real_data, fake_data, input_sem):
    # print real_data.size()
    alpha = torch.rand(args.BatchSize, 1)
    alpha = alpha.expand(real_data.size())
    if args.Cuda:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if args.Cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, Variable(input_sem))

    ones = torch.ones(disc_interpolates.size())
    if args.Cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    # args.GP_Weight = 10
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.GP_Weight
    return gradient_penalty

# syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.semantic, args.SynNum)
# train_X = torch.cat((data.train_feature, syn_feature), 0)
# train_Y = torch.cat((data.train_label, syn_label), 0)
# print("train Y:", train_Y)
# classes = torch.cat((data.seenclasses, data.unseenclasses), 0)
# print("train Y:", classes)
# nclass = args.NClassAll
# cls = classifier_cls.CLASSIFIER(train_X, util.map_label(train_Y, classes), data, nclass, args.Cuda,
#                                         args.Cls_LR, 0.5, 50, 2 * args.BatchSize,
#                                         True)

# syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.semantic, args.SynNum)
# cls = classifier_cls.CLASSIFIER(args, syn_feature, util.map_label(syn_label, data.unseenclasses), data,
#                                      data.unseenclasses.size(0), args.Cuda, args.Cls_LR, 0.5, 50, 10*args.SynNum,
#                                      False, args.Ratio)

# train a classifier on seen classes, obtain \theta of Equation (4)
pretrain_cls = classifier_pretrain.CLASSIFIER(data.train_feature, util.map_label(data.train_label, data.seenclasses),
                                     data.seenclasses.size(0), args.FeaSize, args.Cuda, 0.001, 0.5, 100, 2*args.BatchSize,
                                     args.Pretrained_Classifier)

# freeze the classifier during the optimization
for p in pretrain_cls.model.parameters():  # set requires_grad to False
    p.requires_grad = False



for epoch in range(args.Epoch):
    FP = 0
    mean_lossD = 0
    mean_lossG = 0

    for i in range(0, data.ntrain, args.BatchSize):
        # print("batch...", i)
        # iteratively train the generator and discriminator
        for p in netD.parameters():
            p.requires_grad = True

        # DISCRIMINATOR
        # args.critic_iter = 5, following WGAN-GP
        for iter_d in range(args.Critic_Iter):
            sample()  # sample by batch
            netD.zero_grad()
            # torch.gt: compare the 'input_res[1]' and '0' element by element
            sparse_real = args.FeaSize - input_fea[1].gt(0).sum()  # non sparse number
            input_feav = Variable(input_fea)
            input_semv = Variable(input_sem)

            # loss of real data
            criticD_real = netD(input_feav, input_semv)
            criticD_real = criticD_real.mean()
            criticD_real.backward(mone)
            # loss of generated data
            noise.normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev, input_semv)   # generate samples
            fake_norm = fake.data[0].norm()
            sparse_fake = fake.data[0].eq(0).sum()
            # detach(): return a new variable, do not compute gradient for it
            criticD_fake = netD(fake.detach(), input_semv)
            criticD_fake = criticD_fake.mean()
            criticD_fake.backward(one)

            # loss with Lipschitz constraint
            gradient_penalty = calc_gradient_penalty(netD, input_fea, fake.data, input_sem)
            gradient_penalty.backward()

            Wasserstein_D = criticD_real - criticD_fake
            # Final Loss of Discriminator
            D_cost = criticD_fake - criticD_real + gradient_penalty
            optimizerD.step()

        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = False  # avoid computation
        # GENERATOR
        netG.zero_grad()
        input_semv = Variable(input_sem)
        noise.normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev, input_semv)
        criticG_fake = netD(fake, input_semv)
        criticG_fake = criticG_fake.mean()
        G_cost = -criticG_fake
        # classification loss
        c_errG = cls_criterion(pretrain_cls.model(fake), Variable(input_label))

        # labels = Variable(input_label.view(args.BatchSize, 1))
        # real_proto = Variable(data.real_proto.cuda())  # soul sample
        #
        # dists1 = pairwise_distances(fake, real_proto)
        #
        # min_idx1 = torch.zeros(args.BatchSize, data.train_cls_num)
        # for i in range(data.train_cls_num):
        #     min_idx1[:, i] = torch.min(dists1.data[:, i*args.NClusters:(i+1)*args.NClusters], dim=1)[1] + i*args.NClusters
        # min_idx1 = Variable(min_idx1.long().cuda())
        # L_R1 constraint
        # loss1 = dists1.gather(1, min_idx1).gather(1, labels).squeeze().view(-1).mean()
        #
        # seen_feature, seen_label = generate_syn_feature_with_grad(netG, data.seenclasses, data.semantic, args.NSynClusters)
        # seen_mapped_label = map_label(seen_label, data.seenclasses)
        # transform_matrix = torch.zeros(data.train_cls_num, seen_feature.size(0))  # 150x7057
        # # print("syn feature size:", seen_feature.size(0))  # 4980
        # for i in range(data.train_cls_num):
        #     sample_idx = (seen_mapped_label == i).nonzero().squeeze()
        #     if sample_idx.numel() == 0:
        #         continue
        #     else:
        #         cls_fea_num = sample_idx.numel()
        #         transform_matrix[i][sample_idx] = 1 / cls_fea_num * torch.ones(1, cls_fea_num).squeeze()
        # transform_matrix = Variable(transform_matrix.cuda())
        # fake_proto = torch.mm(transform_matrix, seen_feature)  # 150x2048
        # dists2 = pairwise_distances(fake_proto,Variable(data.real_proto.cuda()))  # 150 x 450
        # min_idx2 = torch.zeros(data.train_cls_num, data.train_cls_num)
        # for i in range(data.train_cls_num):
        #     min_idx2[:, i] = torch.min(dists2.data[:, i*args.NClusters:(i+1)*args.NClusters], dim=1)[1] + i*args.NClusters
        # min_idx2 = Variable(min_idx2.long().cuda())
        # lbl_idx = Variable(torch.LongTensor(list(range(data.train_cls_num))).cuda())
        # L_R2 constraint
        # loss2 = dists2.gather(1, min_idx2).gather(1, lbl_idx.unsqueeze(1)).squeeze().mean()
        # Final Loss of generator
        # errG = G_cost + args.Cls_Weight * c_errG + args.Proto_Weight1 * loss1 + args.Proto_Weight2 * loss2
        errG = G_cost + args.Cls_Weight * c_errG

        errG.backward()
        optimizerG.step()

    print('EP[%d/%d]******************************************************' % (epoch, args.Epoch))

    # evaluate the model, set G to evaluation mode
    netG.eval()
    # train_X: input features (of unseen or seen) for training classifier2 in testing stage
    # train_Y: training labels
    # Generalized zero-shot learning
    if args.GZSL:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.semantic, args.SynNum)
        train_X = torch.cat((data.train_feature1, syn_feature), 0)
        train_Y = torch.cat((data.train_label1, syn_label), 0)
        classes = torch.cat((data.seenclasses, data.unseenclasses), 0)
        nclass = classes.size(0)
        cls = classifier_cls.CLASSIFIER(args, train_X, util.map_label(train_Y, classes), data, nclass, args.Cuda,
                                        args.Cls_LR, 0.5, 50, 5 * args.BatchSize,
                                        True)

        # syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.semantic, args.SynNum)
        # train_X = torch.cat((data.train_feature, syn_feature), 0)
        # train_Y = torch.cat((data.train_label, syn_label), 0)
        # nclass = args.NClassAll
        # cls = classifier_cls.CLASSIFIER(train_X, util.map_label(train_Y, classes), data, nclass, args.Cuda, args.Cls_LR, 0.5, 50, 2*args.BatchSize, True)
        # print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls.acc_unseen, cls.acc_seen, cls.H))
    # Zero-shot learning
    else:
        # synthesize samples of unseen classes, for training classifier2 in testing stage
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.semantic, args.SynNum)
        cls = classifier_cls.CLASSIFIER(args, syn_feature, util.map_label(syn_label, data.unseenclasses), data,
                                     data.unseenclasses.size(0), args.Cuda, args.Cls_LR, 0.5, 50, 10*args.SynNum,
                                     False, args.Ratio, epoch)
        # acc = cls.acc
        # print('unseen class accuracy= ', cls.acc)
    del cls
    cls = None
    # reset G to training mode
    netG.train()
    sys.stdout.flush()

# time_elapsed = time.time() - since
print('End run!!!')
# print('Time Elapsed: {}'.format(time_elapsed))
print(GetNowTime())

