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
import numpy as np
import time
import json


import sys
sys.path.append('../../')
from KG_GAN.Exp2 import config_args
from KG_GAN.Exp2 import model
from KG_GAN.Exp2 import classifier_cls
from KG_GAN.Exp2 import classifier_pretrain
from KG_GAN.Exp2 import util




def GetNowTime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

print(GetNowTime())
print('Begin run!!!')
since = time.time()

args = config_args.loadArgums()
print("Params:")
print('ProposedSplit:{:s}, SemEmbed:{:s}, ExpName: {:s}, Proto_Weight1={:.5f}, Proto_Weight2={:.5f}, NClusters={:d}, SynNum={:d}, GZSL:{:s}, ManualSeed:{:d}'.format(
        str(args.ProposedSplit), args.SemEmbed, args.ExpName, args.Proto_Weight1, args.Proto_Weight2, args.NClusters, args.SynNum, str(args.GZSL), args.ManualSeed))

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
netG = model.MLP_G_GCN(args)
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
# input_sem = torch.FloatTensor(args.BatchSize, args.SemSize)  # (64, 500)
noise = torch.FloatTensor(args.BatchSize, args.NoiseSize)  # (64, 500)
one = torch.FloatTensor([1])
mone = one * -1
input_label = torch.LongTensor(args.BatchSize)
input_index = torch.LongTensor(args.BatchSize)
n2v = torch.FloatTensor(2549, 50)  # (64, 500)

# input_cls_feat = torch.FloatTensor(161, 300)
# input_att_feat = torch.FloatTensor(284, 300)
# input_cls_adj = torch.FloatTensor(161, 161)
# input_att_adj = torch.FloatTensor(284, 284)

input_cls_feat = data.cls_feat
# input_att_feat = data.att_feat
input_cls_adj = data.cls_adj
# input_att_adj = data.att_adj

if args.Cuda:
    netD.cuda()
    netG.cuda()
    input_fea = input_fea.cuda()
    # noise, input_sem = noise.cuda(), input_sem.cuda()
    noise = noise.cuda()


    # input_cls_feat, input_att_feat = input_cls_feat.cuda(), input_att_feat.cuda()
    # input_cls_adj, input_att_adj = input_cls_adj.cuda(), input_att_adj.cuda()
    input_cls_feat = input_cls_feat.cuda()
    input_cls_adj = input_cls_adj.cuda()

    one = one.cuda()
    mone = mone.cuda()
    cls_criterion.cuda()
    input_label = input_label.cuda()
    input_index = input_index.cuda()
    n2v = n2v.cuda()


def sample():
    batch_feature, batch_label = data.next_batch(args.BatchSize)

    input_fea.copy_(batch_feature)
    input_index.copy_(batch_label)
    # input_sem.copy_(batch_sem)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))


def generate_syn_feature2(netG, classes, data, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, args.FeaSize)
    syn_label = torch.LongTensor(nclass * num)
    # syn_sem = torch.FloatTensor(num, args.SemSize)
    syn_noise = torch.FloatTensor(num, args.NoiseSize)
    iclass_index = torch.LongTensor(num)
    if args.Cuda:
        # syn_sem = syn_sem.cuda()
        iclass_index = iclass_index.cuda()
        syn_noise = syn_noise.cuda()

    for i in range(nclass):
        iclass = classes[i]
        # iclass_sem = semantic[iclass]
        # syn_sem.copy_(iclass_sem.repeat(num, 1))
        iclass_index.copy_(iclass.repeat(num))
        syn_noise.normal_(0, 1)
        # output = netG(Variable(syn_noise, volatile=True), Variable(syn_sem, volatile=True))


        # output, cls_embed = netG(Variable(syn_noise, volatile=True), data, iclass_index, input_cls_feat, input_cls_adj, input_att_feat,
        # input_att_adj, n2v)

        output, cls_embed = netG(Variable(syn_noise, volatile=True), data, iclass_index, input_cls_feat, input_cls_adj, n2v)


        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)

    return syn_feature, syn_label

def generate_syn_feature(netG, classes, data, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, args.FeaSize)
    syn_label = torch.LongTensor(nclass * num)
    noise = torch.FloatTensor(num, args.NoiseSize)
    syn_noise = torch.FloatTensor(nclass * num, args.NoiseSize)
    iclass_index = torch.LongTensor(nclass * num)
    if args.Cuda:
        iclass_index = iclass_index.cuda()
        syn_noise = syn_noise.cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_index.narrow(0, i * num, num).fill_(iclass)
        syn_label.narrow(0, i * num, num).fill_(iclass)
        noise.normal_(0, 1)
        syn_noise.narrow(0, i * num, num).copy_(noise)

    # syn_noise.normal_(0, 1)

    # output, cls_embed = netG(Variable(syn_noise, volatile=True), data, iclass_index, input_cls_feat, input_cls_adj,
    #                          input_att_feat,
    #                          input_att_adj, n2v)
    output, cls_embed = netG(Variable(syn_noise, volatile=True), data, iclass_index, input_cls_feat, input_cls_adj,
                              n2v)
    syn_feature.copy_(output.data.cpu())

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
# optimizerG = optim.Adam(netG.parameters(), lr=args.LR, betas=(args.Beta, 0.999))



params = [
    {'params': netG.fc1.parameters(), 'lr': args.LR},
    {'params': netG.fc2.parameters(), 'lr': args.LR},
    {'params': netG.gcn1.parameters(), 'lr': args.LR * 100},
    {'params': netG.gcn2.parameters(), 'lr': args.LR * 100},
]
optimizerG = optim.Adam(params, betas=(args.Beta, 0.999))




    # netG.parameters(), lr=args.LR, betas=(args.Beta, 0.999))



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



# train a classifier on seen classes, obtain \theta of Equation (4)
pretrain_cls = classifier_pretrain.CLASSIFIER(data.train_feature, util.map_label(data.train_label, data.seenclasses),
                                     data.seenclasses.size(0), args.FeaSize, args.Cuda, 0.001, 0.5, 100, 10*args.BatchSize,
                                     args.Pretrained_Classifier)

# freeze the classifier during the optimization
for p in pretrain_cls.model.parameters():  # set requires_grad to False
    p.requires_grad = False




for epoch in range(args.Epoch):
    FP = 0
    mean_lossD = 0
    mean_lossG = 0

    for i in range(0, data.ntrain, args.BatchSize*20):
        print("batch..=========================================================.", i)
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
            # input_semv = Variable(input_sem)

            # loss of generated data

            nois = torch.FloatTensor(args.BatchSize, args.NoiseSize)
            noise = torch.FloatTensor(args.BatchSize*20, args.NoiseSize)

            for i in range(20):
                nois.normal_(0, 1)
                noise.narrow(0, i * args.BatchSize, args.BatchSize).copy_(nois)

            # noise.normal_(0, 1)
            noisev = Variable(noise)
            # fake, cls_embed = netG(noisev, data, input_index, input_cls_feat, input_cls_adj, input_att_feat,
            #                        input_att_adj, n2v)   # generate samples
            fake, cls_embed = netG(noisev, data, input_index, input_cls_feat, input_cls_adj, n2v)
            fake_norm = fake.data[0].norm()
            sparse_fake = fake.data[0].eq(0).sum()
            # detach(): return a new variable, do not compute gradient for it
            criticD_fake = netD(fake.detach(), cls_embed)
            criticD_fake = criticD_fake.mean()

            criticD_fake.backward(one, retain_graph=True)

            # loss of real data
            criticD_real = netD(input_feav, cls_embed)
            criticD_real = criticD_real.mean()
            # criticD_real.backward(mone)
            # criticD_real = -criticD_real.mean()
            # print(criticD_real)
            criticD_real.backward(mone, retain_graph=True)
# criticD_real


            # loss with Lipschitz constraint
            gradient_penalty = calc_gradient_penalty(netD, input_fea, fake.data, cls_embed)
            gradient_penalty.backward(retain_graph=True)

            Wasserstein_D = criticD_real - criticD_fake
            # Final Loss of Discriminator
            D_cost = criticD_fake - criticD_real + gradient_penalty
            optimizerD.step()

        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = False  # avoid computation


        # GENERATOR
        netG.zero_grad()


        # input_semv = Variable(input_sem)
        noise.normal_(0, 1)
        noisev = Variable(noise)
        # fake, cls_embed = netG(noisev, data, input_index, input_cls_feat, input_cls_adj, input_att_feat, input_att_adj, n2v)
        fake, cls_embed = netG(noisev, data, input_index, input_cls_feat, input_cls_adj, n2v)

        criticG_fake = netD(fake, cls_embed)
        criticG_fake = criticG_fake.mean()
        G_cost = -criticG_fake
        # classification loss
        c_errG = cls_criterion(pretrain_cls.model(fake), Variable(input_label))


        errG = G_cost + args.Cls_Weight * c_errG

        errG.backward(retain_graph=True)
        optimizerG.step()

    print('EP[%d/%d]******************************************************' % (epoch, args.Epoch))

    # evaluate the model, set G to evaluation mode
    netG.eval()
    cls_embeds = cls_embed.data
    # train_X: input features (of unseen or seen) for training classifier2 in testing stage
    # train_Y: training labels
    # Generalized zero-shot learning
    if args.GZSL:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, cls_embeds, args.SynNum)
        train_X = torch.cat((data.train_feature1, syn_feature), 0)
        train_Y = torch.cat((data.train_label1, syn_label), 0)
        classes = torch.cat((data.seenclasses, data.unseenclasses), 0)
        nclass = classes.size(0)
        cls = classifier_cls.CLASSIFIER(args, train_X, util.map_label(train_Y, classes), data, nclass, args.Cuda,
                                        args.Cls_LR, 0.5, 50, 5 * args.BatchSize,
                                        True)


    # Zero-shot learning
    else:
        # synthesize samples of unseen classes, for training classifier2 in testing stage
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data, args.SynNum)
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

