import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.init as init


from termcolor import cprint
from time import gmtime, strftime
import numpy as np
import argparse
import os
import glob
import random
import json

# import sys
# sys.path.append('../')

# import Text_GAN.dataset_GBU
# import models
from dataset_GBU import FeatDataLayer, DATA_LOADER
from models import _netD, _netG_att, _param

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ImageNet', help='FLO')
parser.add_argument('--dataroot', default='/home/gyx/ZSL/data', help='path to dataset')
parser.add_argument('--matdataset', default='ImageNet', help='Data in matlab format')
parser.add_argument('--exp_name', default='Exp10', help='path to save ..')


parser.add_argument('--image_embedding', default='res101')
# parser.add_argument('--class_embedding', default='w2v')
parser.add_argument('--class_embedding', default='g2v')



parser.add_argument('--SeenFeaFile', default='Res101_Features/StandardSplit/ILSVRC2012_train')
parser.add_argument('--SeenTestFeaFile', default='Res101_Features/StandardSplit/ILSVRC2012_val')
parser.add_argument('--UnseenFeaFile', default='Res101_Features/StandardSplit/ILSVRC2011')



parser.add_argument('--preprocessing', action='store_true', default=True,
                    help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')

parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
parser.add_argument('--exp_idx', default='', type=str, help='exp idx')
parser.add_argument('--manualSeed', type=int, default=2064, help='manual seed')
parser.add_argument('--resume',  type=str, help='the model to resume')

# for graph embedding
parser.add_argument('--z_dim',  type=int, default=50, help='dimension of the random vector z')

# for word embedding
# parser.add_argument('--z_dim',  type=int, default=100, help='dimension of the random vector z')
parser.add_argument('--disp_interval', type=int, default=20)
parser.add_argument('--save_interval', type=int, default=200)
parser.add_argument('--evl_interval',  type=int, default=40)

opt = parser.parse_args()
print('Running parameters:')
print(json.dumps(vars(opt), indent=4, separators=(',', ':')))

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

""" hyper-parameter  """
opt.GP_LAMBDA = 10    # Gradient penalty lambda
opt.CENT_LAMBDA = 5
opt.REG_W_LAMBDA = 0.001
opt.Adv_LAMBDA = 1

opt.lr = 0.0001
opt.batchsize = 1024  # 512

""" hyper-parameter for testing"""
opt.nSample = 100  # number of fake feature for each unseen class
opt.nSeenSample = 50  # number of fake feature for each seen class

# opt.nSample = 300  # number of fake feature for each class
opt.Knn = 500      # knn: the value of K


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)


def train():
    param = _param()
    dataset = DATA_LOADER(opt)
    param.X_dim = dataset.feature_dim

    data_layer = FeatDataLayer(dataset.train_label.numpy(), dataset.train_feature.numpy(), opt)
    result = Result()
    result_gzsl = Result()

    netG = _netG_att(opt, dataset.text_dim, dataset.feature_dim).cuda()
    netG.apply(weights_init)
    print(netG)
    netD = _netD(dataset.train_cls_num, dataset.feature_dim).cuda()
    netD.apply(weights_init)
    print(netD)



    start_step = 0


    nets = [netG, netD]

    tr_cls_centroid = Variable(torch.from_numpy(dataset.tr_cls_centroid.astype('float32'))).cuda()
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.9))

    for it in range(start_step, 3000+1):
        """ Discriminator """
        for _ in range(5):
            blobs = data_layer.forward()
            feat_data = blobs['data']             # image data
            labels = blobs['labels'].astype(int)  # class labels
            text_feat = np.array([dataset.train_sem[i,:] for i in labels])
            text_feat = Variable(torch.from_numpy(text_feat.astype('float32'))).cuda()
            X = Variable(torch.from_numpy(feat_data)).cuda()
            y_true = Variable(torch.from_numpy(labels.astype('int'))).cuda()
            z = Variable(torch.randn(opt.batchsize, opt.z_dim)).cuda()

            # GAN's D loss
            D_real, C_real = netD(X)
            D_loss_real = torch.mean(D_real)
            C_loss_real = F.cross_entropy(C_real, y_true)
            DC_loss = opt.Adv_LAMBDA *(-D_loss_real + C_loss_real)
            DC_loss.backward()

            # GAN's D loss
            G_sample = netG(z, text_feat).detach()
            D_fake, C_fake = netD(G_sample)
            D_loss_fake = torch.mean(D_fake)


            C_loss_fake = F.cross_entropy(C_fake, y_true)
            DC_loss = opt.Adv_LAMBDA *(D_loss_fake + C_loss_fake)
            DC_loss.backward()

            # train with gradient penalty (WGAN_GP)
            grad_penalty = opt.Adv_LAMBDA * calc_gradient_penalty(netD, X.data, G_sample.data)
            grad_penalty.backward()

            Wasserstein_D = D_loss_real - D_loss_fake
            optimizerD.step()
            reset_grad(nets)

        """ Generator """
        for _ in range(1):
            blobs = data_layer.forward()
            feat_data = blobs['data']  # image data
            labels = blobs['labels'].astype(int)  # class labels
            text_feat = np.array([dataset.train_sem[i, :] for i in labels])
            text_feat = Variable(torch.from_numpy(text_feat.astype('float32'))).cuda()

            X = Variable(torch.from_numpy(feat_data)).cuda()
            y_true = Variable(torch.from_numpy(labels.astype('int'))).cuda()
            z = Variable(torch.randn(opt.batchsize, opt.z_dim)).cuda()

            G_sample = netG(z, text_feat)
            D_fake, C_fake = netD(G_sample)
            _,      C_real = netD(X)

            # GAN's G loss
            G_loss = torch.mean(D_fake)
            # Auxiliary classification loss
            C_loss = (F.cross_entropy(C_real, y_true) + F.cross_entropy(C_fake, y_true)) / 2
            GC_loss = opt.Adv_LAMBDA *(-G_loss + C_loss)

            # Centroid loss
            Euclidean_loss = Variable(torch.Tensor([0.0])).cuda()
            if opt.REG_W_LAMBDA != 0:
                for i in range(dataset.train_cls_num):
                    sample_idx = (y_true == i).data.nonzero().squeeze()
                    if sample_idx.numel() == 0:
                        Euclidean_loss += 0.0
                    else:
                        G_sample_cls = G_sample[sample_idx, :]
                        Euclidean_loss += (G_sample_cls.mean(dim=0) - tr_cls_centroid[i]).pow(2).sum().sqrt()
                Euclidean_loss *= 1.0/dataset.train_cls_num * opt.CENT_LAMBDA

            # ||W||_2 regularization
            reg_loss = Variable(torch.Tensor([0.0])).cuda()
            if opt.REG_W_LAMBDA != 0:
                for name, p in netG.named_parameters():
                    if 'weight' in name:
                        reg_loss += p.pow(2).sum()
                reg_loss.mul_(opt.REG_W_LAMBDA)

            all_loss = GC_loss + Euclidean_loss + reg_loss
            all_loss.backward()
            optimizerG.step()
            reset_grad(nets)

        if it % opt.disp_interval == 0 and it:
            acc_real = (np.argmax(C_real.data.cpu().numpy(), axis=1) == y_true.data.cpu().numpy()).sum() / float(y_true.data.size()[0])
            acc_fake = (np.argmax(C_fake.data.cpu().numpy(), axis=1) == y_true.data.cpu().numpy()).sum() / float(y_true.data.size()[0])

            # log_text = 'Iter-{}; Was_D: {:.3f}; Euc_ls: {:.3f}; reg_ls: {:.3f}; G_loss: {:.3f}; D_loss_real: {:.3f};' \
            #            ' D_loss_fake: {:.3f}; rl: {:.2f}%; fk: {:.2f}%'\
            #             .format(it, Wasserstein_D.data[0],  Euclidean_loss.data[0], reg_loss.data[0],
            #                     G_loss.data[0], D_loss_real.data[0], D_loss_fake.data[0], acc_real * 100, acc_fake * 100)
            log_text = 'Iter-{} *********************'.format(it)
            print(log_text)
            # with open(log_dir, 'a') as f:
            #     f.write(log_text+'\n')



        if it % opt.evl_interval == 0 and it >= 100:
            netG.eval()
            eval_fakefeat_test(it, netG, dataset, param, result)
            # eval_fakefeat_test_Hit(it, netG, dataset, param)
            eval_fakefeat_test_gzsl(it, netG, dataset, param, result_gzsl)



            netG.train()



def save_model(it, netG, netD, random_seed, log, fout):
    torch.save({
        'it': it + 1,
        'state_dict_G': netG.state_dict(),
        'state_dict_D': netD.state_dict(),
        'random_seed': random_seed,
        'log': log,
    }, fout)


def eval_fakefeat_test(it, netG, dataset, param, result):
    gen_feat = np.zeros([0, param.X_dim])
    for i in range(dataset.test_cls_num):
        text_feat = np.tile(dataset.test_sem[i].astype('float32'), (opt.nSample, 1))
        text_feat = Variable(torch.from_numpy(text_feat)).cuda()
        z = Variable(torch.randn(opt.nSample, opt.z_dim)).cuda()
        G_sample = netG(z, text_feat)
        gen_feat = np.vstack((gen_feat, G_sample.data.cpu().numpy()))

    from sklearn.metrics.pairwise import cosine_similarity
    # cosince predict K-nearest Neighbor
    sim = cosine_similarity(dataset.test_unseen_feature, gen_feat)

    # for Hit@1
    idx_mat = np.argsort(-1 * sim, axis=1)
    label_mat = (idx_mat[:, 0:opt.Knn] / opt.nSample).astype(int)
    preds = np.zeros(label_mat.shape[0])
    for i in range(label_mat.shape[0]):
        (values, counts) = np.unique(label_mat[i], return_counts=True)
        preds[i] = values[np.argmax(counts)]


    # produce MCA
    label_T = np.asarray(dataset.test_unseen_label)
    acc = np.zeros(label_T.max() + 1)
    for i in range(label_T.max() + 1):
        acc[i] = (preds[label_T == i] == i).mean()
    acc = acc.mean() * 100

    result.acc_list += [acc]
    result.iter_list += [it]
    result.save_model = False
    if acc > result.best_acc:
        result.best_acc = acc
        result.best_iter = it
        result.save_model = True
    print("{}nn Classifer Accuracy {:.2f}: ".format(opt.Knn, acc))


def eval_fakefeat_test_Hit(it, netG, dataset, param):
    gen_feat = np.zeros([0, param.X_dim])
    for i in range(dataset.test_cls_num):
        text_feat = np.tile(dataset.test_sem[i].astype('float32'), (opt.nSample, 1))
        text_feat = Variable(torch.from_numpy(text_feat)).cuda()
        z = Variable(torch.randn(opt.nSample, opt.z_dim)).cuda()
        G_sample = netG(z, text_feat)
        gen_feat = np.vstack((gen_feat, G_sample.data.cpu().numpy()))

    from sklearn.metrics.pairwise import cosine_similarity
    # cosince predict K-nearest Neighbor
    sim = cosine_similarity(dataset.test_unseen_feature, gen_feat)

    top_retrv = [1, 2, 5]

    # for Hit@1
    idx_mat = np.argsort(-1 * sim, axis=1)


    label_mat = (idx_mat[:, 0:opt.Knn] / opt.nSample).astype(int)
    preds_5 = np.zeros((label_mat.shape[0], 5))
    preds = np.zeros(label_mat.shape[0])

    for i in range(label_mat.shape[0]):
        (values, counts) = np.unique(label_mat[i], return_counts=True)
        preds[i] = values[np.argmax(counts)]
        # print("count length:", len(counts))
        counts_sort_index = np.argsort(-counts)

        preds_5[i] = values[counts_sort_index[:5]]


    # produce MCA

    label_T = np.asarray(dataset.test_unseen_label)
    acc = np.zeros(label_T.max() + 1)

    acc_5 = np.zeros((label_T.max() + 1, 3), dtype=np.float32)

    for i in range(label_T.max() + 1):
        pres = preds_5[label_T == i]
        # print(pres.shape)
        hits = np.zeros(3)
        for p in range(pres.shape[0]):
            for k in range(len(top_retrv)):
                current_len = top_retrv[k]

                for sort_id in range(current_len):
                    # print("predict:", pres[p][sort_id])
                    # print("label:", i)
                    if int(pres[p][sort_id]) == i:
                        hits[k] = hits[k] + 1
                        break
        acc_5[i] = hits/pres.shape[0]
        acc[i] = (preds[label_T == i] == i).mean()
    acc = acc.mean() * 100
    acc_5 = acc_5.mean(axis=0)

    # result.acc_list += [acc]
    # result.iter_list += [it]
    # result.save_model = False
    # if acc > result.best_acc:
        # result.best_acc = acc
        # result.best_iter = it
        # result.save_model = True
    print("{}nn Classifer Accuracy {:.2f}: ".format(opt.Knn, acc))
    print('Hit acc: ', ['{:.2f}'.format(i * 100) for i in acc_5])



def eval_fakefeat_test_gzsl(it, netG, dataset, param, result):
    from sklearn.metrics.pairwise import cosine_similarity
    gen_feat_train_cls = np.zeros([0, param.X_dim])
    for i in range(dataset.train_cls_num):
        text_feat = np.tile(dataset.train_sem[i].astype('float32'), (opt.nSeenSample, 1))
        text_feat = Variable(torch.from_numpy(text_feat)).cuda()
        z = Variable(torch.randn(opt.nSeenSample, opt.z_dim)).cuda()
        G_sample = netG(z, text_feat)
        gen_feat_train_cls = np.vstack((gen_feat_train_cls, G_sample.data.cpu().numpy()))

    gen_feat_test_cls = np.zeros([0, param.X_dim])
    for i in range(dataset.test_cls_num):
        text_feat = np.tile(dataset.test_sem[i].astype('float32'), (opt.nSample, 1))
        text_feat = Variable(torch.from_numpy(text_feat)).cuda()
        z = Variable(torch.randn(opt.nSample, opt.z_dim)).cuda()
        G_sample = netG(z, text_feat)
        gen_feat_test_cls = np.vstack((gen_feat_test_cls, G_sample.data.cpu().numpy()))

    """  S -> T
    """
    sim = cosine_similarity(dataset.test_seen_feature, np.vstack((gen_feat_train_cls, gen_feat_test_cls)))
    idx_mat = np.argsort(-1 * sim, axis=1)
    label_mat = (idx_mat[:, 0:opt.Knn] / opt.nSeenSample).astype(int)
    preds = np.zeros(label_mat.shape[0])
    for i in range(label_mat.shape[0]):
        (values, counts) = np.unique(label_mat[i], return_counts=True)
        preds[i] = values[np.argmax(counts)]
    # produce MCA
    label_T = np.asarray(dataset.test_seen_label)
    acc = np.zeros(label_T.max() + 1)
    for i in range(label_T.max() + 1):
        acc[i] = (preds[label_T == i] == i).mean()
    acc_S_T = acc.mean() * 100

    """  U -> T
    """
    sim = cosine_similarity(dataset.test_unseen_feature, np.vstack((gen_feat_test_cls, gen_feat_train_cls)))
    idx_mat = np.argsort(-1 * sim, axis=1)
    label_mat = (idx_mat[:, 0:opt.Knn] / opt.nSample).astype(int)
    preds = np.zeros(label_mat.shape[0])
    for i in range(label_mat.shape[0]):
        (values, counts) = np.unique(label_mat[i], return_counts=True)
        preds[i] = values[np.argmax(counts)]
    # produce MCA
    label_T = np.asarray(dataset.test_unseen_label)
    acc = np.zeros(label_T.max() + 1)
    for i in range(label_T.max() + 1):
        acc[i] = (preds[label_T == i] == i).mean()
    acc_U_T = acc.mean() * 100
    acc = (2 * acc_S_T * acc_U_T) / (acc_S_T + acc_U_T)

    result.acc_list += [acc]
    result.iter_list += [it]
    result.save_model = False
    if acc > result.best_acc:
        result.best_acc = acc
        result.best_iter = it
        result.best_acc_S_T = acc_S_T
        result.best_acc_U_T = acc_U_T
        result.save_model = True

    print("H (Mean) {:.2f}  Seen {:.2f}  Unseen {:.2f}  ".format(acc, acc_S_T, acc_U_T))


class Result(object):
    def __init__(self):
        self.best_acc = 0.0
        self.best_iter = 0.0
        self.best_acc_S_T = 0.0
        self.best_acc_U_T = 0.0
        self.acc_list = []
        self.iter_list = []


def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        init.xavier_normal(m.weight.data)
        init.constant(m.bias, 0.0)


def reset_grad(nets):
    for net in nets:
        net.zero_grad()


def label2mat(labels, y_dim):
    c = np.zeros([labels.shape[0], y_dim])
    for idx, d in enumerate(labels):
        c[idx, d] = 1
    return c


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(opt.batchsize, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates, _ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.GP_LAMBDA
    return gradient_penalty


if __name__ == "__main__":
    train()

