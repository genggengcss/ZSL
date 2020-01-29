import numpy as np
import random
import scipy, argparse, os, sys, csv, io, time
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score,precision_score
import re
from torch.utils.tensorboard import SummaryWriter
import heapq

from src.data_reader import data_reader, get_vec_mat
from src.count_tools import macro_acc
import DeVise_test_GZSL


import warnings
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
# /home/gyx/ZSL/data/ImageNet/Res101_Features/StandardSplit


'''
Devise: a linear transformation layer from visual feature to semantic feature;
when training, the input is visual feature and the output is semantic feature (word embedding, attribute or trained kg embedding)
visual feature: resnet101, dimension: 2048 (as input dims);
semantic feature: word embedding / kg embedding, dimension: 500/100 (as output dims).
'''
input_dims = 2048
output_dims = 100
p = 0.5  # drop out


# model training
batch_size = 64
lr = 1e-3
wds = 1e-5
epoch_num = 100
GPU = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random_seed = 12345

class devise(nn.Module):
    def __init__(self):
        super(devise, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(input_dims),
            nn.Dropout(p),
            nn.Linear(in_features=input_dims, out_features=2048, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(p),
            nn.Linear(in_features=2048, out_features=output_dims, bias=True))
    def forward(self, x):
        x = self.model(x)
        return x



def train(core_model,folder_dir_train,folder_dir_eval,loss_fn):
    output_folder = './save_model'
    os.makedirs(output_folder, exist_ok=True)
    # load training data
    tr_img = data_reader(folder_dir_train)
    tr = DataLoader(tr_img, batch_size=batch_size, shuffle=True, num_workers=8)
    optimizer_tag = optim.Adam(core_model.parameters(), lr=lr,weight_decay=wds)
    print('using {} as criterion'.format(loss_fn))
    # load testing data
    te_img = data_reader(folder_dir_eval)  # test
    te = DataLoader(te_img, batch_size=50, num_workers=8)  # test
    # get semantic embedding from .mat file (i.e., training labels)
    vec_mat = get_vec_mat()
    pth = folder_dir_train
    dirs_seen = os.listdir(pth)
    pth = folder_dir_eval
    vec_m = []  # seen matrix for training set
    ids = []  # id of 25 seen classes
    for f in dirs_seen:  # data of seen class
        id = int(re.sub("\D", "", f))
        ids.append(id)
        idx = id - 1
        vec_m.append(vec_mat[idx])

    vec_m = np.array(vec_m)
    print("size of seen semantic embedding:", vec_m.shape)  # (25,500)
    vec_m = vec_m.transpose()
    # hit @ 1, 2, 5
    best_acc_te_1, best_acc_te_2, best_acc_te_5 = 0, 0, 0
    writer = SummaryWriter()
    # training and testing, select the better result to save or save every 10 epochs
    for epoch in range(epoch_num):
        core_model.train()
        loss_total = 0
        real_label = []
        pre_label_1 = []
        pre_label_2 = []  # hit 2
        pre_label_5 = []  # hit 5
        print('train begin:')
        for i, (x, y) in enumerate(tr, 1):
            vec_y, tag_y = y  # vec  tag
            x = x.to(GPU, dtype=torch.float)
            vec_y = vec_y.to(GPU, dtype=torch.float)
            core_model.zero_grad()
            #  print(type(x))
            y_pred = core_model(x)
            loss = loss_fn(y_pred, vec_y)
            loss.backward()
            optimizer_tag.step()
            ### pre whether correct? ###
            #
            real_label.extend(tag_y)  # batch_size
            #
            sz = len(tag_y)
            y_pred_cpu = y_pred.cpu().detach().numpy()
            tt = np.dot(y_pred_cpu, vec_m)  # judge by dot Multiplication
            for n in range(sz):
                e = heapq.nlargest(5, range(len(tt[n])), tt[n].take)
                ii = 0
                while ii < 5:
                    if(ids[e[ii]] == tag_y[n]):
                        break
                    ii += 1
                pre_label_1.append(ids[e[0]])
                pre_label_2.append(ids[e[0]])
                pre_label_5.append(ids[e[0]])
                if(ii <= 1):
                    pre_label_2[-1] = tag_y[n]
                    pre_label_5[-1] = tag_y[n]
                elif(ii <= 4):
                    pre_label_5[-1] = tag_y[n]

            loss_total += loss.item()
        acc_1 = macro_acc(real_label, pre_label_1)  # hit 1
        acc_2 = macro_acc(real_label, pre_label_2)  # hit 2
        acc_5 = macro_acc(real_label, pre_label_5)  # hit 5
        print('Epoch {:4d}/{:4d} total_loss:{:06.5f} macro_acc_1: {:04.2f}% macro_acc_2: {:04.2f}% macro_acc_5: {:04.2f}%'.format(epoch,epoch_num, loss_total,acc_1*100,acc_2*100,acc_5*100))
        #print('total time: {:>5.2f}s'.format(time.time() - epoch_st))
        writer.add_scalars('macro_acc',
                           {'hit_1': acc_1,
                            'hit_2': acc_2,
                            'hit_5': acc_5},
                           epoch)
        writer.add_scalar('loss', loss_total, epoch)



        # testing Generalized ZSL
        combine_dir = "./ZSL_DATA/test_2/combine"
        acc_test_1, acc_test_2, acc_test_5, loss_total_test = DeVise_test_GZSL.dtest(te,core_model,combine_dir,GPU,loss_fn)
        #print(
        #    'test macro_prec_1: {:04.2f}% macro_acc_2: {:04.2f}% macro_acc_5: {:04.2f}%'.format(acc_test_1 * 100,
        #                                                                                          acc_test_2 * 100,
        #                                                                                         acc_test_5 * 100))
        writer.add_scalars('test macro_acc',
                           {'hit_1': acc_test_1,
                            'hit_2': acc_test_2,
                            'hit_5': acc_test_5},
                           epoch)
        writer.add_scalar('test loss', loss_total_test, epoch)
        if epoch>5 and (acc_test_1>best_acc_te_1 or acc_test_2>best_acc_te_2 or acc_test_5>best_acc_te_5):
            ans="_"
            if acc_test_1>best_acc_te_1:
                best_acc_te_1 = acc_test_1
                ans+="hit1_"
                ans+=str(acc_test_1*100)[:5]
                ans+="%"
            if acc_test_2>best_acc_te_2:
                best_acc_te_2 = acc_test_2
                ans+="hit2_"
                ans+=str(acc_test_2*100)[:5]
                ans += "%"
            if acc_test_5>best_acc_te_5:
                best_acc_te_5 = acc_test_5
                ans+="hit5_"
                ans+=str(acc_test_5*100)[:5]
                ans += "%"
            # './save_model'
            torch.save(core_model.state_dict(),os.path.join(output_folder, str(round(time.time())) + 'epoch' + str(epoch) +ans+ '.pkl'))

        # each 10 epoch save
        elif(epoch>0 and epoch % 10 ==0):
            torch.save(core_model.state_dict(), os.path.join(output_folder, str(round(time.time()))+'epoch'+str(epoch) + '.pkl'))

    writer.close()
    print("best_acc_te_1:", best_acc_te_1)
    print("best_acc_te_2:", best_acc_te_2)
    print("best_acc_te_5:", best_acc_te_5)


if __name__ == "__main__":
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    core_model = devise().to(GPU)
    folder_dir_train = "./ZSL_DATA/train_seen_2"  # path of training data
    folder_dir_eval = "./ZSL_DATA/test_2/unseen"  # path of unseen data
    loss_fn = nn.MSELoss()
    train(core_model, folder_dir_train, folder_dir_eval, loss_fn)