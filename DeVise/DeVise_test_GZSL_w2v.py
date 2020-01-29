import numpy as np
import random
import scipy, argparse, os, sys, csv, io, time
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score
import re
from torch.utils.tensorboard import SummaryWriter
import heapq
from src.data_reader_w2v import data_reader, get_vec_mat
from src.count_tools import macro_acc
import DeVise_run_w2v



def load_test_data(folder_dir):
    te_img = data_reader(folder_dir, train=False)  # test
    te = DataLoader(te_img, batch_size=50, num_workers=8)  # test
    return te

def dtest(te,model,combine_dir,devce,loss_fn=nn.MSELoss()):

    # get vec mat
    vec_mat = get_vec_mat()
    pth = combine_dir
    dirs_general = os.listdir(pth)
    vec_m_combine = []  # unseen 训练集的矩阵
    ids_combine = []
    for f in dirs_general:  # 对于seen和unseen类
        id = int(re.sub("\D", "", f))
        ids_combine.append(id)
        idx = id - 1
        vec_m_combine.append(vec_mat[idx])


    vec_m_combine = np.array(vec_m_combine)
    #print("size of vec test mat :", vec_m_combine.shape)  # 49+19=68,500
    print('test begin:')
    vec_m_combine = vec_m_combine.transpose()


    with torch.no_grad():
        model.eval()
        real_label_test = []
        pre_label_test_1 = []
        pre_label_test_2 = []  # hit 2
        pre_label_test_5 = []  # hit 5
        loss_total_test = 0
        for (vx, vy) in te:
            val_vec_y, val_tag_y = vy
            val_vec_y = val_vec_y.to(devce, dtype=torch.float)
            vx = vx.to(devce, dtype=torch.float)
            vy_pred = model(vx)
            vloss = loss_fn(vy_pred, val_vec_y)  # test set loss compute
            loss_total_test += vloss.item()
            ### pre whether correct? ###
            #
            real_label_test.extend(val_tag_y)
            #
            vy_pred_cpu = vy_pred.cpu().detach().numpy()
            vsz = len(val_tag_y)
            vtt = np.dot(vy_pred_cpu, vec_m_combine)  # judge by dot Multiplication
            for n in range(vsz):
                e = heapq.nlargest(5, range(len(vtt[n])), vtt[n].take) # top 5 hit
                vi = 0
                while vi < 5:
                    if (ids_combine[e[vi]] == val_tag_y[n]):  # pre right
                        break
                    vi += 1
                pre_label_test_1.append(ids_combine[e[0]])
                pre_label_test_2.append(ids_combine[e[0]])
                pre_label_test_5.append(ids_combine[e[0]])

                if (vi <= 1):
                    pre_label_test_2[-1] = val_tag_y[n]
                    pre_label_test_5[-1] = val_tag_y[n]
                elif (vi <= 4):
                    pre_label_test_5[-1] = val_tag_y[n]


        acc_test_1 = macro_acc(real_label_test, pre_label_test_1)
        acc_test_2 = macro_acc(real_label_test, pre_label_test_2)
        acc_test_5 = macro_acc(real_label_test, pre_label_test_5)
        print('test macro_acc_1: {:04.2f}% macro_acc_2: {:04.2f}% macro_acc_5: {:04.2f}%'.format(acc_test_1 * 100,
                                                                                                  acc_test_2 * 100,
                                                                                                  acc_test_5 * 100))
        return acc_test_1, acc_test_2, acc_test_5, loss_total_test

if __name__ == "__main__":
    pth="./save_model/w2v1579363909epoch20.pkl"
    devce = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DeVise_run_w2v.devise().to(devce)
    model.load_state_dict(torch.load(pth))
    folder_dir="./ZSL_DATA/test/unseen"#unseen/ seen
    combine_dir="./ZSL_DATA/test/unseen"#unseen/ seen/ combine
    te=load_test_data(folder_dir)
    dtest(te ,model, combine_dir,devce)