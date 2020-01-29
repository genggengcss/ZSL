from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy.io as sio
import os
import re
from collections import defaultdict

def get_vec_mat():
    vec_pth = "./ZSL_DATA/ImageNet/g2v.mat"
    vec_mat = sio.loadmat(vec_pth)
    vec_mat = vec_mat['n2v']
    vec_mat=vec_mat.astype(np.float64)
    return vec_mat


class data_reader(Dataset):
    def __init__(self, folder_dir, train=True):
        ### Use C.E data ###
        #
        vec_mat=get_vec_mat()
        self.pth=os.path.join(folder_dir)
        dirs = os.listdir(self.pth)
        self.x=[]
        self.y_tag = [] #tag
        self.y_vec=[] #vec
        for f in dirs:
            tmp_dir=os.path.join(self.pth,f)
            mat = sio.loadmat(tmp_dir)
            features=mat['features'].astype(np.float64)
            #get id
            id=int(re.sub("\D","",f))
        #id tag build
            for i in range(features.shape[0]): # 50 or //features.shape[0]
                self.y_tag.append(id)
            # vec build
            idx=id-1 # must -1
            for i in range(features.shape[0]):
                self.y_vec.append(vec_mat[idx])
            #features build
            if len(self.x) == 0:
                self.x = features
            else:
                self.x = np.concatenate((self.x, features), axis=0)


        self.x=self.x.astype(np.float64)
        self.y_vec = np.array(self.y_vec).astype(np.float64)
        print("features data size: ",self.x.shape) # (24700, 2048)  2450
        print("tag data len: ", len(self.y_tag)) # (24700)  2450

        set1 = set(self.y_tag)
        print("length of tag:",len(set1))
        value_cnt = defaultdict(int)
        for value in self.y_tag:
            # get(value, num)
            value_cnt[value] = value_cnt[value] + 1
        for key in value_cnt.keys():
            print(key, 'num of class:', value_cnt[key])
        print("vec data size: ", self.y_vec.shape)  # (24700,500)  2450


        #self.word2idx = word2idx
        #self.word_vec_dict = word_vec_dict

    def __len__(self):
        return (self.x.shape[0])

    def __getitem__(self, idx):
        tmp_x = self.x[idx]
        tmp_y_tag = self.y_tag[idx]
        tmp_y_vec = self.y_vec[idx]

        return (tmp_x, (tmp_y_vec, tmp_y_tag)) #vec  tag

if __name__ == "__main__":
    folder_dir="../ZSL_DATA/test/unseen"
    word2idx=[]
    word_vec_dict=[]
    batch_size=20
    tr_img = data_reader(folder_dir,train=False)
    tr = DataLoader(tr_img, batch_size=batch_size, shuffle=True, num_workers=8)
    x,y=tr_img[1400]
    (y1,y2)=y
    print(y2)