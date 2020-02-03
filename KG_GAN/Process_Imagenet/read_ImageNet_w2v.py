import numpy as np
import h5py


file_name = '/Users/geng/Downloads/ImageNet_w2v/ImageNet_w2v.mat'
matcontent = h5py.File(file_name, 'r')
print(matcontent.keys())



## read w2v
w2v = np.array(matcontent['w2v']).T   # (21842, 500)
print(w2v[0])

## read wnids
wnids = np.array(matcontent['wnids']).squeeze()
# read the first one
print(wnids[0])
wnid_0 = ''.join([chr(v[0]) for v in matcontent[(wnids[0])]])
print(wnid_0)

# read all
# wnid_file = '/Users/geng/Downloads/ImageNet_w2v/wnids.txt'  # save path
# wr_fp = open(wnid_file, 'w')
# wnid_list = list()
# for i in range(len(wnids)):
#     wnid = ''.join([chr(v[0]) for v in matcontent[(wnids[i])]])
#     # print(wnid)
#     wnid_list.append(wnid)
#     wr_fp.write('%s\n' % wnid)
# print(len(wnid_list))
# wr_fp.close()

## read no_w2v_loc
no_w2v_loc = np.array(matcontent['no_w2v_loc']).squeeze()
no_w2v_list = [int(i) for i in no_w2v_loc]
print("no w2v length:", len(no_w2v_list))
print(no_w2v_list)

# read words
# cannot read from .mat file, directly save from known resource (words.txt)







