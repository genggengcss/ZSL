import struct
import numpy as np
import scipy.io as scio

filename = '/Users/geng/Desktop/ImageNet/2011_unseen_feature/2032.bin'
savename = '/Users/geng/Desktop/ImageNet/2011_unseen_feature/2032.mat'



# Solution #1: struct unpack
fp = open(filename, 'rb')

nx = struct.unpack("i", fp.read(4))[0]
print(nx)
ny = struct.unpack("i", fp.read(4))[0]
print(ny)
# nx = 80
# ny = 2048
test = np.empty((nx, ny), dtype=np.double)
for i in range(nx):
    for j in range(ny):
        data = fp.read(8)
        elem = struct.unpack("d", data)[0]
        test[i][j] = elem
scio.savemat(savename, {'features': test})


# Solution #2: numpy
with open(filename, 'rb') as fp:
    nx = struct.unpack("i", fp.read(4))[0]
    ny = struct.unpack("i", fp.read(4))[0]

with open(filename, 'r') as f:
    data = np.fromfile(f, dtype=np.double)[1:]
    data = np.reshape(data, (nx, ny), order='C')
    print(data[0:3, 0:3])