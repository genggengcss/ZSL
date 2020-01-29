import scipy.io as sio
floder_dir='D:/pycharm/code_server/ZSL/ZSL_DATA/ImageNet/split.mat'
txt_dir='D:/pycharm/code_server/ZSL/ZSL_DATA/ImageNet/new/seen.txt'
# animal_dir=('./ZSL_DATA/ImageNet/materials/wnids-animal.txt')
mat=sio.loadmat(floder_dir)

#m1=np.transpose(mat['features'])
#print(mat['seen'])
wnids=mat['allwnids']
words=mat['allwords']
print(type(wnids))#<class 'numpy.ndarray'>
print(wnids.shape)#(1000, 1)
print(type(words))#<class 'numpy.ndarray'>
print(words.shape)#(1000, 1)
data=[]
animal=[]
for line in open(txt_dir,"r"):
    data.append(line[:-1])
# for line in open(animal_dir,"r"):
#     animal.append(line[:-1])
wnids=wnids.tolist()
sz=len(wnids)
print(type(wnids))#<class 'numpy.ndarray'>
print()
all=0
for i in range(sz):
    for j in data:
        # if j in animal:
        #     continue
        if(wnids[i][0].item()==j):
            print(i+1,j,words[i][0])
            all+=1

print('all:',all)
