# ImageNet



'''
Data Loading
'''

DATADIR = '/home/gyx/ZSL/data'  # path to dataset
Workers = 2  # number of data loading workers
# for ImageNet
DATASET = 'ImageNet'
SeenFeaFile = 'Res101_Features/ILSVRC2012_Res101_Features_train'
UnseenFeaFile = 'Res101_Features/ILSVRC2011_Res101_Features'
SemFile = 'w2v.mat'  # the file to store class embedding
SplitFile = 'split.mat'
SemEmbed = 'w2v'
FeaSize = 2048  # size of visual features
SemSize = 500  # size of semantic features
NoiseSize = 500  # dim of noise vector, equal to SemSize
NClassAll = 610  # number of all classes
Exp_Name = 'Exp2'
NSample = 500  # constrain the number of samples <= 500
PerClassAcc = True
# for AWA, CUB, etc
# DATASET = 'AWA1'
# FeaFile = 'res101.mat'
# SemFile = 'att_splits.mat'  # the file to store class embedding
# FeaSize = 2048  # size of visual features
# SemSize = 103  # size of semantic features
# NoiseSize = 103  # dim of noise vector, equal to SemSize
# NClassAll = 200  # number of all classes
'''
Generator and Discriminator
'''
NetG_Path = ''  # path to netG (to continue training)
NetD_Path = ''  # path to netD (to continue training)
Pretrained_Classifier = ''  # path to pretrain classifier (to continue training)
NetG_Name = 'MLP_G'
NetD_Name = 'MLP_CRITIC'
NGH = 4096  # size of the hidden units in generator
NDH = 4096  # size of the hidden units in discriminator
Critic_Iter = 5  # critic iteration of discriminator, default=5, following WGAN-GP setting
GP_Weight = 10.0  # gradient penalty regularizer, default=10, the completion of Lipschitz Constraint in WGAN-GP
Cls_Weight = 0.01  # loss weight for the supervised classification loss

Proto_Weight1 = 0.01  # para for soul samples's regularization2
Proto_Weight2 = 0.01  # para for soul samples's regularization1
NClusters = 3  # number of real clusters
Cluster_Save_File = 'save_cluster/real_proto.pickle'
NSynClusters = 20  # number of fake clusters?
SynNum = 300  # number features to generate per class; awa_default = 300
'''
Training Parameter
'''
GZSL = False  # enable generalized zero-shot learning
PreProcess = True  # enbale MinMaxScaler on visual features, default=True
Standardization = False
Cross_Validation = False  # enable cross validation mode
Cuda = True  # enable cuda
NGPU = 1  # number of GPUs to use
CUDA_DEVISE = "2,3"
# ManualSeed = None  # manual seed, type:int; awa_default = 9182
ManualSeed = 9416
BatchSize = 2048  # input batch size
Epoch = 50  # number of epochs to train
LR = 0.0001  # learning rate to train GAN
Cls_LR = 0.001  # after generating unseen features, the learning rate for training softmax classifier
Ratio = 0.1  # ratio of easy samples
Beta = 0.5  # beta for adam, default=0.5

OutFolder = './checkpoint/'  # folder to output data and model checkpoints
OutName = 'awa'  # folder to output data and model checkpoints
SaveEvery = 100
PrintEvery = 1
ValEvery = 1
StartEvery = 0



