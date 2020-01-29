load ../data/AWA_inform_release.mat Ytr Yte attr2 tr_loc te_loc class_order
load ../data/AwA_googlenet.mat X
Ytr = Ytr(:); % train-对应的训练集标签 19*  1300 \code_server\ZSL\ZSL_DATA\train_seen
Yte = Yte(:); % test-对应的测试集标签 49* <50 = 2325 \code_server\ZSL\ZSL_DATA\test\unseen
X(isnan(X)) = 0;
X(isinf(X)) = 0;% train和test的 featrues集合 (24700+2325) * 2048
X = bsxfun(@rdivide, X, sqrt(sum(X .^ 2, 2))); % normalize
X(isnan(X)) = 0; X(isinf(X)) = 0;
Xte = X(te_loc, :);% 取出来test的  2325 * 2048 的数据
Xtr = X(tr_loc, :);% 取出来test的  24700 * 2048 的数据
clear X;
attr2(attr2==-1)=0; % attr2-对应w2v矩阵

disp('load test ok')