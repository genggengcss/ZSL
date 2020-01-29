load ../data/AWA_inform_release.mat Ytr Yte attr2 tr_loc te_loc class_order
load ../data/AwA_googlenet.mat X
Ytr = Ytr(:); % train-��Ӧ��ѵ������ǩ 19*  1300 \code_server\ZSL\ZSL_DATA\train_seen
Yte = Yte(:); % test-��Ӧ�Ĳ��Լ���ǩ 49* <50 = 2325 \code_server\ZSL\ZSL_DATA\test\unseen
X(isnan(X)) = 0;
X(isinf(X)) = 0;% train��test�� featrues���� (24700+2325) * 2048
X = bsxfun(@rdivide, X, sqrt(sum(X .^ 2, 2))); % normalize
X(isnan(X)) = 0; X(isinf(X)) = 0;
Xte = X(te_loc, :);% ȡ����test��  2325 * 2048 ������
Xtr = X(tr_loc, :);% ȡ����test��  24700 * 2048 ������
clear X;
attr2(attr2==-1)=0; % attr2-��Ӧw2v����

disp('load test ok')