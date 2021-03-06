function testing(C, T)

%% Input
% C & T: both are scalars
C=2;
T=10;
%% Settings
norm_method = 'L2';

%% Data loading and preprocessing
addpath('../liblinear/matlab'); % #### Need to fix the path ####


%attr2:  w2v ...*��
%Yte:49*<50*1��
%Ytr:19*1300*1 ��
%Xtr:19*1300*2048
train_path='../data/train_seen'
fileFolder=fullfile(train_path);%  train set path
dirOutput=dir(fullfile(fileFolder,'*.mat'));
fileNames={dirOutput.name};
n = length(fileNames);
Ytr=[];
for i = 1:n %25
    pth=fullfile(train_path,fileNames{i});
    load(pth);
    nn=size(features);
    nn=nn(1);
    idx=str2num(fileNames{i}(isstrprop(fileNames{i},'digit')));
    for j=1:nn
        Ytr=[Ytr;idx];% train-label 25*  1300 
    end
end

disp('load training set ok')   
%% test data
test_path='../data/test/combine' %combine or seen or unseen
fileFolder=fullfile(test_path);
dirOutput=dir(fullfile(fileFolder,'*.mat'));
fileNames={dirOutput.name};
n = length(fileNames);
Xte=[];
Yte=[];
file=[];
for i = 1:n % 55
    idx=str2num(fileNames{i}(isstrprop(fileNames{i},'digit')));
    file=[file idx];
    file=sort(file); % use sorting to make combine data in order. most seen classes are <1000 and unseen classer are >1000
end
for i = 1:n % 55
    %disp(file(i))
    tmp=[num2str(file(i)),'.mat'];
    pth=fullfile(test_path,tmp);
    load(pth);
    nn=size(features);
    nn=nn(1); %dim 1
    Xte=[Xte;double(features(1:nn,:))];% get test data 55* all * 2048 

    for j=1:nn
        Yte=[Yte,file(i)];% train-label 25*  1300 
    end
end
Yte=Yte';

fileFolder=fullfile('../data/g2v-2.mat'); %choose g2v/w2v
load(fileFolder);
attr2= n2v; % attr2-to-n2v/w2v matrix
clear features n nn fileNames pth dirOutput i j idx fileFolder fileNames w2v wnids tmp
Sig_Y = get_class_signatures_sanity(attr2, norm_method);

%% Testing

load(['../results/conse_model_new_C' num2str(C) '.mat'], 'svm_model');
[~, ~, prob_pred] = predict(Ytr(1) * ones(size(Yte)), sparse(Xte), svm_model, '-b 1');
[~, loc] = sort(svm_model.Label);
attr2_pred  = predict_ConSE(Sig_Y(unique(Ytr), :), T, prob_pred(:, loc));
% because in order so... 
seen_y=Yte(1:5000);% 10 seen class * 50 nums
unseen_y=Yte(5001:end);

% seen_1_y=Yte(1:150);
% unseen_1_y=Yte(151:1450);
% seen_2_y=Yte(1451:1700);
% unseen_2_y=Yte(1701:3000);
% seen_3_y=Yte(3001:3150);
% unseen_3_y=Yte(3151:4450);
% seen_4_y=Yte(4451:5150);
% unseen_4_y=Yte(5151:end);
% 

% seen_y=[seen_1_y;seen_2_y;seen_3_y;seen_4_y];
% unseen_y=[unseen_1_y;unseen_2_y;unseen_3_y;unseen_4_y];
% disp(unique(seen_y));
% disp('len');
% disp(num2str(length(unique(seen_y))))


%% hit 1
Ypred_te = test_ConSE(Sig_Y(unique(Yte), :), attr2_pred, unique(Yte),Yte,1);
seen_p=Ypred_te(1:5000);
unseen_p=Ypred_te(5001:end);
% seen_1_p=Ypred_te(1:150);
% unseen_1_p=Ypred_te(151:1450);
% 
% %disp(unique(seen_1_p))
% %disp(['1len:' ,num2str(length(unique(seen_1_p)))]);
% %disp(unique(seen_1_y))
% %disp(['1len:' ,num2str(length(unique(unseen_1_y)))]);
% 
% 
% seen_2_p=Ypred_te(1451:1700);
% unseen_2_p=Ypred_te(1701:3000);
% 
% %disp(unique(seen_2_p))
% %disp(['2len:' ,num2str(length(unique(seen_2_p)))]);
% %disp(unique(seen_2_y))
% %disp(['2len:' ,num2str(length(unique(unseen_2_y)))]);
% 
% seen_3_p=Ypred_te(3001:3150);
% unseen_3_p=Ypred_te(3151:4450);
% 
% %disp(unique(seen_3_p))
% %disp(['3len:' ,num2str(length(unique(seen_3_p)))]);
% %disp(unique(seen_3_y))
% %disp(['3len:' ,num2str(length(unique(unseen_3_y)))]);
% 
% seen_4_p=Ypred_te(4451:5150);
% unseen_4_p=Ypred_te(5151:end);
% 
% %disp(unique(seen_4_p))
% %disp(['4len:' ,num2str(length(unique(seen_4_p)))]);
% %disp(unique(seen_4_y))
% %disp(['4len:' ,num2str(length(unique(unseen_4_y)))]);
% 
% seen_p=[seen_1_p;seen_2_p;seen_3_p;seen_4_p];
% unseen_p=[unseen_1_p;unseen_2_p;unseen_3_p;unseen_4_p];
% 
acc_seen_1 = evaluate_easy(seen_p, seen_y);
acc_unseen_1=evaluate_easy(unseen_p,unseen_y);
acc_te_1 = evaluate_easy(Ypred_te, Yte);

%% hit 2
Ypred_te = test_ConSE(Sig_Y(unique(Yte), :), attr2_pred, unique(Yte),Yte,2);
seen_p=Ypred_te(1:5000);
unseen_p=Ypred_te(5001:end);
% seen_1_p=Ypred_te(1:150);
% unseen_1_p=Ypred_te(151:1450);
% seen_2_p=Ypred_te(1451:1700);
% unseen_2_p=Ypred_te(1701:3000);
% seen_3_p=Ypred_te(3001:3150);
% unseen_3_p=Ypred_te(3151:4450);
% seen_4_p=Ypred_te(4451:5150);
% unseen_4_p=Ypred_te(5151:end);
% 
% seen_p=[seen_1_p;seen_2_p;seen_3_p;seen_4_p];
% unseen_p=[unseen_1_p;unseen_2_p;unseen_3_p;unseen_4_p];
% 
acc_seen_2 = evaluate_easy(seen_p, seen_y);
acc_unseen_2=evaluate_easy(unseen_p,unseen_y);
acc_te_2 = evaluate_easy(Ypred_te, Yte);

%% hit 5
Ypred_te = test_ConSE(Sig_Y(unique(Yte), :), attr2_pred, unique(Yte),Yte,5);
seen_p=Ypred_te(1:5000);
unseen_p=Ypred_te(5001:end);
% seen_1_p=Ypred_te(1:150);
% unseen_1_p=Ypred_te(151:1450);
% seen_2_p=Ypred_te(1451:1700);
% unseen_2_p=Ypred_te(1701:3000);
% seen_3_p=Ypred_te(3001:3150);
% unseen_3_p=Ypred_te(3151:4450);
% seen_4_p=Ypred_te(4451:5150);
% unseen_4_p=Ypred_te(5151:end);
% 
% seen_p=[seen_1_p;seen_2_p;seen_3_p;seen_4_p];
% unseen_p=[unseen_1_p;unseen_2_p;unseen_3_p;unseen_4_p];
% 
% 
acc_seen_5 = evaluate_easy(seen_p, seen_y);
acc_unseen_5=evaluate_easy(unseen_p,unseen_y);
acc_te_5 = evaluate_easy(Ypred_te, Yte);

%% save the answer
save(['../results/hit1-2-5'  '_' 'ZSL_new.mat'], 'acc_te_1','acc_te_2','acc_te_5'); %this is ZSL 
%this is GZSL 
save(['../results/hit1-2-5'  '_' 'GZSL_new_g2v.mat'], 'acc_seen_1','acc_unseen_1', 'acc_seen_2','acc_unseen_2', 'acc_seen_5','acc_unseen_5');
end