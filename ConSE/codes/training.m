function training(C)

%% Input
% C: a vector of the C values to train logistic regression
C=[2.0];



%% Data loading and preprocessing
addpath('../liblinear/matlab'); % #### Need to fix the path ####

%ÐèÒª£º
%Xtr:19*1300*2048¡¢
%Ytr:19*1300*1
train_path='../data/train_seen';%  train set path
fileFolder=fullfile(train_path); 
dirOutput=dir(fullfile(fileFolder,'*.mat'));
fileNames={dirOutput.name};
n = length(fileNames);
Xtr=[];
Ytr=[];
for i = 1:n %25
    pth=fullfile(train_path,fileNames{i});
    load(pth);
    Xtr=[Xtr;double(features)];% get training data 25*   1300 * 2048 
    nn=size(features);
    nn=nn(1);
    idx=str2num(fileNames{i}(isstrprop(fileNames{i},'digit')));
    for j=1:nn
        Ytr=[Ytr;idx];% train-label 19*  1300 
    end
end
clear features n nn fileNames pth dirOutput i j idx fileFolder fileNames

%% Training
for c = 1 : length(C)

    %% training
    svm_model = train(double(Ytr), sparse(Xtr), ['-s 0 -c ' num2str(C(c)) ' -e 0.001 -q']);
    save(['../results/conse_model_new_C' num2str(C(c)) '.mat'], 'C', 'svm_model');
end
end