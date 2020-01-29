function SynC_fast(task, dataset, opt, direct_test)

%% Input
% task: 'train', 'val', 'test'
% dataset: 'AWA', 'CUB', 'SUN'
% opt: opt.lambda: the regularizer coefficient on W in training (e.g, 2 .^ (-24 : -9))
%      opt.Sim_scale: the RBF scale parameter for computing semantic similarities (e.g., 2 .^ (-5 : 5))
%      opt.ind_split: AWA: []; CUB: choose one from 1:4; SUN: choose one from 1:10
%      opt.loss_type: 'OVO', 'CS', 'struct'
% direct_test: test on a specific [lambda, Sim_scale] pair without cross-validation

%% Settings
set_path;
norm_method = 'L2'; Sim_type = 'RBF_norm';

%% training Data
if(strcmp(dataset, 'AWA'))
    %load ../data/AWA_inform_release.mat Ytr Yte attr2 tr_loc te_loc class_order
    %load ../data/AwA_googlenet.mat X
    train_path='../data/train_seen'
    fileFolder=fullfile(train_path);
    dirOutput=dir(fullfile(fileFolder,'*.mat'));
    fileNames={dirOutput.name};
    n = length(fileNames);
    Xtr=[];
    Ytr=[];
    class_order=[];
    for i = 1:n %25
        class_order=[class_order,i];
        pth=fullfile(train_path,fileNames{i});
        load(pth);
        Xtr=[Xtr;double(features)];%  training data 25*   1300 * 2048 
        nn=size(features);
        nn=nn(1);
        idx=str2num(fileNames{i}(isstrprop(fileNames{i},'digit')));
        for j=1:nn
            Ytr=[Ytr,idx];% train-label 25*  1300 
        end
    end
        
    disp('load train ok')   
 %% testing Data
    test_path='../data/test/seen' % seen/unseen/combine
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
        file=sort(file);
    end
    for i = 1:n % 55
        %disp(file(i))
        tmp=[num2str(file(i)),'.mat'];
        pth=fullfile(test_path,tmp);
        load(pth);
        nn=size(features);
        nn=nn(1); %dim 1
        Xte=[Xte;double(features(1:nn,:))];% get test data 55* n * 2048 
        
        for j=1:nn
            Yte=[Yte,file(i)];% train-label 55*  n
        end
    end
    Yte=Yte';
    Ytr=Ytr';
    
    %Ytr = Ytr(:); % train-对应的训练集标签 19*  1300 \code_server\ZSL\ZSL_DATA\train_seen
    %Yte = Yte(:); % test-对应的测试集标签 49* <50 = 2325 \code_server\ZSL\ZSL_DATA\test\unseen
    %X(isnan(X)) = 0;
    %X(isinf(X)) = 0;% train和test的 featrues集合 (24700+2325) * 2048
    % X = bsxfun(@rdivide, X, sqrt(sum(X .^ 2, 2))); % normalize
    %X(isnan(X)) = 0; X(isinf(X)) = 0;
    %Xte = X(te_loc, :);% 取出来test的  2325 * 2048 的数据
    %Xtr = X(tr_loc, :);% 取出来test的  24700 * 2048 的数据
    %clear X;
    
    %% g2v Data
    fileFolder=fullfile('../data/g2v-2.mat');%g2v or w2v
    load(fileFolder);
    attr2= n2v; 
    
    disp('load test ok')
    clear features w2v wnids n nn fileNames pth dirOutput i j idx fileFolder 
    
else
    display('Wrong dataset!');
    return;

end

Sig_Y = get_class_signatures(attr2, norm_method);
Sig_dist = Sig_dist_comp(Sig_Y);

save(['Sig.mat'],'Sig_Y')
%load Sig.mat  Sig_Y 
%load Sig.mat  Sig_Y

%% 5-fold class-wise cross validation splitting (for 'train' and 'val')
if (strcmp(task, 'train') || strcmp(task, 'val'))
    nr_fold = 5;
    labelSet = unique(Ytr);
    labelSetSize = length(labelSet);
    fold_size = floor(labelSetSize / nr_fold);
    fold_loc = cell(1, nr_fold);

    for i = 1 : nr_fold %5
        for j = 1 : fold_size  %3
            fold_loc{i} = [fold_loc{i}; find(Ytr == labelSet(class_order((i - 1) * fold_size + j)))];
        end
        fold_loc{i} = sort(fold_loc{i});
    end
end

%% training
if (strcmp(task, 'train'))
    for i = 1 : length(opt.lambda)
        W_record = cell(1, nr_fold);
        for j = 1 : nr_fold
            Xbase = Xtr;
            Xbase(fold_loc{j}, :) = [];
            Ybase = Ytr;
            Ybase(fold_loc{j}) = [];
            
            if (strcmp(opt.loss_type, 'OVO'))
                W = train_W_OVO([], Xbase, Ybase, opt.lambda(i));
            else
                display('Wrong loss type!');
                return;
            end
            W_record{j} = W;

            save(['../SynC_CV_classifiers/SynC_fast_' opt.loss_type '_classCV_' dataset '_split' num2str(opt.ind_split)  norm_method '_' Sim_type...
                '_lambda' num2str(opt.lambda(i)) '.mat'], 'W_record');
        end
    end
end

%% validation
if (strcmp(task, 'val'))
    acc_val = zeros(length(opt.lambda), length(opt.Sim_scale));
    for i = 1 : length(opt.lambda)
        load(['../SynC_CV_classifiers/SynC_fast_' opt.loss_type '_classCV_' dataset '_split' num2str(opt.ind_split) norm_method '_' Sim_type...
            '_lambda' num2str(opt.lambda(i)) '.mat'], 'W_record');
        for j = 1 : nr_fold
            Ybase = Ytr;
            Ybase(fold_loc{j}) = [];
            Xval = Xtr(fold_loc{j}, :);
            Yval = Ytr(fold_loc{j});
            W = W_record{j};

            for k = 1 : length(opt.Sim_scale)
                Sim_base = Compute_Sim(Sig_Y(unique(Ybase), :), Sig_Y(unique(Ybase), :), opt.Sim_scale(k), Sim_type);
                Sim_val = Compute_Sim(Sig_Y(unique(Yval), :), Sig_Y(unique(Ybase), :), opt.Sim_scale(k), Sim_type);
                V = pinv(Sim_base) * W;            
                Ypred_val = test_V(V, Sim_val, Xval, Yval,1);
                acc_val(i, k) = acc_val(i, k) + evaluate_easy(Ypred_val, Yval) / nr_fold;          
            end
            clear W;
        end
        clear W_record;
    end
    save(['../SynC_CV_results/SynC_fast_' opt.loss_type '_classCV_' dataset '_split' num2str(opt.ind_split) norm_method '_' Sim_type '.mat'], 'acc_val', 'opt');
end

%% testing
if (strcmp(task, 'test'))
    if(isempty(direct_test))
        load(['../SynC_CV_results/SynC_fast_' opt.loss_type '_classCV_' dataset '_split' num2str(opt.ind_split) norm_method '_' Sim_type '.mat'], 'acc_val', 'opt');
        [loc_lambda, loc_Sim_scale] = find(acc_val == max(acc_val(:)));
        lambda = opt.lambda(loc_lambda(1)); Sim_scale = opt.Sim_scale(loc_Sim_scale(1));
    else % using determined parameters
        lambda = direct_test(1); Sim_scale = direct_test(2);
    end
    
    if (strcmp(opt.loss_type, 'OVO'))
        W = train_W_OVO([], Xtr, Ytr, lambda);
        %save(['w.mat'],'W');  %25 2048
        %load w.mat W
    elseif (strcmp(opt.loss_type, 'CS'))
        W = train_W_CS([], Xtr, Ytr, lambda);
    elseif (strcmp(opt.loss_type, 'struct'))
        W = train_W_struct([], Xtr, Ytr, Sig_dist(unique(Ytr), unique(Ytr)), lambda);
    else
        display('Wrong loss type!');
        return;
    end
    
    Sim_tr = Compute_Sim(Sig_Y(unique(Ytr), :), Sig_Y(unique(Ytr), :), Sim_scale, Sim_type); %Ytr和  %19*19 similarity calculation
    %GZSL:
    Sim_te = Compute_Sim(Sig_Y(unique(Yte), :), Sig_Y(unique(Ytr), :), Sim_scale, Sim_type);
    %Sim_te = Compute_Sim(Sig_Y(unique(Yte), :), Sig_Y(unique(Ytr), :), Sim_scale, Sim_type);
    V = pinv(Sim_tr) * W;
    Ypred_te_1 = test_V(V, Sim_te, Xte, Yte,1);  %hit 1
    Ypred_te_2 = test_V(V, Sim_te, Xte, Yte,2);  %hit 2
    Ypred_te_5 = test_V(V, Sim_te, Xte, Yte,5);  %hit 5
    
    acc_1=evaluate_easy(Ypred_te_1,Yte); %ZSL
    acc_2=evaluate_easy(Ypred_te_2,Yte); %ZSL
    acc_5=evaluate_easy(Ypred_te_5,Yte); %ZSL
    %% seen_y generate
    %seen_1_y=Yte(1:150);
    %unseen_1_y=Yte(151:1450);
    %seen_2_y=Yte(1451:1700);
    %unseen_2_y=Yte(1701:3000);
    %seen_3_y=Yte(3001:3150);
    %unseen_3_y=Yte(3151:4450);
    %seen_4_y=Yte(4451:5150);
    %unseen_4_y=Yte(5151:end);
%     seen_y=Yte(1:5000);
%     unseen_y=Yte(5001:end);
    
    
    %seen_y=[seen_1_y;seen_2_y;seen_3_y;seen_4_y];
    %unseen_y=[unseen_1_y;unseen_2_y;unseen_3_y;unseen_4_y];
    %Ypred_te = test_V(V, Sim_te, Xte, Yte, hit);  %该函数决定hit 几
    %display(size(Ypred_te));
    
    %seen_p=Ypred_te(1:1250);
    %seen_y=Yte(1:1250);
    %unseen_p=Ypred_te(1251:end);
    %unseen_y=Yte(1251:end);
    
    %acc_seen = evaluate_easy(seen_p, seen_y);
    %acc_unseen=evaluate_easy(unseen_p,unseen_y);
    %% pre generate hit1
%     seen_p_1=Ypred_te_1(1:5000);
%     unseen_p_1=Ypred_te_1(5001:end);
    
    
    %seen_1_p=Ypred_te_1(1:150);
    %unseen_1_p=Ypred_te_1(151:1450);
    %seen_2_p=Ypred_te_1(1451:1700);
    %unseen_2_p=Ypred_te_1(1701:3000);
    %seen_3_p=Ypred_te_1(3001:3150);
    %unseen_3_p=Ypred_te_1(3151:4450);
    %seen_4_p=Ypred_te_1(4451:5150);
    %unseen_4_p=Ypred_te_1(5151:end);

    %seen_p_1=[seen_1_p;seen_2_p;seen_3_p;seen_4_p];
    %unseen_p_1=[unseen_1_p;unseen_2_p;unseen_3_p;unseen_4_p];
    %% pre generate hit2
%     seen_p_2=Ypred_te_2(1:5000);
%     unseen_p_2=Ypred_te_2(5001:end);


    %seen_1_p=Ypred_te_2(1:150);
    %unseen_1_p=Ypred_te_2(151:1450);
    %seen_2_p=Ypred_te_2(1451:1700);
    %unseen_2_p=Ypred_te_2(1701:3000);
    %seen_3_p=Ypred_te_2(3001:3150);
    %unseen_3_p=Ypred_te_2(3151:4450);
    %seen_4_p=Ypred_te_2(4451:5150);
    %unseen_4_p=Ypred_te_2(5151:end);

    %seen_p_2=[seen_1_p;seen_2_p;seen_3_p;seen_4_p];
    %unseen_p_2=[unseen_1_p;unseen_2_p;unseen_3_p;unseen_4_p];
	%% pre generate hit5
%     seen_p_5=Ypred_te_5(1:5000); % 10 classes *50 nums
%     unseen_p_5=Ypred_te_5(5001:end);


    %seen_1_p=Ypred_te_5(1:150);
    %unseen_1_p=Ypred_te_5(151:1450);
    %seen_2_p=Ypred_te_5(1451:1700);
    %unseen_2_p=Ypred_te_5(1701:3000);
    %seen_3_p=Ypred_te_5(3001:3150);
    %unseen_3_p=Ypred_te_5(3151:4450);
    %seen_4_p=Ypred_te_5(4451:5150);
    %unseen_4_p=Ypred_te_5(5151:end);

    %seen_p_5=[seen_1_p;seen_2_p;seen_3_p;seen_4_p];
    %unseen_p_5=[unseen_1_p;unseen_2_p;unseen_3_p;unseen_4_p];
    
%     acc_seen_1=evaluate_easy(seen_p_1, seen_y); %GZSL
%     acc_seen_2=evaluate_easy(seen_p_2, seen_y); %GZSL
%     acc_seen_5=evaluate_easy(seen_p_5, seen_y); %GZSL
%     acc_unseen_1=evaluate_easy(unseen_p_1,unseen_y); %GZSL
%     acc_unseen_2=evaluate_easy(unseen_p_2,unseen_y); %GZSL
%     acc_unseen_5=evaluate_easy(unseen_p_5,unseen_y); %GZSL
    
    %save(['../SynC_results/hit' num2str(hit) 'GZSL.mat'], 'lambda', 'Sim_scale', 'acc_seen','acc_unseen');
    %save(['../SynC_results/hit1-2-5'  'GZSL-new-w2v.mat'], 'lambda', 'Sim_scale', 'acc_seen_1','acc_seen_2','acc_seen_5', 'acc_unseen_1','acc_unseen_2','acc_unseen_5');
    save(['../SynC_results/hit1-2-5-'  'ZSL-new-g2v-seen.mat'], 'lambda', 'Sim_scale','acc_1','acc_2','acc_5');
end

end

function Sig_dist = Sig_dist_comp(Sig_Y)
inner_product = Sig_Y * Sig_Y';
C = size(Sig_Y, 1);
Sig_dist = max(diag(inner_product) * ones(1, C) + ones(C, 1) * diag(inner_product)' - 2 * inner_product, 0);
Sig_dist = sqrt(Sig_dist);
end