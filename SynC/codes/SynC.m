function SynC(task, dataset, opt, direct_test)

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

%% Data
if(strcmp(dataset, 'AWA'))
    load ../data/AWA_inform_release.mat Ytr Yte attr2 tr_loc te_loc class_order
    load ../data/AwA_googlenet.mat X
    Ytr = Ytr(:); Yte = Yte(:);
    X(isnan(X)) = 0; X(isinf(X)) = 0;
    X = bsxfun(@rdivide, X, sqrt(sum(X .^ 2, 2)));
    X(isnan(X)) = 0; X(isinf(X)) = 0;
    Xtr = X(tr_loc, :); Xte = X(te_loc, :); clear X;
    attr2(attr2 == -1) = 0;
    
elseif(strcmp(dataset, 'CUB'))
    load ../data/CUB_inform_release.mat Y attr2 CUB_class_loc class_order
    load ../data/CUB_googlenet.mat X
    Y = Y(:);
    X(isnan(X)) = 0; X(isinf(X)) = 0;
    X = bsxfun(@rdivide, X, sqrt(sum(X .^ 2, 2)));
    X(isnan(X)) = 0; X(isinf(X)) = 0;
    Xtr = X; Xte = X(CUB_class_loc{opt.ind_split}, :); Xtr(CUB_class_loc{opt.ind_split}, :) = []; clear X;
    Ytr = Y; Yte = Y(CUB_class_loc{opt.ind_split}); Ytr(CUB_class_loc{opt.ind_split}) = []; clear Y;

elseif(strcmp(dataset, 'SUN'))
    load ../data/SUN_inform_release.mat attr2 SUN_class_loc class_order
    load ../data/SUN_googlenet.mat X Y
    Y = Y(:);
    X(isnan(X)) = 0; X(isinf(X)) = 0;
    X = bsxfun(@rdivide, X, sqrt(sum(X .^ 2, 2)));
    X(isnan(X)) = 0; X(isinf(X)) = 0;
    Xtr = X; Xte = X(SUN_class_loc{opt.ind_split}, :); Xtr(SUN_class_loc{opt.ind_split}, :) = []; clear X;
    Ytr = Y; Yte = Y(SUN_class_loc{opt.ind_split}); Ytr(SUN_class_loc{opt.ind_split}) = []; clear Y;
    class_order = class_order{opt.ind_split};
    
else
    display('Wrong dataset!');
    return;

end

Sig_Y = get_class_signatures(attr2, norm_method);
Sig_dist = Sig_dist_comp(Sig_Y);

%% 5-fold class-wise cross validation splitting (for 'train' and 'val')
if (strcmp(task, 'train') || strcmp(task, 'val'))
    nr_fold = 5;
    labelSet = unique(Ytr);
    labelSetSize = length(labelSet);
    fold_size = floor(labelSetSize / nr_fold);
    fold_loc = cell(1, nr_fold);

    for i = 1 : nr_fold
        for j = 1 : fold_size
            fold_loc{i} = [fold_loc{i}; find(Ytr == labelSet(class_order((i - 1) * fold_size + j)))];
        end
        fold_loc{i} = sort(fold_loc{i});
    end
end

%% training
if (strcmp(task, 'train'))
    for i = 1 : length(opt.lambda)
        V_record = cell(nr_fold, length(opt.Sim_scale));
        for j = 1 : nr_fold
            Xbase = Xtr;
            Xbase(fold_loc{j}, :) = [];
            Ybase = Ytr;
            Ybase(fold_loc{j}) = [];
            
            for k = 1 : length(opt.Sim_scale)
                Sim_base = Compute_Sim(Sig_Y(unique(Ybase), :), Sig_Y(unique(Ybase), :), opt.Sim_scale(k), Sim_type);
                if (strcmp(opt.loss_type, 'OVO'))
                    V = train_V_OVO([], Sim_base, Xbase, Ybase, opt.lambda(i));
                elseif (strcmp(opt.loss_type, 'CS'))
                    V = train_V_CS([], Sim_base, Xbase, Ybase, opt.lambda(i));
                elseif (strcmp(opt.loss_type, 'struct'))
                    V = train_V_struct([], Sim_base, Xbase, Ybase, Sig_dist(unique(Ybase), unique(Ybase)), opt.lambda(i));
                else
                    display('Wrong loss type!');
                    return;
                end
                V_record{j, k} = V;
            end

            save(['../SynC_CV_classifiers/SynC_' opt.loss_type '_classCV_' dataset '_split' num2str(opt.ind_split) '_googleNet_' norm_method '_' Sim_type...
                '_lambda' num2str(opt.lambda(i)) '.mat'], 'V_record');
        end
    end
end

%% validation
if (strcmp(task, 'val'))
    acc_val = zeros(length(opt.lambda), length(opt.Sim_scale));
    for i = 1 : length(opt.lambda)
        load(['../SynC_CV_classifiers/SynC_' opt.loss_type '_classCV_' dataset '_split' num2str(opt.ind_split) '_googleNet_' norm_method '_' Sim_type...
            '_lambda' num2str(opt.lambda(i)) '.mat'], 'V_record');
        for j = 1 : nr_fold
            Ybase = Ytr;
            Ybase(fold_loc{j}) = [];
            Xval = Xtr(fold_loc{j}, :);
            Yval = Ytr(fold_loc{j});

            for k = 1 : length(opt.Sim_scale)
                V = V_record{j, k};
                Sim_val = Compute_Sim(Sig_Y(unique(Yval), :), Sig_Y(unique(Ybase), :), opt.Sim_scale(k), Sim_type);          
                Ypred_val = test_V(V, Sim_val, Xval, Yval);
                acc_val(i, k) = acc_val(i, k) + evaluate_easy(Ypred_val, Yval) / nr_fold;
                clear V;
            end
        end
        clear V_record;
    end
    save(['../SynC_CV_results/SynC_' opt.loss_type '_classCV_' dataset '_split' num2str(opt.ind_split) '_googleNet_' norm_method '_' Sim_type '.mat'], 'acc_val', 'opt');
end

%% testing
if (strcmp(task, 'test'))
    if(isempty(direct_test))
        load(['../SynC_CV_results/SynC_' opt.loss_type '_classCV_' dataset '_split' num2str(opt.ind_split) '_googleNet_' norm_method '_' Sim_type '.mat'], 'acc_val', 'opt');
        [loc_lambda, loc_Sim_scale] = find(acc_val == max(acc_val(:)));
        lambda = opt.lambda(loc_lambda(1)); Sim_scale = opt.Sim_scale(loc_Sim_scale(1));
    else
        lambda = direct_test(1); Sim_scale = direct_test(2);
    end
    
    Sim_tr = Compute_Sim(Sig_Y(unique(Ytr), :), Sig_Y(unique(Ytr), :), Sim_scale, Sim_type);
    if (strcmp(opt.loss_type, 'OVO'))
        V = train_V_OVO([], Sim_tr, Xtr, Ytr, lambda);
    elseif (strcmp(opt.loss_type, 'CS'))
        V = train_V_CS([], Sim_tr, Xtr, Ytr, lambda);
    elseif (strcmp(opt.loss_type, 'struct'))
        V = train_V_struct([], Sim_tr, Xtr, Ytr, Sig_dist(unique(Ytr), unique(Ytr)), lambda);
    else
        display('Wrong loss type!');
        return;
    end
    
    Sim_te = Compute_Sim(Sig_Y(unique(Yte), :), Sig_Y(unique(Ytr), :), Sim_scale, Sim_type);
    Ypred_te = test_V(V, Sim_te, Xte, Yte);
    acc_te = evaluate_easy(Ypred_te, Yte);

    save(['../SynC_results/SynC_' opt.loss_type '_' dataset '_split' num2str(opt.ind_split) '_googleNet_' norm_method '_' Sim_type...
        '_lambda' num2str(lambda) '_Sim_scale' num2str(Sim_scale) '.mat'], 'V', 'lambda', 'Sim_scale', 'acc_te');
end

end

function Sig_dist = Sig_dist_comp(Sig_Y)
inner_product = Sig_Y * Sig_Y';
C = size(Sig_Y, 1);
Sig_dist = max(diag(inner_product) * ones(1, C) + ones(C, 1) * diag(inner_product)' - 2 * inner_product, 0);
Sig_dist = sqrt(Sig_dist);
end