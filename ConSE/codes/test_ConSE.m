function Ypred = test_ConSE(attr2_te, attr2_pred, labelSet,Y,hit)

%% Input:
% attr2_te: the m-dim semantic embeddings (SE) of Nu "Unseen" classes (Nu-by-m) 
% attr2_pred: the predicted SEs of the test samples (Nu-by-m)
% labelSet: a Nu-by-1 vector recording the unseen class labels

%% Output:
% Ypred: the predicted unseen class labels of the test samples

attr2_te = bsxfun(@rdivide, attr2_te, sqrt(sum(attr2_te .^ 2, 2)));
attr2_pred = bsxfun(@rdivide, attr2_pred, sqrt(sum(attr2_pred .^ 2, 2)));

cos_sim = attr2_pred * attr2_te';
cos_sim(isnan(cos_sim)) = 0; 
cos_sim(isinf(cos_sim)) = 0;

%cos_sim---equal to---XW
sz=size(cos_sim);
Ypred=[];
for i=1:sz(1) %for test data
    t=sort(cos_sim(i,:),'descend'); %get sort
    ok=t(1:hit); %get num of top n hit
    [m,n]=find(cos_sim(i,:)==ok(1));
    Ypred=[Ypred labelSet(n(1))];
    for j=1:hit
        [m,n]=find(cos_sim(i,:)==ok(j));
        if labelSet(n(1))==Y(i)
          Ypred(end)=Y(i);
          break
        end
    end
end
Ypred=Ypred';
%[~, loc] = max(cos_sim, [], 2);
%Ypred = labelSet(loc);
end