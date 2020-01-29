function attr2_pred  = predict_ConSE(attr2, T, prob_pred)

%% Input:
% attr2: the m-dim semantic embeddings (SE) of Ns "Seen" classes (Ns-by-m) 
% T: the top T Seen classes to consider in predicting the SEs of the test samples
% prob_pred: the predicted probabilistic membership of the Nu test samples into the Ns Seen classes (Nu-by-Ns)

%% Output:
% attr2_pred: the predicted SEs of the test samples (Nu-by-m)

%% Main codes
Ns = size(prob_pred, 1);
[~, loc] = sort(prob_pred, 2, 'descend');
loc = loc(:, T + 1 : end);
if(~isempty(loc))
    ignored_loc = bsxfun(@plus, (loc - 1) * Ns, (1 : Ns)');
    ignored_loc = ignored_loc(:);
    prob_pred(ignored_loc) = 0;
    prob_pred = bsxfun(@rdivide, prob_pred, sum(abs(prob_pred), 2));
    prob_pred(isnan(prob_pred)) = 0; prob_pred(isinf(prob_pred)) = 0;
end
attr2_pred = prob_pred * attr2;
end