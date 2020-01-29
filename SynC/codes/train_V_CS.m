function [V, flag] = train_V_CS(V0, Sim, X, Y, lambda)

%% interface description:
% Input:
%   V0: a C-by-d matrix, the initialization of V
%   Sim: a C-by-R matrix, the similarity between seen and phantom classes
%   X: a N-by-d matrix, with each row a training instance
%   Y: a N-by-1 vctor, with each element the class label (1,...,C) of the corresponding instance
%   lambda: balance coefficient

% Output:
%   V: a R-by-d matrix, with each row the 1-vs-all classifier of a basis class. 

%% pre-settig:
[N, d] = size(X);
[C, R] = size(Sim);
labels = unique(Y);
ind = zeros(N, C);
y_loc = zeros(N, 1);
for i = 1 : C
    ind(Y == labels(i), i) = 1;
    y_loc(Y == labels(i), 1) = i;
end
if (length(labels) ~= C)
    display('Error: train_V');
    return;
end

%% Parameter initialization
options.Display = 'off';
options.Method = 'lbfgs';
options.optTol = 1e-10;
options.progTol = 1e-10;
options.MaxIter = 2000;
options.MaxFunEvals = 2400;

%% Begin training
if isempty(V0)
    V0 = randn(R, d) / 100;
end
funObj = @(arg)compute_fg_V(arg, Sim, X, ind, y_loc, lambda);
[V, ~, flag] = minFunc(funObj, V0(:), options);
V = reshape(real(V), [R, d]);

end


function [f, g] = compute_fg_V(V0, Sim, X, ind, y_loc, lambda)

[N, d] = size(X);
[~, R] = size(Sim);
V = reshape(real(V0), [R, d]);
W = construct_W(V, Sim);
XW = X * W';
y_index = N * (y_loc - 1) + (1 : N)';
XW_star = XW(y_index);
diff_XW = bsxfun(@minus, XW + 1 - ind, XW_star);
[val, loc] = max(diff_XW, [], 2);
max_index = N * (loc - 1) + (1 : N)';
diff_ind = -ind;
diff_ind(max_index) = diff_ind(max_index) + 1;
L = max(0, val);
f = sum(L) / N + lambda / 2 * sum(sum(W .^ 2));
g = (Sim' * diff_ind' * X) / N + lambda * Sim' * W;
g = real(g(:));

end