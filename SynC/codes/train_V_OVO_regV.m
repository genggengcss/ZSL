function [V, flag] = train_V_OVO_regV(V0, Sim, X, Y, lambda)

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
ind = -ones(N, C);
for i = 1 : C
    ind(Y == labels(i), i) = 1;
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
funObj = @(arg)compute_fg_V(arg, Sim, X, ind, lambda);
[V, ~, flag] = minFunc(funObj, V0(:), options);
V = reshape(real(V), [R, d]);

end


function [f, g] = compute_fg_V(V0, Sim, X, ind, lambda)

[N, d] = size(X);
[C, R] = size(Sim);
V = reshape(real(V0), [R, d]);
W = construct_W(V, Sim);
XW = X * W';
L = max(0, 1 - ind .* XW) .^ 2;
SV = double(L > eps);
f = sum(sum(L)) / C / N + lambda / 2 * sum(sum(V .^ 2));
g = 2 * (Sim' * ((XW - ind) .* SV)' * X) / C / N + lambda * V;
g = real(g(:));

end