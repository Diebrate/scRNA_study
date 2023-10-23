clear; clc;
Iter = 50;
%%% load data
data = readtable('../../data/simulation_data/simulation_id0.csv');
T = max(data.time);
d = max(data.type) + 1;
n0 = sum(data.time == 0);
n = n0 * ones(T+1, 1);
%%%
c0_A = 2.0; 
c0_B = 1.2;
Lh_A = zeros(Iter); th_A = cell(Iter); 
Lh_B_temp = zeros(Iter, T-1); th_B_temp = cell(Iter, T-1);  %%% at most T-1 changepoints
Lh_B = zeros(Iter); th_B = cell(Iter); 
Lh = zeros(Iter); th = cell(Iter); 
tic; 
for iter=1:Iter
    X = zeros(T+1, d);
    for t=0:T
        X(t, :) = sortrows(grpstats(data(data.time == t,:), {'type'}), 'type').GroupCount;
    end
    %%% selective: hat_A, added on Dec. 4, 2016
    hq = sum(X, 1)/sum(n); 
    [hq_sort, hq_sort_index] = sort(hq, 2, 'descend'); 
    coss = (-1-hq_sort(1:(d-1)).*hq_sort(2:d))./sqrt((1+hq_sort(1:(d-1)).^2).*(1+hq_sort(2:d).^2)); 
    [~, coss_max_index] = max(coss); 
    hat_A = hq_sort_index(1:coss_max_index); 
    hat_B = hq_sort_index((coss_max_index+1):d); 
    %%% est:A
    xi = c0_A*(log(T+1))^1.5; 
    beta = xi; 
    [Lh_A(iter), th_A{iter}] = PELT_A(X(:, hat_A), beta, T+1, n, hq(hat_A)); 
    %%% est:B
    th_A_all = [0, th_A{iter}, T+1]; 
    % tuning
    X_B = X(:, hat_B); 
    qqh_vec = sum(X_B.^2-X_B, 2)/(n0*(n0-1)); 
    qqh = mean(qqh_vec); 
    for a=1:(Lh_A(in0, iter)+1)
        X_new = X((th_A_all(a)+1):th_A_all(a+1), hat_B); 
        T_new = size(X_new, 1); 
        n_new = n((th_A_all(a)+1):th_A_all(a+1)); 
        eta = c0_B*(log(T))^(1.1)*sqrt(qqh); 
        beta = sum(sum(X_new, 2))/sum(n_new)+eta; 
        [Lh_B_temp(in0, iter, a), th_B_temp{in0, iter, a}] = PELT_B(X_new, beta, T_new, n_new); 
        th_B_temp{in0, iter, a} = th_B_temp{in0, iter, a} + th_A_all(a); 
    end
    Lh_B(in0, iter) = sum(Lh_B_temp(in0, iter, 1:(Lh_A(in0, iter)+1))); 
    th_B{in0, iter} = [th_B_temp{in0, iter, :}]; 
    Lh(in0, iter) = Lh_A(in0, iter) + Lh_B(in0, iter); 
    th{in0, iter} = [th_A{in0, iter}, th_B{in0, iter}]; 
end
cpt = cell(n_n0, Iter);
for i=1:n_n0
    for j=1:Iter
        cpt{i, j} = th{i, j} - 1;
    end
end
toc;