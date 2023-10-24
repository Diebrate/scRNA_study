clear; clc;
tic;
m = str2double(getenv('SLURM_ARRAY_TASK_ID'));
%m = 0;
Iter = 50;

strings = {'../../data/simulation_data/simulation_id1_0.csv'};
data = readtable(strjoin(strings, ''));
%data = readtable('../../data/simulation_data/simulation_id1_0.csv');
T = max(data.time) + 1;
p = max(data.type) + 1;
n0 = sum(data.time == 0);
n = n0 * ones(T, 1);

n_n0 = 1;
in0 = 1;
c0_A = 2.0; 
c0_B = 1.2;
Lh_A = zeros(n_n0, Iter); th_A = cell(n_n0, Iter); 
Lh_B_temp = zeros(n_n0, Iter, T-1); th_B_temp = cell(n_n0, Iter, T-1);  %%% at most T-1 changepoints
Lh_B = zeros(n_n0, Iter); th_B = cell(n_n0, Iter); 
Lh = zeros(n_n0, Iter); th = cell(n_n0, Iter); 
cpt = cell(n_n0, Iter);
res = zeros(Iter, T-1);

for iter=1:Iter
    disp(['m=', num2str(m), ' b=', num2str(iter)]);
    %%% load data
    %strings = {'../../data/simulation_data/simulation_id', num2str(m), '_', num2str(iter-1), '.csv'};
    %data = readtable(strjoin(strings, ''));
    data = readtable('../../data/simulation_data/simulation_id0.csv');
    %%%
    X = zeros(T, p);
    for t=1:T
        X(t, :) = sortrows(grpstats(data(data.time == (t - 1), 'type'), 'type'), 'type').GroupCount;
    end
    %%% selective: hat_A, added on Dec. 4, 2016
    hq = sum(X, 1)/sum(n); 
    [hq_sort, hq_sort_index] = sort(hq, 2, 'descend'); 
    coss = (-1-hq_sort(1:(p-1)).*hq_sort(2:p))./sqrt((1+hq_sort(1:(p-1)).^2).*(1+hq_sort(2:p).^2)); 
    [~, coss_max_index] = max(coss); 
    hat_A = hq_sort_index(1:coss_max_index); 
    hat_B = hq_sort_index((coss_max_index+1):p); 
    %%% est:A
    xi = c0_A*(log(T))^1.5; 
    beta = xi; 
    [Lh_A(in0, iter), th_A{in0, iter}] = PELT_A(X(:, hat_A), beta, T, n, hq(hat_A)); 
    %%% est:B
    th_A_all = [0, th_A{in0, iter}, T]; 
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
    cpt{1, iter} = th{1, iter} - 1;
    for cp=cpt{1, iter}
        if cp > 0
            res(iter,cp) = 1;
        end
    end
end
filename = {'../../results/simulation/test_mn_id', num2str(m), '.mat'};
save(strjoin(filename, ''), 'res');
toc;