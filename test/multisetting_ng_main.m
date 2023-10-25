clc
load ../../results/simulation/multisetting_data_ng.mat
n_ns = length(ns); n_pwrs = length(pwrs);
T = double(T); p = double(p);
c0_A = 2.0; 
c0_B = 1.2; 
Iter = double(M); 
Lh_A = zeros(n_ns, n_pwrs, Iter); th_A = cell(n_ns, n_pwrs, Iter); 
Lh_B_temp = zeros(n_ns, n_pwrs, Iter, T-1); th_B_temp = cell(n_ns, n_pwrs, Iter, T-1);  %%% at most T-1 changepoints
Lh_B = zeros(n_ns, n_pwrs, Iter); th_B = cell(n_ns, n_pwrs, Iter); 
Lh = zeros(n_ns, n_pwrs, Iter); th = cell(n_ns, n_pwrs, Iter); 
tic; 
for ind_n=1:n_ns
    n = double(ns(ind_n)); n_rep = n*ones(T, 1);
    for ind_pwr=1:n_pwrs
        pwr = double(pwrs(ind_pwr));
        disp(['n=', num2str(n), ', ', 'pwr=', num2str(pwr)])
        for iter=1:Iter
            X = double(data{ind_n, ind_pwr}(:, :, iter));
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
            [Lh_A(ind_n, ind_pwr, iter), th_A{ind_n, ind_pwr, iter}] = PELT_A(X(:, hat_A), beta, T, n_rep, hq(hat_A)); 
            %%% est:B
            th_A_all = [0, th_A{ind_n, ind_pwr, iter}, T]; 
            % tuning
            X_B = X(:, hat_B); 
            qqh_vec = sum(X_B.^2-X_B, 2)/(n*(n-1)); 
            qqh = mean(qqh_vec); 
            for a=1:(Lh_A(ind_n, ind_pwr, iter)+1)
                X_new = X((th_A_all(a)+1):th_A_all(a+1), hat_B); 
                T_new = size(X_new, 1); 
                n_new = n_rep((th_A_all(a)+1):th_A_all(a+1)); 
                eta = c0_B*(log(T))^(1.1)*sqrt(qqh); 
                beta = sum(sum(X_new, 2))/sum(n_new)+eta; 
                [Lh_B_temp(ind_n, ind_pwr, iter, a), th_B_temp{ind_n, ind_pwr, iter, a}] = PELT_B(X_new, beta, T_new, n_new); 
                th_B_temp{ind_n, ind_pwr, iter, a} = th_B_temp{ind_n, ind_pwr, iter, a} + th_A_all(a); 
            end
            Lh_B(ind_n, ind_pwr, iter) = sum(Lh_B_temp(ind_n, ind_pwr, iter, 1:(Lh_A(ind_n, ind_pwr, iter)+1))); 
            th_B{ind_n, ind_pwr, iter} = [th_B_temp{ind_n, ind_pwr, iter, :}]; 
            Lh(ind_n, ind_pwr, iter) = Lh_A(ind_n, ind_pwr, iter) + Lh_B(ind_n, ind_pwr, iter); 
            th{ind_n, ind_pwr, iter} = [th_A{ind_n, ind_pwr, iter}, th_B{ind_n, ind_pwr, iter}];
        end
    end
end
cpt = cell(n_ns, n_pwrs, Iter);
for i=1:n_ns
    for j=1:n_pwrs
        for k=1:Iter
            cpt{i, j, k} = th{i, j, k} - 1;
        end
    end
end
toc; 
save multisetting_res_ng_switch.mat