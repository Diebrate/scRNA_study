clc
offsets = [true, false];
for offset = offsets
    if offset
        load ..\..\results\simulation\multisetting_data_cor.mat
    else
        load ..\..\results\simulation\multisetting_data.mat
    end
    n_ns = length(ns); n_nus = length(nus); n_zetas = length(zetas);
    T = double(T); p = double(p);
    c0_A = 2.0; 
    c0_B = 1.2; 
    Iter = double(M); 
    Lh_A = zeros(n_ns, n_nus, n_zetas, Iter); th_A = cell(n_ns, n_nus, n_zetas, Iter); 
    Lh_B_temp = zeros(n_ns, n_nus, n_zetas, Iter, T-1); th_B_temp = cell(n_ns, n_nus, n_zetas, Iter, T-1);  %%% at most T-1 changepoints
    Lh_B = zeros(n_ns, n_nus, n_zetas, Iter); th_B = cell(n_ns, n_nus, n_zetas, Iter); 
    Lh = zeros(n_ns, n_nus, n_zetas, Iter); th = cell(n_ns, n_nus, n_zetas, Iter); 
    tic; 
    for ind_n=1:n_ns
        n = double(ns(ind_n)); n_rep = n*ones(T, 1);
        for ind_nu=1:n_nus
            nu = double(nus(ind_nu));
            for ind_zeta=1:n_zetas
                zeta = double(zetas(ind_zeta));
                disp(['n=', num2str(n), ', ', 'nu=', num2str(nu), ', ', 'eta=', num2str(zeta)])
                for iter=1:Iter
                    X = double(data{ind_n, ind_nu, ind_zeta}(:, :, iter));
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
                    [Lh_A(ind_n, ind_nu, ind_zeta, iter), th_A{ind_n, ind_nu, ind_zeta, iter}] = PELT_A(X(:, hat_A), beta, T, n_rep, hq(hat_A)); 
                    %%% est:B
                    th_A_all = [0, th_A{ind_n, ind_nu, ind_zeta, iter}, T]; 
                    % tuning
                    X_B = X(:, hat_B); 
                    qqh_vec = sum(X_B.^2-X_B, 2)/(n*(n-1)); 
                    qqh = mean(qqh_vec); 
                    for a=1:(Lh_A(ind_n, ind_nu, ind_zeta, iter)+1)
                        X_new = X((th_A_all(a)+1):th_A_all(a+1), hat_B); 
                        T_new = size(X_new, 1); 
                        n_new = n_rep((th_A_all(a)+1):th_A_all(a+1)); 
                        eta = c0_B*(log(T))^(1.1)*sqrt(qqh); 
                        beta = sum(sum(X_new, 2))/sum(n_new)+eta; 
                        [Lh_B_temp(ind_n, ind_nu, ind_zeta, iter, a), th_B_temp{ind_n, ind_nu, ind_zeta, iter, a}] = PELT_B(X_new, beta, T_new, n_new); 
                        th_B_temp{ind_n, ind_nu, ind_zeta, iter, a} = th_B_temp{ind_n, ind_nu, ind_zeta, iter, a} + th_A_all(a); 
                    end
                    Lh_B(ind_n, ind_nu, ind_zeta, iter) = sum(Lh_B_temp(ind_n, ind_nu, ind_zeta, iter, 1:(Lh_A(ind_n, ind_nu, ind_zeta, iter)+1))); 
                    th_B{ind_n, ind_nu, ind_zeta, iter} = [th_B_temp{ind_n, ind_nu, ind_zeta, iter, :}]; 
                    Lh(ind_n, ind_nu, ind_zeta, iter) = Lh_A(ind_n, ind_nu, ind_zeta, iter) + Lh_B(ind_n, ind_nu, ind_zeta, iter); 
                    th{ind_n, ind_nu, ind_zeta, iter} = [th_A{ind_n, ind_nu, ind_zeta, iter}, th_B{ind_n, ind_nu, ind_zeta, iter}];
                end
            end
        end
    end
    toc;
    cpt = cell(n_ns, n_nus, n_zetas, Iter);
    for i=1:n_ns
        for j=1:n_nus
            for k=1:n_zetas
                for ind_iter=1:Iter
                    cpt{i, j, k, ind_iter} = th{i, j, k, ind_iter} - 1;
                end
            end
        end
    end
    if offset
        save ..\..\results\simulation\multisetting_res_mn_cor.mat
    else
        save ..\..\results\simulation\multisetting_res_mn.mat
    end
end