clear; clc; 
%%% change-point model
T = 20; 
p = 10; 
kappa = [0.25, 0.5, 0.75]; L = length(kappa);
pwr = 1.25;
nu = 0.1;
eta = 1.75;
delta = 0;
growth = true;
rng(10086);
if ~growth
    Q = zeros(L+1, p);
    for l=0:L
        Q(l+1, :) = ones([1, p]);
        if rem(l, 2) == 0
            Q(l+1, 1:0.5*p) = pwr;
        else
            Q(l+1, 0.5*p+1:p) = pwr;
        end
        Q(l+1, :) = Q(l+1, :)/sum(Q(l+1, :));
    end
else
    %B1 = ones(p, p);
    %B1(:, 0.5*p:p) = eta;
    B1 = rand(p);
    B1 = diag(1 ./ sum(B1, 2)) * B1;
    %B2 = ones(p, p);
    %B2(:, 1:0.5*p) = eta;
    B2 = rand(p);
    B2 = diag(1 ./ sum(B2, 2)) * B2;
    Q = zeros(T, p);
    Q(1, :) = 1/p;
    pop_type = 0;
    sep = floor(p/2);
    for l=2:T
        if rem(pop_type, 2) == 0
            g_rate = ones(1, p);
            g_rate(1:sep) = exp(randn(1, sep)*nu + delta);
            g_rate(sep+1:p) = exp(randn(1, p-sep)*nu);
            B = B1;
        else
            g_rate = ones(1, p);
            g_rate(1:sep) = exp(randn(1, sep)*nu);
            g_rate(sep+1:p) = exp(randn(1, p-sep)*nu + delta);
            B = B2;
        end
        Q(l, :) = Q(l-1, :) .* g_rate;
        Q(l, :) = Q(l, :)/sum(Q(l, :));
        if any((l-1)/T == kappa)
            pop_type = pop_type + 1;
            Q(l, :) = B' * Q(l, :)';
        end
    end
end
sum(Q, 2)
diag(Q*Q')
Delta = zeros(L, 1); 
for l=1:L
    d = Q(l+1, :) - Q(l, :); 
    Delta(l) = d*d'; 
end
disp(Delta)
%%% 
n0_all = [250, 500, 750, 1000]; n_n0 = length(n0_all); 
c0_A = 2.0; 
c0_B = 1.2; 
Iter = 1000; 
Lh_A = zeros(n_n0, Iter); th_A = cell(n_n0, Iter); 
Lh_B_temp = zeros(n_n0, Iter, T-1); th_B_temp = cell(n_n0, Iter, T-1);  %%% at most T-1 changepoints
Lh_B = zeros(n_n0, Iter); th_B = cell(n_n0, Iter); 
Lh = zeros(n_n0, Iter); th = cell(n_n0, Iter); 
tic; 
% load ..\..\results\simulation\multisetting_data.mat
% T = double(T); p = double(p);
for in0=1:n_n0
    n0 = n0_all(in0); n = n0*ones(T, 1); 
    disp(['n0=', num2str(n0)])
    for iter=1:Iter
        if growth
            X = DGP(kappa, Q, T, p, L, n);
        else
            X = DGP_growth(Q, T, p, n(1));
        end
        %X = double(data{in0, 2, 2}(:, :, iter))
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
    end
end
toc; 
save main_L3.mat