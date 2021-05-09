function [n_cp, cp_final] = PELT_A(X, beta, T, n, hq)
%UNTITLED3 Summary of this function goes here
%   ghwang 
F = zeros(T+1, 1); F(0+1) = -beta; 
cp = cell(T+1, 1); cp{0+1}=[]; 
R = cell(T, 1); R{1}=0;  %%% tau\in\R_ts, ts=1,\ldots,T
for ts=1:T
    tau_all = R{ts}; n_tau = length(tau_all); 
    temp = zeros(n_tau, 1); 
    for itau=1:n_tau
        tau = tau_all(itau); 
        temp(itau) = F(tau+1) + LOSS_A(tau, ts, X, n, hq); 
    end
    [F(ts+1), itau1] = min(temp+beta); tau1 = tau_all(itau1); 
    cp{ts+1} = [cp{tau1+1}, tau1]; 
    if ts<T
        tau_select = tau_all(temp<F(ts+1)); 
        R{ts+1} = [tau_select, ts]; 
    end
end
cp_final = cp{T+1}; 
cp_final = cp_final(cp_final~=0); 
n_cp = length(cp_final); 
end