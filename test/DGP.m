function X = DGP(kappa, Q, T, p, L, n)
X = zeros(T, p); 
ts = [0, kappa, 1]*T; 
for l=0:L
    ql = Q(l+1, :); 
    ts_a = ts(l+1); ts_b = ts(l+1+1); 
    for t=(ts_a+1):ts_b
        X(t, :) = mnrnd(n(t), ql); 
    end
end
end