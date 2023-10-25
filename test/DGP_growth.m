function X = DGP_growth(Q, T, p, n)
X = zeros(T, p); 
for t = 1:T
    X(t, :) = mnrnd(n, Q(t, :)); 
end
end