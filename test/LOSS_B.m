function C = LOSS_B(t1, t2, X, n)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
Xbar = sum(X((t1+1):t2, :), 1)/sum(n((t1+1):t2));
C = 0;
for t=(t1+1):t2
    Xt = X(t, :)-n(t)*Xbar;
    C = C + Xt*Xt'/n(t);
end
end