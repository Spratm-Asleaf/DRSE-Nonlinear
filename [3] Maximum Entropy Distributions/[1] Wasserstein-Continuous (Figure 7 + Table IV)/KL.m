function [res] = KL(p,q)
% KL divergence for discrete distributions
    [a, ~] = size(p);
    if a == 1
        p = p';
    end
    [b, ~] = size(q);
    if b == 1
        q = q';
    end
    res = sum(p.*log(p./q));
end

