function [res] = Entropy(p)
% Entropy for discrete distributions
    res = -sum(p.*log(p));
end

