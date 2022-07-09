function ret = g(x,y,xi,lambda_i)
% "g" may simutaneously work for many (x, y) pairs.
% Suppose the size of matrices x and y is [m, n].
% "g" can simutaneously work for m*n pairs of (x,y).
% Each (x, y) pair is defined by [x(i,j); y(i,j)], where i = 1:1:m and j = 1:1:n.

    [m,n] = size(x);
    ret = zeros(m,n);
    for i = 1:m
        for j = 1:n
            temp_xy = [x(i,j); y(i,j)];
            d = norm(temp_xy - xi,2);
            ret(i,j) = d - lambda_i;
        end
    end
end

