function ret = min_g(x,y,xi,lambda)
% "min_g" may simutaneously work for many (x, y) pairs.
% Suppose the size of matrices x and y is [m, n].
% "min_g" can simutaneously work for m*n pairs of (x,y).
% Each (x, y) pair is defined by [x(i,j); y(i,j)], where i = 1:1:m and j = 1:1:n.

    [m,n] = size(x);
    [~,b] = size(xi);
    ret = zeros(m,n);
    for i = 1:m
        for j = 1:n
            temp_xy = [
                x(i,j)
                y(i,j)
            ];
            d = zeros(b,1);
            for k = 1:b
                d(k) = norm(temp_xy - xi(:,k),2);
            end
            ret(i,j) = min(d - lambda);
        end
    end
end

