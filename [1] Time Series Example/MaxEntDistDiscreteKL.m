%{
---------------------------------------------------------------------------------------------------------
Online supplementary materials of the paper titled
"Distributionally Robust State Estimation for Nonlinear Systems".

Author:     Shixiong Wang 
Email:      s.wang@u.nus.edu; wsx.gugo@gmail.com
Affiliate:  Department of Industrial Systems Engineering and Management, National University of Singapore
Date:       4 July 2022
Website:    https://github.com/Spratm-Asleaf/DRSE-Nonlinear

---------------------------------------------------------------------------------------------------------
Warrant:    Free use, in the original or modified form, for any academic and teaching purpose.
            But please cite this paper if you are planing to use codes here for your publications.

---------------------------------------------------------------------------------------------------------
Disclaimer: Codes here are mainly for your reference, not for your picky criticisms.
            Please refrain from finding bones in an egg.
            However, friendly comments/suggestions are always appreciated and welcomed.
            Whenever you have problems in reproducing my claimed results, do contact me.
---------------------------------------------------------------------------------------------------------
%}

function p = MaxEntDistDiscreteKL(q)
    %{
    To find maximum entropy distribution p given reference distribution q.
    Ball: using KL divergence.

    Maximum  : Entropy(p)
    Constrait: KL(p,q) <= t
               sum(p)  == 1
               p >= 0
    
    See Theoem 6
    %}

    lambda0 = 2;
    lambda1 = 1;

    % Parameters for projected gradient descent
    stride = 0.05;
    numerical_eps = 1e-4;

    % Reference distribution q
    n = length(q);
    uniform_dist = ones(n, 1)*1/n;
    q = q + 1e-6;
    q = q/sum(q);
    theta = KL(q, uniform_dist)*0.05;

    % Maximum Entropy Distribution
    p = 0 * q;

    iter = 500;
    while true
        iter = iter - 1;

        %% Ask current summation
        p_sum = sum(p);

        %% Exit iteration
        if iter <= 0 || abs(p_sum - 1) < numerical_eps
            p = p/p_sum;    % "Early stop" does not guarantee p_sum to be unit
            break;
        end

        %% Evaluate probability
        for i = 1:n
            p(i) = exp((-lambda0*log(q(i)) + lambda1)/(-(lambda0 + 1))-1);
        end

        %% Gradient with respect to lambda0 and lambda1, respectively
        F0 = 0;
        F1 = 0;
        for i = 1:n
            F0 = F0 + p(i) * (1 + (log(q(i)) + lambda1)/(lambda0 + 1));
            F1 = F1 - p(i);
        end
        F0 = theta + F0;
        F1 = 1 + F1;

        %% Gradient descent
        lambda0 = lambda0 - stride * F0;
        if lambda0 < 0      % Project lambda0 to [0, inf]
            lambda0 = 0;
        end

        lambda1 = lambda1 - stride * F1;
    end

end

function [res] = KL(p,q)
% KL divergence for discrete distributions
    n = length(q);
    res = 0;
    for i = 1:n
        if p(i) < 1e-6 || q(i) < 1e-6
            t = 0;
        else
            t = p(i)*log(p(i)/q(i));
        end
        res = res + sum(t);
    end
end

