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
            If you cannot reproduce my claimed results, do contact me.
=========================================================================================================
=========================================================================================================
=========================================================================================================
Function:
To find discrete maximum entropy distribution p given discrete reference distribution q.
Ball Constructed: KL divergence.

Maximum  : Entropy(p)
Constrait: KL(p,q) <= theta
           sum(p)  == 1
           p >= 0
%}

clear all;
close all;
clc;

% See Theoem 6
lambda0 = 2;
lambda1 = 0;

% Parameters for projected gradient descent
alpha = 0.05;
numerical_eps = 1e-8;

% Reference distribution q
% q = rand(6, 1); q = q./sum(q);
q = [0.1993 0.2907 0.0974 0.0492 0.1505 0.2128]';
n = length(q);
uniform_dist = ones(n, 1)*1/n;
theta = KL(q, uniform_dist)*0.1;

% Maximum Entropy Distribution
p = 0 * q;          % p and q must have the same length

S = 50000;
while true
    S = S - 1;
    
    %% Ask current summation
    p_sum = sum(p);
    
    %% Exit iteration
    if S <= 0 || abs(p_sum - 1) < numerical_eps
        display(['Current Sum_p: ' num2str(p_sum)]);
        
        p = p/p_sum;    % "Early stopping" does not guarantee p_sum to be unit
        
        display(['Entropy p: ' num2str(Entropy(p))]);
        display(['Entropy q: ' num2str(Entropy(q))]);
        display(['KL(p,q): ' num2str(KL(p,q))]);
        display(['theta: ' num2str(theta)]);
        display(['Extreme difference of p and q, respectively:    ' num2str(max([p,q]) - min([p,q]))]);
        
        if S <= 0    % "Early stop strategy" applied
            warning('main :: early stopping applied.');
        end
        
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
    lambda0 = lambda0 - alpha * F0;
    if lambda0 < 0      % Project lambda0 to [0, inf]
        lambda0 = 0;
    end
    
    lambda1 = lambda1 - alpha * F1;
end

%% Plot
handle = bar([p,q], 'FaceColor', 'flat');
legend('$p$','$q$', 'Interpreter', 'latex');
axis([0 n+1 0 1.1*max([p;q])]);
handle(1).CData = [0 1 0];
handle(2).CData = [1 0 0];
set(gca, 'FontSize', 16);
ylabel('Distributions','FontSize', 18, 'Interpreter', 'latex');
xlabel('$i$','FontSize', 18, 'Interpreter', 'latex');

