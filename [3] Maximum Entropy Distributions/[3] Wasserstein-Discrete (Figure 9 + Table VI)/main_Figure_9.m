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
Ball Constructed: Wasserstein distance.

Maximum  : Entropy(p)
Constrait: W(p,q)  <= theta
           sum(p)  == 1
           p >= 0
%}

clear all;
close all;
clc;

rng(45);            % I forgot which random seed generates the results in my paper

% Parameters for projected gradient descent
alpha = 1.05;
numerical_eps = 1e-4;

% Refrence distribution q
N = 4;
xi = [
    abs(sin(rand(1, 4)))
    abs(cos(exp(rand(1,4))))
];
q = rand(N, 1);
q = q./sum(q);
uniform_dist = ones(N, 1)*1/N;
theta = KL(q, uniform_dist)*1.5;

M = 6;
xj = rand(2,M);
p = zeros(M, 1);

% Joint distribution
P = zeros(N, M);

% See Theorem 5
v0 = 2;
lambda = zeros(N,1);

% Show reference points x_i
hold on;
scatter(xi(1,:),xi(2,:),100,'ro','filled');
for i = 1:N
    text(xi(1,i)+0.02,xi(2,i)+0.02,num2str(i),'fontsize',16);
end

% Show real points x_j
hold on;
scatter(xj(1,:),xj(2,:),100,'gs','filled');
for j = 1:M
    text(xj(1,j)+0.02,xj(2,j)+0.02,num2str(j),'fontsize',16);
end

% Figure format
box on;
axis([-0.05 1.05 -0.05 1.05]);
set(gca, 'FontSize', 16);
ylabel('$y_2$','FontSize', 18, 'Interpreter', 'latex');
xlabel('$y_1$','FontSize', 18, 'Interpreter', 'latex');

S = 5000000;
while true
    S = S - 1;
    
    %% Ask current sum of the joint distribution P
    P_sum = sum(sum(P));
    
    display(['Current Iter: ' num2str(S) ';   Current Summation of Joint P: ' num2str(P_sum)]);
    
    %% Exit iteration
    if S <= 0 || abs(P_sum - 1) < numerical_eps
        display(['Current Sum_P: ' num2str(P_sum)]);
        
        P = P/P_sum;    % "Early stop" does not guarantee p_sum to be unit
        
        p = sum(P, 1);
        
        display(['Entropy p: ' num2str(Entropy(p))]);
        display(['Entropy q: ' num2str(Entropy(q))]);
        
        if S <= 0    % "Early stop strategy" applied
            warning('error in solving');
        end
        
        break;
    end
    
    %% Evaluate joint probability
    for i = 1:N
        for j = 1:M
        	P(i,j) = exp(-v0*norm(xi(:,i) - xj(:,j),2) - lambda(i) - 1);
        end
    end

    %% Gradient with respect to v0 and lambda, respectively
    Gv0 = 0;
    for i = 1:N
        for j = 1:M
        	Gv0 = Gv0 + norm(xi(:,i) - xj(:,j),2) * exp(-v0*norm(xi(:,i) - xj(:,j),2) - lambda(i) - 1);
        end
    end
    Gv0 = theta - Gv0;
    Glambda = q - sum(P,2);
    
    %% Gradient descent
    v0 = v0 - alpha * Gv0;
    if v0 < 0      % Project lambda0 to [0, inf]
        v0 = 0;
    end
    
    for i = 1:N
        lambda(i) = lambda(i) - alpha * Glambda(i);
    end
end

disp('Note: This function is not stable. The initial values of q, xi, xj, and theta influence a lot on the final results.');


%% Plot
figure;
bar(q, 'FaceColor', 'flat');
legend('q');
axis([0 N+1 0 1.1*max(q)]);
set(gca, 'FontSize', 16);
ylabel('Distributions','FontSize', 18, 'Interpreter', 'latex');
xlabel('$i$','FontSize', 18, 'Interpreter', 'latex');

figure;
bar(p, 'FaceColor', 'flat');
legend('p');
axis([0 M+1 0 1.1*max(p)]);
set(gca, 'FontSize', 16);
ylabel('Distributions','FontSize', 18, 'Interpreter', 'latex');
xlabel('$i$','FontSize', 18, 'Interpreter', 'latex');
