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
To find continuous maximum entropy distribution p given discrete reference distribution q.
Ball Constructed: Wasserstein distance.

Maximum  : Entropy(p)
Constrait: W(p,q)  <= theta
           int(p)  == 1
           p >= 0
%}

clear all;
close all;
clc;

% Cardinality of the support set of q
N = 6;

% See Theorem 3
v0 = 2;
v1 = 0;
lambda = zeros(N,1);

% Monte-Carlor Integration samples
int_N = 2000;                   % "int" for "integration"
small_area = 1/int_N;           % the total area of the square is "1*1 = 1"
points = rand(2,int_N);         % uniformly generate samples from "[0,1] * [0,1]"

% Monte-Carlor integrals
p_int = 0;                      % integral of p(x)
part_int = zeros(N, 1);         % integral of p(x) in the sub-region C_i

% Parameters for projected gradient descent
alpha = 0.05;
numerical_eps = 1e-3;

% Generate samples for q
% q are weights
%q = rand(N, 1); q = q./sum(q);
q = [0.0583    0.2695    0.0340    0.3496    0.1453    0.1433]';
uniform_dist = ones(N, 1)*1/N;
theta = KL(q, uniform_dist)*0.1;    % theta = 0.025
% x^i are supporting points
xi = [%rand(2,N);
    0.5007    0.2397    0.7338    0.7065    0.3739    0.4450
    0.8763    0.1513    0.0323    0.6066    0.1581    0.4139
];
x_range = [0 1];
y_range = [0 1];

% Gradients
Gv0 = 0;
Gv1 = 0;
Glambda = zeros(N, 1);

% Parameters for projected gradient descent iterations
S = 500;
logdata = [];                   % record tentative data
last_p_int = 0;                 % integral of p(x) in the last iteration step
while true
    S = S - 1;
    
    %% Calculate current integrals
    % Classification of Monte-Carlor Integration samples
    class  = zeros(1,int_N);
    for i = 1:int_N
        d = zeros(1,N);
        for j = 1:N
            d(j) = norm(points(:,i) - xi(:,j),2) - lambda(j);
        end
        [min_value, index] = min(d);
        class(i) = index;                   % the "sample i" belongs to the class "index"
    end
    
    % Monte-Carlor Integral of p(x)
    p = @(x,y) exp(-v0.*min_g(x,y,xi,lambda) - v1 - 1);
    p_int = 0;
    for i = 1:int_N
        p_int = p_int + p(points(1,i),points(2,i));
    end
    p_int = p_int * small_area;
    
    % Monte-Carlor-Integrals of p in C_i
    part_int = zeros(N, 1);
    for i = 1:int_N
        class_of_sample_i = class(i);
        part_p = @(x,y) exp(-v0*g(x,y,xi(:,class_of_sample_i),lambda(class_of_sample_i)) - v1 - 1);
        part_int(class_of_sample_i) = part_int(class_of_sample_i) + part_p(points(1,i),points(2,i));
    end
    part_int = part_int * small_area;
    
    display(['Current Iter: ' num2str(S) ';   Current Integral of p(x): ' num2str(p_int)]);
    
    %% Exit iteration
    if S <= 0 || abs(p_int - 1) < numerical_eps
        if S <= 0    % "Early stop strategy" applied
            warning('main :: early stopping applied.');
        end
        break;
    end

    %% Gradient with respect to v0, v1, and lambda_i, respectively
    gp = @(x,y) min_g(x,y,xi,lambda) .* exp(-v0.*min_g(x,y,xi,lambda) - v1 - 1);
    gp_int = 0;
    for i = 1:int_N
        gp_int = gp_int + gp(points(1,i),points(2,i));
    end
    gp_int = gp_int * small_area;
    
    % with respect to v0
    Gv0 = theta - lambda'*q  - gp_int;
    % with respect to v1
    Gv1 = 1 - p_int;
    % with respect to lambda_i
    for i = 1:N
        Glambda(i) = v0*(-q(i) + part_int(i));
    end
    
    %% Gradient descent
    v0 = v0 - alpha * Gv0;
    if v0 < 0      % Project v0 to [0, inf]
        v0 = 0;
    end
    
    v1 = v1 - alpha * Gv1;
    
    for i = 1:N
        lambda(i) = lambda(i) - alpha * Glambda(i);
    end

    %% Save data
    logdata = [logdata;
        S p_int v0 v1 lambda' part_int' Gv0 Gv1 Glambda'
    ];

    last_p_int = p_int;
end

%% Plot partition
% Monte Carlo samples for plot
disp('Plotting...');
plot_N = 100000;
plot_points = rand(2,plot_N);
plot_class  = zeros(1,plot_N);
for i = 1:plot_N
    d = zeros(1,N);
    for j = 1:N
        d(j) = norm(plot_points(:,i) - xi(:,j),2) - lambda(j);
    end
    [min_value, index] = min(d);
    plot_class(i) = index;
end

% Show samples for plot 
hold on;
color = [
102 255 255 
255 255 102
102 255 102
255 180 255
255 178 102
204 160 255
]/255;
for i = 1:N
    index = find(plot_class==i);
    scatter(plot_points(1,index),plot_points(2,index),[],color(i,:),'marker','.');
end

% Show reference points x_i
hold on;
scatter(xi(1,:),xi(2,:),100,'ro','filled');
for i = 1:N
    text(xi(1,i)+0.02,xi(2,i)+0.02,num2str(i),'fontsize',16);
end

% Plot format
box on;
set(gca, 'FontSize', 16);
ylabel('$x_2$','FontSize', 18, 'Interpreter', 'latex');
xlabel('$x_1$','FontSize', 18, 'Interpreter', 'latex');

%% Plot p(x)
figure;
fmesh(p,[0 1 0 1]);
set(gca, 'FontSize', 16);
ylabel('$x_2$','FontSize', 18, 'Interpreter', 'latex');
xlabel('$x_1$','FontSize', 18, 'Interpreter', 'latex');
zlabel('$p(\bf \it x)$','FontSize', 18, 'Interpreter', 'latex');
