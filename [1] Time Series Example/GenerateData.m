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

global SimulationStep;
global TrueF TrueH

Q = 10; 
R = 1;     

x = 0;

X = zeros(1, SimulationStep);       % real state
Y = zeros(1, SimulationStep);       % real measurement
for k = 1:SimulationStep
    % Generate true state
    w = sqrt(Q)*randn;
    x = TrueF(x, k) + w;
    
    % Generate true measurements
    v = sqrt(R)*randn;
    y = TrueH(x, k) + v;

    % Save system state and system measurement
    X(:,k) = x;
    Y(:,k) = y;
end

%% Nominal process noise covariance
% Because we know the true value of process noise covariance.
Q = Q;                    

%% Nominal measurement noise covariance
% Because we know the true value of measurement noise covariance.
R = R;



