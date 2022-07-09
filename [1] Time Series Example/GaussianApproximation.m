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

function [X_hat] = GaussianApproximation(Y,Q,R,x0_hat,P0)
    % This function realizes the Ensemble Kalman Filter (EnKF).
    % cf. the function file "UKF.m". (N.B., the different is how many prior state particles is used and how to sample them.)
    % In UKF, only "2*n" samples (called sigma points) are used.
    % But in EnKF, samples are directly sampled from its distribution.
    % The similarity is that they both use canonical (i.e., closed-form) Kalman iterations.
    
    global SimulationStep;
    global N_particle;
    global NominalF NominalH;
    
    n = 1;
    m = 1;
    p = 1;
    q = 1;
    
    %% Init z at k = 0
    z = zeros(n, N_particle);
    mu = x0_hat;
    P = P0;
    
    X_hat = zeros(n,SimulationStep);
    
    %% Time
    for k = 1:SimulationStep
        %% Get X Prior
        for i = 1:N_particle
            z(:,i) = mu + chol(P)*randn(n,1);
        end
        % Propagate z and w to obtain x_prior
        x = zeros(n, N_particle);
        for i = 1:N_particle
            x(:,i) = NominalF(z(:,i), k);
        end
        
        % Get Prior Covariance
        mu_x = mean(x,2);
        P_xx = 0;
        for i = 1:N_particle
            P_xx = P_xx + (x(:,i) - mu_x)*(x(:,i) - mu_x)';
        end
        P_xx = P_xx/N_particle + Q;
        
        %% Predict Measurement
        y_hat = zeros(m, N_particle);
        for i = 1:N_particle
            y_hat(:,i) = NominalH(x(:,i));
        end
        
        % Get Measurement Covariance
        mu_y = mean(y_hat,2);
        P_yy = 0;
        for i = 1:N_particle
            P_yy = P_yy + (y_hat(:,i) - mu_y)*(y_hat(:,i) - mu_y)';
        end
        P_yy = P_yy/N_particle + R;
        
        %% Get cross varaince
        P_xy = 0;
        for i = 1:N_particle
            P_xy = P_xy + (x(:,i) - mu_x)*(y_hat(:,i) - mu_y)';
        end
        P_xy = P_xy/N_particle;
        
        %% Update
        K = P_xy*P_yy^-1;
        mu =  mu_x + K*(Y(:,k) - mu_y);
        X_hat(:,k) = mu;
        P = P_xx - K*P_yy*K';
    end
end

