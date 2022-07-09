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

function [X_hat] = UKF(Y,Q,R,x0_hat,P0)
    % This function realizes the Unscented Kalman Filter (UKF).
    % Simon, D. (2006). Optimal state estimation: Kalman, H infinity, and nonlinear approaches. John Wiley & Sons.
    % See Pages 448-450.
    
    global SimulationStep;
    global NominalF NominalH;
    
    n = 1;
    m = 1;
    p = 1;
    q = 1;
    
    N_particle = 2*n;           % number of sigma points; in UKF, this vaue is fixed as "2*n".
    
    %% Init z at k = 0
    % Eq. (14.57)
        z = zeros(n, N_particle);
        mu = x0_hat;
        P = P0;
    
    X_hat = zeros(n,SimulationStep);
    
    %% Time
    for k = 1:SimulationStep
        %% Get X Prior
        % Eq. (14.58)
            temp = chol(n*P);
            for i = 1:n
                z(:,i) = mu + (temp(i, :))';
                z(:,n+i) = mu - (temp(i, :))';
            end
        % Propagate z and w to obtain x_prior
        % Eq. (14.59)
            x = zeros(n, N_particle);
            for i = 1:N_particle
                x(:,i) = NominalF(z(:,i), k);
            end
        
        % Get Prior Covariance
        % Eq. (14.60)
            mu_x = mean(x,2);
        % Eq. (14.61)
            P_xx = 0;
            for i = 1:N_particle
                P_xx = P_xx + (x(:,i) - mu_x)*(x(:,i) - mu_x)';
            end
            P_xx = P_xx/N_particle + Q;
        
        %% Predict Measurement
        % Eq. (14.62). N.B., Not necessarily to be implemented.
%             temp = chol(n*P_xx);
%             for i = 1:n
%                 x(:,i) = mu_x + (temp(i, :))';
%                 x(:,n+i) = mu_x - (temp(i, :))';
%             end
        
        % Eq. (14.63)
            y_hat = zeros(m, N_particle);
            for i = 1:N_particle
                y_hat(:,i) = NominalH(x(:,i));
            end
        
        % Get Measurement Covariance
        % Eq. (14.64)
            mu_y = mean(y_hat,2);
        % Eq. (14.65)
            P_yy = 0;
            for i = 1:N_particle
                P_yy = P_yy + (y_hat(:,i) - mu_y)*(y_hat(:,i) - mu_y)';
            end
            P_yy = P_yy/N_particle + R;
        
        %% Get cross varaince
        % Eq. (14.66)
            P_xy = 0;
            for i = 1:N_particle
                P_xy = P_xy + (x(:,i) - mu_x)*(y_hat(:,i) - mu_y)';
            end
            P_xy = P_xy/N_particle;
        
        %% Update
        % Eq. (14.67)
            K = P_xy*P_yy^-1;
            mu =  mu_x + K*(Y(:,k) - mu_y);
            X_hat(:,k) = mu;
            P = P_xx - K*P_yy*K';
    end
end

