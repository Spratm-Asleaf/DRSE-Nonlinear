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

function [X_hat] = ParticleFiltering(Y,Q,R,x0_hat,P0)
    % This function realizes the standard Particle Filter.
    % "z" contains posterior state particles at the last time step.
    % "x" contains prior state particles at the current time step.
    
    global SimulationStep;
    global N_particle;
    global NominalF NominalH;
    
    n = 1;
    m = 1;
    p = 1;
    q = 1;
    
    %% Init z at k = 0
    z = zeros(n, N_particle);
    P_Prior = P0;
    for i = 1:N_particle
        z(:,i) = x0_hat + chol(P_Prior)*randn(n,1);
    end
    u = ones(1, N_particle)/N_particle;
    
    X_hat = zeros(n,SimulationStep);
    
    %% Time
    for k = 1:SimulationStep
        %% Get X Prior
        % Sample w
        w = zeros(p, N_particle);
        for i = 1:N_particle
            w(:,i) = zeros(p,1) + chol(Q)*randn(p,1);
        end
        % Propagate z and w to obtain x_prior
        x = zeros(n, N_particle);
        for i = 1:N_particle
            x(:,i) = NominalF(z(:,i), k) + w(:,i);
        end
        
        %% Compute Likelihood
        likelihood = zeros(1, N_particle);
        for i = 1:N_particle
            MU = NominalH(x(:,i));
            SIGMA = R;
            likelihood(i) =  mvnpdf(Y(:,k),MU,SIGMA);
        end

        %% Update weights; obtain posterior weights
        u = u.*likelihood;
        u_sum = sum(u);
        u = u./u_sum;
        
        %% from Prior to Posterior
        z = x;
        
        %% Resampling
%         if 1/(sum(u.^2)) < N_particle*0.25
            [z,u] = Resampling(z,u,n,N_particle);
%         end

        %% Return weighted mean
        X_hat(:,k) = z*u';
    end
end

function [z_new,u_new] = Resampling(z,u,n,N_particle)
    z_new = zeros(n,N_particle);
    cum_sum = cumsum(u);
    for i = 1:N_particle
        index = find(cum_sum<rand);
        if isempty(index)
            index = 0;
        end
        if index(end) == N_particle
            index = N_particle - 1;
        end

        z_new(:,i) = z(:,index(end)+1);
    end
    u_new = ones(1,N_particle)/N_particle;
end