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


clear all;
close all;
clc;

%% Reproducibility
% If you want to reproduce results in my paper, keep the statement below.
% If you want to try other possibilities, comment out the statement below.
rng(45);                                % Use whatever value you want. But I like 45.

%% Global constants
global SimulationStep;                  % How many time steps in each episode?
SimulationStep = 100;

global N_particle;                      % Number of particles.
N_particle = 100;

%% Global functions
global TrueF TrueH NominalF NominalH;
% True functions
TrueF =     @(x,k) x/2 + 25*x/(1+x^2) + 8*cos(1.2*k);
TrueH =     @(x,k) x^2/20 + 0.5*sin(x);
% Nominal functions
NominalF =  @(x,k) x/2 + 25*x/(1+x^2) + 8*cos(1.2*k);
if true
    NominalH =  @(x,k) x^2/20;                       % exist model error
else
    NominalH =  @(x,k) x^2/20 + 0.5*sin(x);          % exist no model error
end

%% Scene Simulator
GenerateData;                           % Generate true state trajectory.
MonteCarloEpisode = 100;                % How many independent episodes?

%% Assign Data
PF_Avrg_Time_Array = zeros(1,MonteCarloEpisode);
PF_RMSE_Array = zeros(1,MonteCarloEpisode);
PF_RMSE_Array_Every_Episode = cell(1, MonteCarloEpisode);
    
GA_Avrg_Time_Array = zeros(1,MonteCarloEpisode);
GA_RMSE_Array = zeros(1,MonteCarloEpisode);
GA_RMSE_Array_Every_Episode = cell(1, MonteCarloEpisode);
    
UKF_Avrg_Time_Array = zeros(1,MonteCarloEpisode);
UKF_RMSE_Array = zeros(1,MonteCarloEpisode); 
UKF_RMSE_Array_Every_Episode = cell(1, MonteCarloEpisode);

RPF_Avrg_Time_Array = zeros(1,MonteCarloEpisode);
RPF_RMSE_Array = zeros(1,MonteCarloEpisode);
RPF_RMSE_Array_Every_Episode = cell(1, MonteCarloEpisode);

%% Do we need to show performance versus \theta?
isFindGoodTheta = false;
if isFindGoodTheta
    Theta = 1:0.5:10;
else
    Theta = 5;
end
theta_len = length(Theta);
RTAMSE_VS_Theta = zeros(1, theta_len);
for theta_index = 1:theta_len
    theta = Theta(theta_index);
    if isFindGoodTheta
        disp(['Trying theta = ' num2str(theta)]);
    end
    %% Filtering
    for iter = 1:MonteCarloEpisode
        %% Scene Simulator
        if false                        % Do you need different trajectory for every Monte Carlo run?
            GenerateData;               % Whether true or false does not matter.
        end                             % Try it if you want!

        if ~isFindGoodTheta
            disp(['Current Monte Carlo Episode: ' num2str(iter)]);
        end

        %% Initial Condition
        % N.B., when there is no uncertainty, the smaller the P0 is, the better
        % the standard particle filter is.
        x0_hat = 0;
        P0 = 1;

        %% Particle Filter
        tic
        [X_PF_hat] = ParticleFiltering(Y,Q,R,x0_hat,P0);
        PF_Avrg_Time_Array(iter) = toc/SimulationStep;
        PF_RMSE_Array(iter) = sqrt(mean((X_PF_hat - X).^2));
        PF_RMSE_Array_Every_Episode{iter} = sqrt((X_PF_hat - X).^2);

        %% Gaussian Approximation Filter (Ensemble Kalman Filter; EnKF)
        tic
        [X_GA_hat] = GaussianApproximation(Y,Q,R,x0_hat,P0);
        GA_Avrg_Time_Array(iter) = toc/SimulationStep;
        GA_RMSE_Array(iter) = sqrt(mean(((X_GA_hat - X).^2)));
        GA_RMSE_Array_Every_Episode{iter} = sqrt(((X_GA_hat - X).^2));

        %% Gaussian Approximation Filter (Unscented Kalman Filter; UKF)
        tic
        [X_UKF_hat] = UKF(Y,Q,R,x0_hat,P0);
        UKF_Avrg_Time_Array(iter) = toc/SimulationStep;
        UKF_RMSE_Array(iter) = sqrt(mean(((X_UKF_hat - X).^2)));
        UKF_RMSE_Array_Every_Episode{iter} = sqrt(((X_UKF_hat - X).^2));

        %% Distributionally Robust (i.e., maximum-entropy) Particle Filter
        tic
        [X_MaxEnt_hat] = RobustParticleFiltering(Y,Q,R,x0_hat,P0,theta);
        RPF_Avrg_Time_Array(iter) = toc/SimulationStep;
        RPF_RMSE_Array(iter) = sqrt(mean(((X_MaxEnt_hat - X).^2)));
        RPF_RMSE_Array_Every_Episode{iter} = sqrt(((X_MaxEnt_hat - X).^2));
    end
    
    RTAMSE_VS_Theta(theta_index) = mean(RPF_RMSE_Array);
end

if isFindGoodTheta
    plot(Theta, RTAMSE_VS_Theta, 'r', 'linewidth', 2);
    return;
end

%% Show Overall Performance Statistics
disp('----------------------------------');
disp(['Number of Particle: ' num2str(N_particle) ]);
disp('----------------------------------');
PF_Avrg_Time = mean(PF_Avrg_Time_Array);
PF_Avrg_RMSE = mean(PF_RMSE_Array);
display(['PF RTAMSE: ' num2str(PF_Avrg_RMSE) '        Time: ' num2str(PF_Avrg_Time)]);

GA_Avrg_Time = mean(GA_Avrg_Time_Array);
GA_Avrg_RMSE = mean(GA_RMSE_Array);
display(['GA-EnKF RTAMSE: ' num2str(GA_Avrg_RMSE) '        Time: ' num2str(GA_Avrg_Time)]);

UKF_Avrg_Time = mean(UKF_Avrg_Time_Array);
UKF_Avrg_RMSE = mean(UKF_RMSE_Array);
display(['UKF RTAMSE: ' num2str(UKF_Avrg_RMSE) '        Time: ' num2str(UKF_Avrg_Time)]);

RPF_Avrg_Time = mean(RPF_Avrg_Time_Array);
RPF_Avrg_RMSE = mean(RPF_RMSE_Array);
display(['RPF RTAMSE: ' num2str(RPF_Avrg_RMSE) '        Time: ' num2str(RPF_Avrg_Time)]);

%% Show Trajectory
Time = 1:SimulationStep;
plot(Time,X,'r','linewidth',2);
hold on;
plot(Time,X_PF_hat,'b--','linewidth',2);
plot(Time,X_GA_hat,'m--','linewidth',2);
plot(Time,X_MaxEnt_hat,'g-','linewidth',2);

% Plot Format
title('Trajectory','FontSize', 15);
set(gca, 'FontSize', 18);
set(gca, 'TitleFontWeight', 'normal');
xlabel('Time','FontSize', 20, 'Interpreter', 'latex');
leg1 = legend({'True', 'PF', 'GA-EnKF', 'RPF'}, 'Interpreter', 'latex', 'Location', 'northwest');
set(leg1,'FontSize',15);


%% Show Filtering Performance v.s. Episode
figure;
EpisodeArray = 1:MonteCarloEpisode;
% plot(EpisodeArray, GA_RMSE_Array,'g-.','linewidth',2);
plot(EpisodeArray, UKF_RMSE_Array,'g-x','linewidth',1);
hold on;
plot(EpisodeArray, PF_RMSE_Array,'r-s','linewidth',1);
plot(EpisodeArray, RPF_RMSE_Array,'b','linewidth',2);
legend('GA-UKF','PF','RPF');
set(gca, 'FontSize', 18);
ylabel('RTAMSE','FontSize', 20, 'Interpreter', 'latex');
xlabel('Episode','FontSize', 20, 'Interpreter', 'latex');
axis([1 MonteCarloEpisode ...
    min([RPF_RMSE_Array, UKF_RMSE_Array, PF_RMSE_Array]) - 0.1 ...
    max([RPF_RMSE_Array, UKF_RMSE_Array, PF_RMSE_Array]) + 0.1 ...
    ]);

%% Show Filtering Performance v.s. Time
figure;
PF_RMSE_Array_Every_Episode_Avrg = 0;
GA_RMSE_Array_Every_Episode_Avrg = 0;
UKF_RMSE_Array_Every_Episode_Avrg = 0;
RPF_RMSE_Array_Every_Episode_Avrg = 0;
for i = 1:MonteCarloEpisode
    PF_RMSE_Array_Every_Episode_Avrg = PF_RMSE_Array_Every_Episode_Avrg + PF_RMSE_Array_Every_Episode{i};
    GA_RMSE_Array_Every_Episode_Avrg = GA_RMSE_Array_Every_Episode_Avrg + GA_RMSE_Array_Every_Episode{i};
    UKF_RMSE_Array_Every_Episode_Avrg = UKF_RMSE_Array_Every_Episode_Avrg + UKF_RMSE_Array_Every_Episode{i};
    RPF_RMSE_Array_Every_Episode_Avrg = RPF_RMSE_Array_Every_Episode_Avrg + RPF_RMSE_Array_Every_Episode{i};
end
PF_RMSE_Array_Every_Episode_Avrg = PF_RMSE_Array_Every_Episode_Avrg/MonteCarloEpisode;
GA_RMSE_Array_Every_Episode_Avrg = GA_RMSE_Array_Every_Episode_Avrg/MonteCarloEpisode;
UKF_RMSE_Array_Every_Episode_Avrg = UKF_RMSE_Array_Every_Episode_Avrg/MonteCarloEpisode;
RPF_RMSE_Array_Every_Episode_Avrg = RPF_RMSE_Array_Every_Episode_Avrg/MonteCarloEpisode;

Time = 1:SimulationStep;
% plot(Time, GA_RMSE_Array_Every_Episode_Avrg,'g-.','linewidth',2);
plot(Time, UKF_RMSE_Array_Every_Episode_Avrg,'g-x','linewidth',2);
hold on;
plot(Time, PF_RMSE_Array_Every_Episode_Avrg,'r-s','linewidth',1);
plot(Time, RPF_RMSE_Array_Every_Episode_Avrg,'b','linewidth',2);
% legend('RPF','GA-EnKF','UKF','PF');
legend('GA-UKF','PF','RPF');
% legend('RPF','PF');
set(gca, 'FontSize', 18);
ylabel('RMSE','FontSize', 20, 'Interpreter', 'latex');
xlabel('Time','FontSize', 20, 'Interpreter', 'latex');

%% This is the end of simulation. Thanks. Good bye.





