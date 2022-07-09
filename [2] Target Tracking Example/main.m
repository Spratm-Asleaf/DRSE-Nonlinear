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

global MeasurementUncertaintyAmp;       % Uncertainty Amplitude: for measurement dynamics only.
MeasurementUncertaintyAmp = 1;          % Set to "0" if no measurement uncertainty.

global N_particle;                      % Number of particles.
N_particle = 1000;

%% Scene Simulator
GenerateData;                           % Generate sensor's and target's trajectory.
MonteCarloEpisode = 50;                 % How many independent episodes?

%% RMSE Index
% "position" and "velocity" do not have the same unit.
% hence, you can calculate the RMSE in individual or together.
% use [1;3] for "position", and use [2;4] for "velocity".
RMSE_Mode = 'velocity';
switch RMSE_Mode                               
    case 'position'
        DataIndex = [1;3];                      
    case 'velocity'
        DataIndex = [2;4];
    case 'all'
        RMSE_Mode = 'position + velocity';
        DataIndex = 1:4;
    otherwise
        error('main :: Error in RMSE mode');
end

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

%% Filtering
for iter = 1:MonteCarloEpisode
    %% Scene Simulator
    if false                        % Do you need different trajectory for every Monte Carlo run?
        GenerateData;               % Whether true or false does not matter.
    end                             % Try it if you want!
    
    disp(['Current Monte Carlo Episode: ' num2str(iter)]);
    
    %% Initial Condition
    % N.B., when there is no uncertainty, the smaller the P0 is, the better
    % the standard particle filter is.
    x0_hat = [5 0 5 0]';
    P0 = diag([0.1 0.1 0.1 0.1]);
    % P0 = P0 * 2;                  % Try this if you want. You may see that
                                    % the standard particle filter
                                    % degrades, no matter whether uncertainty exists.
                                    % However, when the number of particles
                                    % is sufficiently large, the value of
                                    % P0 becomes less important.
    
    %% Particle Filter
    tic
    [X_PF_hat] = ParticleFiltering(Y,X0,F,G,Q,R,x0_hat,P0);
    PF_Avrg_Time_Array(iter) = toc/SimulationStep;
    PF_RMSE_Array(iter) = sqrt(mean(sum((X_PF_hat(DataIndex,:) - X(DataIndex,:)).^2)));
    PF_RMSE_Array_Every_Episode{iter} = sqrt(sum((X_PF_hat(DataIndex,:) - X(DataIndex,:)).^2));

    %% Gaussian Approximation Filter (Ensemble Kalman Filter; EnKF)
    tic
    [X_GA_hat] = GaussianApproximation(Y,X0,F,G,Q,R,x0_hat,P0);
    GA_Avrg_Time_Array(iter) = toc/SimulationStep;
    GA_RMSE_Array(iter) = sqrt(mean(sum((X_GA_hat(DataIndex,:) - X(DataIndex,:)).^2)));
    GA_RMSE_Array_Every_Episode{iter} = sqrt(sum((X_GA_hat(DataIndex,:) - X(DataIndex,:)).^2));

    %% Gaussian Approximation Filter (Unscented Kalman Filter; UKF)
    tic
    [X_UKF_hat] = UKF(Y,X0,F,G,Q,R,x0_hat,P0);
    UKF_Avrg_Time_Array(iter) = toc/SimulationStep;
    UKF_RMSE_Array(iter) = sqrt(mean(sum((X_UKF_hat(DataIndex,:) - X(DataIndex,:)).^2)));
    UKF_RMSE_Array_Every_Episode{iter} = sqrt(sum((X_UKF_hat(DataIndex,:) - X(DataIndex,:)).^2));

    %% Distributionally Robust (i.e., maximum-entropy) Particle Filter
    tic
    [X_MaxEnt_hat] = RobustParticleFiltering(Y,X0,F,G,Q,R,x0_hat,P0);
    RPF_Avrg_Time_Array(iter) = toc/SimulationStep;
    RPF_RMSE_Array(iter) = sqrt(mean(sum((X_MaxEnt_hat(DataIndex,:) - X(DataIndex,:)).^2)));
    RPF_RMSE_Array_Every_Episode{iter} = sqrt(sum((X_MaxEnt_hat(DataIndex,:) - X(DataIndex,:)).^2));
end

%% Show Overall Performance Statistics
disp('----------------------------------');
if MeasurementUncertaintyAmp > 0
    disp('Exist Measurement Uncertainty::');
else
    disp('Do Not Exist Measurement Uncertainty::');
end
disp('----------------------------------');
disp(['Number of Particle: ' num2str(N_particle) ]);
disp('----------------------------------');
PF_Avrg_Time = mean(PF_Avrg_Time_Array);
PF_Avrg_RMSE = mean(PF_RMSE_Array);
display(['PF RTAMSE (' RMSE_Mode '): ' num2str(PF_Avrg_RMSE) '        Time: ' num2str(PF_Avrg_Time)]);

GA_Avrg_Time = mean(GA_Avrg_Time_Array);
GA_Avrg_RMSE = mean(GA_RMSE_Array);
display(['GA-EnKF RTAMSE (' RMSE_Mode '): ' num2str(GA_Avrg_RMSE) '        Time: ' num2str(GA_Avrg_Time)]);

UKF_Avrg_Time = mean(UKF_Avrg_Time_Array);
UKF_Avrg_RMSE = mean(UKF_RMSE_Array);
display(['GA-UKF RTAMSE (' RMSE_Mode '): ' num2str(UKF_Avrg_RMSE) '        Time: ' num2str(UKF_Avrg_Time)]);

RPF_Avrg_Time = mean(RPF_Avrg_Time_Array);
RPF_Avrg_RMSE = mean(RPF_RMSE_Array);
display(['RPF RTAMSE (' RMSE_Mode '): ' num2str(RPF_Avrg_RMSE) '        Time: ' num2str(RPF_Avrg_Time)]);

%% Show Trajectory
plot(X0(1,:),X0(2,:),'k','linewidth',2);
hold on;
plot(X(1,:),X(3,:),'r','linewidth',2);
plot(X_PF_hat(1,:),X_PF_hat(3,:),'b--','linewidth',2);
plot(X_GA_hat(1,:),X_GA_hat(3,:),'m--','linewidth',2);
plot(X_MaxEnt_hat(1,:),X_MaxEnt_hat(3,:),'g-','linewidth',2);

% Plot Format
title('Trajectories','FontSize', 15);
set(gca, 'FontSize', 18);
set(gca, 'TitleFontWeight', 'normal');
ylabel('y-axis','FontSize', 20, 'Interpreter', 'latex');
xlabel('x-axis','FontSize', 20, 'Interpreter', 'latex');
leg1 = legend({'Sensor', 'Target (True)', 'Target (PF)', 'Target (GA-EnKF)', 'Target (RPF)'}, 'Interpreter', 'latex', 'Location', 'northwest');
set(leg1,'FontSize',15);
axis([-10 80 -10 80]);

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





