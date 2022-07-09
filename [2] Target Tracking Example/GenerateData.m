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

global SimulationStep;
global MeasurementUncertaintyAmp;
    
global delta_t;
delta_t = 0.5;                      %sampling time

F = [
    1 delta_t
    0 1
];
F = diagmx(F,F);

G = [                   % I do not like this matrix, but one reviewer is forcing me to use this matrix.
    delta_t^2/2         % But, anyway, using this matrix or the matrix below does not matter.
    delta_t
];
% G = [                   % In fact, I like this matrix more because it seems simpler.
%     delta_t
%     1
% ];
%{
    n.b., Different definition of G means different meaning of process noise
    w_k. In the first case, w_k means the acceleration noise, but in the
    second case, w_k means the velocity noise.
%}
G = diagmx(G,G);

Q = diag([0.25,0.25]);              % true process noise covariance  
R = diag([0.1 0.0001]*10);          % true measurement noise covariance       

x = [5 0 5 0]';                     % target's initial position and velocity
v = 2;                              % target's real moving speed
theta = linspace(0,pi/2,SimulationStep);

x0 = [0 0]';                        % sensor's initial position
v0 = 1;                             % sensor's moving speed
theta0 = pi/2;                      % sensor's moving heading

X = zeros(4, SimulationStep);       % target's real state: real position and velocity
Y = zeros(2, SimulationStep);       % real measurement: real range and azimuth
X0 = zeros(2, SimulationStep);      % sensor's real position
for k = 1:SimulationStep
    % The sensor moves along the vertical axi.
    % Hence, its heading angle is "theta0 = pi/2".
    x0(1) = x0(1) + v0*cos(theta0)*delta_t;
    x0(2) = x0(2) + v0*sin(theta0)*delta_t;
    X0(:,k) = x0;
    
    % The target moves along a curved trajectory
    w = [0 0]' + chol(Q)*randn(2,1);
    x(2) = v*cos(theta(k));
    x(4) = v*sin(theta(k));
    x = F*x + G*w;
    
    % Sensor's positioning error
    x_noise = randn*MeasurementUncertaintyAmp;      % sensor's positioning error in x-axis
    y_noise = randn*MeasurementUncertaintyAmp;      % sensor's positioning error in y-axis
    
    % Generate true measurements
    y = [
        sqrt((x(1) - x0(1) + x_noise)^2 + (x(3) - x0(2) + y_noise)^2) + chol(R(1,1))*randn;     % range
        % Since positions are guaranteed to be positive in this example,
        %       there is no difference between "atan" and "atan2".
        atan((x(3) - x0(2) + y_noise)/(x(1) - x0(1) + x_noise)) + chol(R(2,2))*randn;           % azimuth
    ];

    % Save system state and system measurement
    X(:,k) = x;
    Y(:,k) = y;
end

%% Nominal process noise covariance
% Because we do not know the true value. 
% We use a large Q to hedge aginst the restrictive model assumption of the CV model.
Q = diag([5 5]);                    

%% Nominal measurement noise covariance
% Because we know the true value of measurement noise covariance.
R = R;



