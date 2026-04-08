function [c, h0] = softRobotControlcfc(lnnName,TruthName, outputprediction)
% Compare:
%   1) Koopman-MPC (truth-model MPC)
%   2) LNN-MPC
%
% truth plant = Koopman discrete model
% LNN model   = fast MATLAB cfc model

% -----------------------------
% Initial conditions
% -----------------------------
% Truth plant state is 12x1:
% [x y z a b g dx dy dz da db dg]
IC = 0.02;
vector_length = 12;
half_length = vector_length / 2;
x0_true = zeros(vector_length,1 );
x0_true(1:half_length)=IC;

%x0_true = zeros(12,1);

% LNN controller state is 8x1
% first 6 = physical outputs, last 2 = internal memory neurons
h0 = zeros(64,1);

Ts = 0.0075;
p = 15;

% Simulation Setup
tf_sim = 2.5;
dt_step = Ts; % Step size for the simulation loop (must match Ts )

T_val = 0:dt_step:tf_sim;
Steps  = length(T_val);

truthFcn = str2func(TruthName);
lnnFcn   = str2func(lnnName);
cfcoutpred=str2func(outputprediction);

% -----------------------------
% Open-loop simulation (u = 0)
% -----------------------------

figure;
Xol = zeros(12, Steps+1);
Xol(:,1) = x0_true;
u_zero = zeros(3,1);

% Simulation loop
x = x0_true;
for k = 1:Steps
    x = truthFcn(x, u_zero);
    Xol(:,k+1) = x;
end

tt = (0:Steps)*Ts;
labels = {'x', 'y', 'z', '\alpha', '\beta', '\gamma'};

% Create 6 subplots
for i = 1:6
    subplot(3, 2, i); % 3 rows, 2 columns, position i
    plot(tt, Xol(i, :), '.');
    xlabel('time (s)');
    ylabel(labels{i});
    grid on;
end

% Add a main title for the entire figure
sgtitle('Open-Loop (Koopman truth plant, u = 0)');

% ==================================================
% Closed-loop simulation 1: MPC uses truth Koopman model
% ==================================================
c = nlmpcMultistage(p, 12, 3);   % truth controller state = 12
c.Ts = Ts;
% Example MV constraints (edit as needed)
for i = 1:3
    c.ManipulatedVariables(i).Min = -2.5;
    c.ManipulatedVariables(i).Max =  3;
end

for ct = 2:(p+1)
    c.Stages(ct).CostFcn = "softRobotCostFcn";
end

c.Model.StateFcn = TruthName;
c.Model.IsContinuousTime = false;

simdata = getSimulationData(c);

x_true = x0_true;
mv = zeros(3,1);

figure;
labels = {'x', 'y', 'z', '\alpha', '\beta', '\gamma'};

% Initialize subplots with formatting
for i = 1:3
    subplot(3, 2, i);
    hold on;
    grid on;
    xlabel('time (s)');
    ylabel(labels{i});
end
sgtitle('Closed-Loop (dot = Koopman-MPC), circle = LNN-MPC)');

tic
for ct = 0:Steps
    for i=1:3
        subplot(3,2,i);
        plot(ct*Ts, x_true(i), '.');
    end
    drawnow

    [mv, simdata, info] = nlmpcmove(c, x_true, mv, simdata); %#ok<ASGLU>
    x_true = truthFcn(x_true, mv);
end
toc
hold on;
% ==================================================
% Closed-loop simulation 2: MPC uses LNN model
% ==================================================
c = nlmpcMultistage(p, 64, 3);    % LNN controller state = 64
c.Ts = Ts;

for i = 1:3
    c.ManipulatedVariables(i).Min = -2.5;
    c.ManipulatedVariables(i).Max =  3;
end

for ct = 2:(p+1)
    c.Stages(ct).CostFcn = "softRobotCostFcnCfC";
end

c.Model.StateFcn = lnnName;
c.Model.IsContinuousTime = false;

simdata = getSimulationData(c);

x_true = x0_true;   % truth plant state (12x1)
h = h0;             % LNN controller state (64x1)
mv = zeros(3,1);

%warm up
for k=1:5
    h=CfCStateFcnFast(h,mv);
end

tic
for ct = 0:Steps
    for i=1:3
        subplot(3,2,i);
        plot(ct*Ts, x_true(i), 'o');
    end
    drawnow

    % MPC solve using LNN state
    [mv, simdata, info] = nlmpcmove(c, h, mv, simdata); %#ok<ASGLU>

    % Propagate true Koopman plant
    x_true_next = truthFcn(x_true, mv);

    % Propagate LNN internal state
    h_pred= lnnFcn(h, mv);
    
    % plant measurement
    y_true = x_true_next(1:6);  %Size 6x1

    % Update the output prediction using the corrected hidden state
    y_pred = cfcoutpred(h_pred,mv);  %Size 6x1
  
   
    % One-step prediction error
    prediction_error(ct+1) = norm(y_pred(1:6) - y_true(1:6));

    % Update LNN internal state
    h =h_pred; 

    % Update true plant
    x_true = x_true_next;
end
toc

figure
plot(prediction_error,'LineWidth',1.2)
grid on
xlabel('time step')
ylabel('||y_{CfC} - y_{true}||')
title('CfC one-step prediction error')

end

%----------------------------------------------------------
