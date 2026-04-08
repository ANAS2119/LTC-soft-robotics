%% =========================================================
%  NL MPC tracking for first 3 positions: x, y, z
%  Plant state: 6
%  MV: 3
%  Output tracked by MPC: first 3 states only
%% =========================================================

clear; clc;

%% -----------------------------
% User settings
%% -----------------------------
Ts = 0.05;              % controller sample time
p  = 15;                % prediction horizon
m=10;

truthFcn = @koopmanTruthStateFcn;
lnnName=@CfCStateFcnFast;
outputprediction=@cfcOutputFcnFast;
h_corrected=@cfcCorrectHiddenState;

%% -----------------------------
% Initial condition
%% -----------------------------
IC = [0,0.0015,0.000738800516653];
vector_length = 12;
x0_true = zeros(vector_length,1 );
x0_true(1:3)=[IC(1),IC(2),IC(3)];
mv= zeros(3,1);

% LNN controller state is 64x1
h0 = zeros(64,1);
%{
% ---------- TEMPORARY EXAMPLE ----------

% Simulation Setup
tf_sim = 10;
dt_step = Ts; % Step size for the simulation loop (must match Ts )

T_val = 0:dt_step:tf_sim;
N_steps  = length(T_val);
Y_val = zeros(N_steps,6);
Y_val(:,1) = 0.002*sin(10*T_val);          % x ref
Y_val(:,2) = 0.0015*cos(10.5*T_val);         % y ref
Y_val(:,3) = 0.0025*sin(11*T_val + 0.3);   % z ref
% --------------------------------------
%}

%% -----------------------------
% Build reference trajectory
% Only first 3 outputs are used
%% -----------------------------
[Y_ref,T_val]=GenerateRef("N1Hz_input_data.csv", ...
    "N1Hz_output_data.csv", ...
    "N1Hz_time_data.csv");

% MPC simulation grid based on validation duration
t0 = T_val(1);
tf = T_val(end);
timeHist = (t0:Ts:tf)';
N_steps = length(timeHist);

% Interpolate reference to MPC grid
X_ref = interp1(T_val, Y_ref, timeHist, 'linear', 'extrap');

Y_val = zeros(N_steps,6);
Y_val(:,1) = X_ref(:,1);          % x ref
Y_val(:,2) = X_ref(:,2);         % y ref
Y_val(:,3) = X_ref(:,3);   % z ref
X_ref =Y_val;  % N x 6
%% -----------------------------
% Build reference trajectory
% Only first 3 outputs are used
%% -----------------------------


% Optional: smooth if noisy
% X_ref(:,1) = smoothdata(X_ref(:,1),'movmean',5);
% X_ref(:,2) = smoothdata(X_ref(:,2),'movmean',5);
% X_ref(:,3) = smoothdata(X_ref(:,3),'movmean',5);


%% -----------------------------
% Create NL MPC
%% -----------------------------
nx = 64;
ny = 6;
nu = 3;

c = nlmpc(nx, ny, nu);
c.Ts = Ts;
c.PredictionHorizon = p;
c.ControlHorizon = m;

% enable parallel computing
c.Optimization.SolverOptions.UseParallel = true;

% Start parallel pool
if isempty(gcp('nocreate'))
    parpool;
end

% State transition function
c.Model.StateFcn = lnnName;
%c.Jacobian.StateFcn="CfCStateFcnFastJacobian";

c.Model.OutputFcn=@cfcOutputFcnFast;
%c.Jacobian.OutputFcn = "cfcOutputFcnFastJacobian";

c.Model.IsContinuousTime = false;

c.Optimization.SolverOptions.Algorithm = "sqp";


% MV constraints
for i = 1:3
    c.MV(i).Min = -2.5;
    c.MV(i).Max =  3.0;
end

% Weights
c.Weights.OutputVariables = [1e6 1e6 1e6 0 0 0];
c.Weights.ManipulatedVariables = [0 0 0];
c.Weights.ManipulatedVariablesRate = [1e-7 1e-7 1e-7];

% Validate functions
validateFcns(c, h0, zeros(3,1));




x_true=x0_true; % truth plant state (12x1)
h = h0;             % LNN controller state (64x1)
mv = zeros(3,1);

%warm up
for k=1:10
    h=lnnName(h,mv);
end

%% -----------------------------
% Allocate history arrays
%% -----------------------------

xHist    = zeros(N_steps, 6);
yHist    = zeros(N_steps, 3);
uHist    = zeros(N_steps, 3);
hHist    = zeros(N_steps, 64);
refHist  = X_ref(1:N_steps, :);

%% -----------------------------
opt = nlmpcmoveopt;

% Closed-loop simulation
tic
for k = 1:N_steps

    % Get reference for the prediction horizon
    if k + p - 1 <= N_steps
        ref_preview = X_ref(k:k+p-1, :);
    else
        % Pad end with last value
        ref_preview = [X_ref(k:end, :); repmat(X_ref(end,:), p-(N_steps-k+1), 1)];
    end

    % Solve MPC
    [mv,opt,info] = nlmpcmove(c, h, mv, ref_preview,[],opt);

    % Apply plant
    x_true_next = truthFcn(x_true, mv);
    
    % plant measurement
    y_true = x_true_next(1:6);  %Size 6x1

    % Propagate LNN internal state
    h_pred= lnnName(h, mv); %Size 64x1

    % LNN Output
    y_pred = outputprediction(h_pred,mv);  %Size 6x1
    k
   
    % One-step prediction error
    %prediction_error(ct+1) = norm(y_pred(1:6) - y_true(1:6));

    % Update LNN internal state
    %h = h_corrected(h_pred, y_true);
    h=h_pred;
    % Store history
    %xHist(k,:) = x_true(:)';
    yHist(k,:) = y_pred(1:3)';
    hHist(k,:) = h_pred';
    uHist(k,:) = mv(:)';
end
toc

%% -----------------------------
% Compute tracking errors
%% -----------------------------
err = yHist - refHist(:,1:3);
mse_xyz = mean(err.^2, 1);

disp('Tracking MSE:');
disp(table(mse_xyz(1), mse_xyz(2), mse_xyz(3), ...
    'VariableNames', {'MSE_x','MSE_y','MSE_z'}));

%% -----------------------------
% Plot outputs and references
%% -----------------------------
figure;
labels = {'x','y','z'};

for i = 1:3
    subplot(3,1,i);
    plot(timeHist, yHist(:,i), 'b', 'LineWidth', 1.5); hold on;
    plot(timeHist, refHist(:,i), '--r', 'LineWidth', 1.5);
    grid on;
    xlabel('Time (s)');
    ylabel(labels{i});
    legend('Output','Reference','Location','best');
end
sgtitle('LNN-MPC Closed-Loop Tracking: Output vs Reference');

%% -----------------------------
% Plot control inputs
%% -----------------------------
figure;
uLabels = {'u_1','u_2','u_3'};
for i = 1:3
    subplot(3,1,i);
    plot(timeHist, uHist(:,i), 'LineWidth', 1.5);
    grid on;
    xlabel('Time (s)');
    ylabel(uLabels{i});
end
sgtitle('LNN-MPC Control Inputs');
