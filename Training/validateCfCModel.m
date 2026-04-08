function results = validateCfCModel(inputFile, outputFile, timeFile, makePlots, doPerturbation)
% Validate CfC model using dataset
% Normal mode:
%   - read u, x_true, time
%   - initialize model state
%   - run model forward
%   - compare predicted with measured
%   - compute MAE
%   - plots
%
% Perturbation mode:
%   - create perturbed input
%   - rerun model from same initial state
%   - analyze:
%       1) Stability
%       2) Smoothness / physical plausibility
%       3) Drift magnitude
%       4) Sensitivity
% Uses user-defined functions:
%   CfCStateFcnFast(h,u)
%   cfcOutputFcnFast(h,u)

if nargin < 4
    makePlots = true;
end
if nargin < 5
    doPerturbation = true;
end

%% Load validation data
uu = readmatrix(inputFile)';     % 3 x T
xx = readmatrix(outputFile)';    % 6 x T
timeT = readmatrix(timeFile);
M = length(timeT)-1;

%% =========================
%  CLEAN RUN (baseline)
% ==========================
% CfCStateFcnFast
h = zeros(64,1);
for k=1:5
    h=CfCStateFcnFast(h,uu(:,k));
end
% Simulation
x_pred = zeros(6,M+1);
x_pred(:,1) =cfcOutputFcnFast( h, zeros(3,1));

for k = 1:M

    u = uu(:,k);

    h=CfCStateFcnFast(h,u); % 8x1
    x_pred(:,k+1) = cfcOutputFcnFast(h,u);

end

% Error

errorT = xx(:,1:M+1) - x_pred;

maeX = mae(errorT(1,:));
maeY = mae(errorT(2,:));
maeZ = mae(errorT(3,:));
maeA = mae(errorT(4,:));
maeB = mae(errorT(5,:));
maeG = mae(errorT(6,:));

maeVec = [maeX*1e3 maeY*1e3 maeZ*1e3 maeA maeB maeG];

% Results structure

results.mae = maeVec;
results.time = timeT;
results.x_true = xx;
results.x_pred = x_pred;
results.error = errorT;

%% =========================
%  PERTURBED RUN
% ==========================
if doPerturbation
    %Step 1: Build perturbation
    uu_pert = uu;

    amp   = [0.8; 0.9; 0.95];    % amplitude
    freq  = [1; 1.1; 1.2];        % Hz
    phase = [0.0; 0.4; 0.8];        % rad
    
    for i = 1:3
        uu_pert(i,:) = uu_pert(i,:) + amp(i) * sin(2*pi*freq(i)*timeT' + phase(i));
    end

    % Step 2: rerun model from SAME initial hidden state ----

    x_pred_pert = zeros(6, M+1);
    x_pred_pert(:,1) = cfcOutputFcnFast(h, zeros(3,1));

    for k = 1:M
        u = uu_pert(:,k);
        h = CfCStateFcnFast(h, u);
        x_pred_pert(:,k+1) = cfcOutputFcnFast(h, u);
    end

    %  Step 3: robustness metrics ----

    % 3.1 Difference magnitude: perturbed prediction vs clean prediction
    drift = x_pred_pert - x_pred;  % 6 x (M+1)
    driftNorm = vecnorm(drift, 2, 1);   % 1 x (M+1)

    meanDrift  = mean(driftNorm);
    maxDrift   = max(driftNorm);
    finalDrift = driftNorm(end);
    
     % 3.2 Stability: boundedness of perturbed prediction
    stateNormPert = vecnorm(x_pred_pert, 2, 1); % 1 x (M+1)
    maxStateNorm  = max(stateNormPert);
    maxAbsState   = max(abs(x_pred_pert), [], 2); % 6 x 1

    stabilityThreshold = 0.02; % choose based on expected physical scale
    isStable = all(isfinite(x_pred_pert), 'all') && max(abs(x_pred_pert(:))) < stabilityThreshold;

    % 3.3 Smoothness / physical plausibility
    dx_pert  = diff(x_pred_pert, 1, 2);   % 6 x M

    meanAbsDiff      = mean(abs(dx_pert), 2);   % 6 x 1

     % 3.4 Sensitivity: output change normalized by input perturbation
    du = uu_pert(:,1:M) - uu(:,1:M);           % 3 x M
    duNorm = vecnorm(du, 2, 1);                % 1 x M

    % drift for k=2:end corresponds roughly to input over 1:M
    driftNormStep = driftNorm(2:end);          % 1 x M

    epsVal = 1e-8;
    sensitivity = driftNormStep ./ (duNorm + epsVal);

    meanSensitivity = mean(sensitivity);
    maxSensitivity  = max(sensitivity);

    % Store perturbation results
    results.perturbation.enabled = true;
    results.perturbation.u_pert = uu_pert(:,1:M);
    results.perturbation.x_pred_pert = x_pred_pert;

    results.perturbation.drift.signal = drift;
    results.perturbation.drift.norm = driftNorm;
    results.perturbation.drift.mean = meanDrift;
    results.perturbation.drift.max = maxDrift;
    results.perturbation.drift.final = finalDrift;

    results.perturbation.stability.isStable = isStable;
    results.perturbation.stability.stateNorm = stateNormPert;
    results.perturbation.stability.maxStateNorm = maxStateNorm;
    results.perturbation.stability.maxAbsState = maxAbsState;

    results.perturbation.smoothness.meanAbsDiff = meanAbsDiff;

    results.perturbation.sensitivity.signal = sensitivity;
    results.perturbation.sensitivity.mean = meanSensitivity;
    results.perturbation.sensitivity.max = maxSensitivity;

    results.perturbation.config.amp = amp;
    results.perturbation.config.freq = freq;
    results.perturbation.config.phase = phase;
else
    results.perturbation.enabled=false;
end

%% Plot results

if makePlots

    figure('Position',[100 100 1200 700])

    for j = 1:6

        subplot(6,1,j)

        if j<=3
            plot(timeT,1*xx(j,:),'k','LineWidth',1)
            hold on
            plot(timeT,1*x_pred(j,:),'b','LineWidth',1)
        else
            plot(timeT,xx(j,:),'k','LineWidth',1)
            hold on
            plot(timeT,x_pred(j,:),'b','LineWidth',1)
        end

        grid on
        ylabel(['x_' num2str(j)])

    end

    xlabel('time (s)')
    legend('truth states','predicted','Location','northeast')
    sgtitle('CfC,  Validation: Measured vs Predicted')

    figure
    bar(maeVec)
    title('MAE of CfC Model')
    grid on

    if doPerturbation
        figure('Position',[100 100 1000 500]);
        for i = 1:3
            subplot(3,1,i)
            plot(timeT(1:M), uu(i,1:M), 'b', 'LineWidth', 1); hold on
            plot(timeT(1:M), uu_pert(i,1:M), 'r--', 'LineWidth', 1);
            grid on
            ylabel(['u_' num2str(i)])
        end
        xlabel('time (s)')
        legend('clean input','perturbed input','Location','northeast')
        sgtitle('Input Signals: Clean vs Perturbed')

        figure('Position',[100 100 1200 700]);
        for j = 1:6
            subplot(6,1,j)
            plot(timeT, x_pred(j,:), 'b', 'LineWidth', 1); hold on
            plot(timeT, x_pred_pert(j,:), 'r--', 'LineWidth', 1);
            grid on
            ylabel(['x_' num2str(j)])
        end
        xlabel('time (s)')
        legend('clean states','perturbed states','Location','northeast')
        sgtitle('CfC Predictions: Clean vs Perturbed Input')

        figure;
        plot(timeT, driftNorm, 'LineWidth', 1.5)
        grid on
        xlabel('time (s)')
        ylabel('||x_{pert} - x_{clean}||_2')
        title('CfC, Drift Magnitude Over Time')

        figure;
        plot(timeT, stateNormPert, 'LineWidth', 1.5)
        grid on
        xlabel('time (s)')
        ylabel('||x_{pert}||_2')
        title('CfC, Perturbed Prediction Norm (Stability Check)')

        figure;
        plot(timeT(2:end), sensitivity, 'LineWidth', 1.5)
        grid on
        xlabel('time (s)')
        ylabel('Sensitivity')
        title('CfC, Sensitivity to Input Perturbation')

        figure;
        bar(meanAbsDiff)
        grid on
        title('CfC, Mean |First Difference| per Output')
        ylabel('mean |dx|')
    end
end

end