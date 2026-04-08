function results = validateKoopmanModel(inputFile, outputFile, timeFile, makePlots, A, B, doPerturbation)
% Validate Koopman model using dataset
%
% A,B must already exist in workspace
% Uses user-defined functions:
%   koopmanPsi(x)
%   koopmanPlantStep(x,u,A,B)

if nargin < 4
    makePlots = true;
end
if nargin < 7
    doPerturbation = true;
end

%% Load validation data
uu = readmatrix(inputFile)';     % 3 x T
xx = readmatrix(outputFile)';    % 6 x T
timeT = readmatrix(timeFile);

%% =========================
%  CLEAN RUN (baseline)
% ==========================
% Build augmented state (12 states)

xx_aug = zeros(12,size(xx,2));

xx_aug(1:6,:) = xx(1:6,:);

% initialize derivative states
xx_aug(7:12,1) = xx_aug(1:6,1);

% discrete differences
xx_aug(7:12,2:end) = xx_aug(1:6,2:end) - xx_aug(1:6,1:end-1);

x_true = xx_aug(:,1);

% Simulation

M = length(timeT)-1;

x_pred = zeros(12,M+1);
x_pred(:,1) = x_true;

for k = 1:M

    u = uu(:,k);

    x_pred(:,k+1) = koopmanPlantStep(x_pred(:,k),u,A,B);

end

% Error

errorT = xx_aug(:,1:M+1) - x_pred;

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
results.x_true = xx_aug;
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

    x_pred_pert = zeros(12,M+1);
    x_pred_pert(:,1) = x_true;

    for k = 1:M

        u = uu_pert(:,k);

        x_pred_pert(:,k+1) = koopmanPlantStep(x_pred_pert(:,k),u,A,B);

    end

    %  Step 3: robustness metrics ----

    % 3.1 Difference magnitude: perturbed prediction vs clean prediction
    drift = x_pred_pert - x_pred;  % 12 x (M+1)
    driftNorm = vecnorm(drift, 2, 1);   % 1 x (M+1)

    meanDrift  = mean(driftNorm);
    maxDrift   = max(driftNorm);
    finalDrift = driftNorm(end);
    
     % 3.2 Stability: boundedness of perturbed prediction
    stateNormPert = vecnorm(x_pred_pert, 2, 1); % 1 x (M+1)
    maxStateNorm  = max(stateNormPert);
    maxAbsState   = max(abs(x_pred_pert), [], 2); % 12 x 1

    stabilityThreshold = 0.02; % choose based on expected physical scale
    isStable = all(isfinite(x_pred_pert), 'all') && max(abs(x_pred_pert(:))) < stabilityThreshold;

    % 3.3 Smoothness / physical plausibility
    dx_pert  = diff(x_pred_pert, 1, 2);   % 12 x M

    meanAbsDiff      = mean(abs(dx_pert), 2);   % 12 x 1

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

    for j = 1:4

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
    sgtitle('koopman Validation: Measured vs Predicted')

    figure
    bar(maeVec)
    title('MAE of koopman Model')
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
        for j = 1:3
            subplot(3,1,j)
            plot(timeT, x_pred(j,:), 'b', 'LineWidth', 1); hold on
            plot(timeT, x_pred_pert(j,:), 'r--', 'LineWidth', 1);
            grid on
            ylabel(['x_' num2str(j)])
        end
        xlabel('time (s)')
        legend('clean states','perturbed states','Location','northeast')
        sgtitle('Koopman Predictions: Clean vs Perturbed Input')

        figure;
        plot(timeT, driftNorm, 'LineWidth', 1.5)
        grid on
        xlabel('time (s)')
        ylabel('||x_{pert} - x_{clean}||_2')
        title('Koopman, Drift Magnitude Over Time')

        figure;
        plot(timeT, stateNormPert, 'LineWidth', 1.5)
        grid on
        xlabel('time (s)')
        ylabel('||x_{pert}||_2')
        title('Koopman, Perturbed Prediction Norm (Stability Check)')

        figure;
        plot(timeT(2:end), sensitivity, 'LineWidth', 1.5)
        grid on
        xlabel('time (s)')
        ylabel('Sensitivity')
        title('Koopman, Sensitivity to Input Perturbation')

        figure;
        bar(meanAbsDiff)
        grid on
        title(' Koopman, Mean |First Difference| per Output')
        ylabel('mean |dx|')
    end
end

end