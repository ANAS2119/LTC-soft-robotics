function [X_ref, T_val]=GenerateRef(inputFile, outputFile, timeFile)
%% %% Load validation data
U = readmatrix(inputFile);     % T x 3
Y = readmatrix(outputFile);    % T x 6
T = readmatrix(timeFile);   % T x 1

T_val = T(:);
X_ref = Y(:,1:3);

%Ts = T(2)-T(1);
%% Extract X,Y,Z References
%X_ref= Y(:,1:3);
%% Downsample to MPC Ts
%Ts_mpc  = 0.0075; 
%ratio = round(Ts_mpc / Ts);
%X_ref = X_ref(1:ratio:end, :);