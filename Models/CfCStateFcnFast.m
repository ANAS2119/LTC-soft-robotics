function h_next  = CfCStateFcnFast(h, u)
% Fast plain-MATLAB Cfc one-step function
% h = full neuron state after 10 internal steps, size 64x1
% u = control input 3*1
%Return: output = next full neuron state 64x1

persistent P S
if isempty(P)
    P =load("cfc_params.mat");
    S=load("scalars.mat");
end

% Full recurrent state
h = double(h(:));   % 64x1
u_raw=double(u(:)); % 3x1


% -----------------------------
% Current physical output y = out_proj(h)
% -----------------------------
Wout = double(P.out_proj_weight);   % 6x64
bout = double(P.out_proj_bias(:));  % 6x1

y_norm = Wout*h + bout;                  % 6x1

% -----------------------------
% Build CFC input [u; y]
% -----------------------------
%normalized input
u_norm=(u_raw-S.X_mean(1:3)') ./S.X_scale(1:3)';

inp= [u_norm(:); y_norm(:)];                       % 9x1

% Parameters
% -----------------------------
% Backbone input [inp; h]
% -----------------------------
r = [inp; h];                       % 73x1
% -----------------------------
% Backbone: Linear + tanh
% -----------------------------
Wb = double(P.cell_backbone_0_weight);   % 4x73
bb = double(P.cell_backbone_0_bias(:));  % 4x1

b = tanh(Wb*r + bb);                     % 4x1

% -----------------------------
% Heads
% -----------------------------
Wf = double(P.cell_head_f_weight);       % 64x4
bf = double(P.cell_head_f_bias(:));      % 64x1

Wg = double(P.cell_head_g_weight);       % 64x4
bg = double(P.cell_head_g_bias(:));      % 64x1

Wh = double(P.cell_head_h_weight);       % 64x4
bh = double(P.cell_head_h_bias(:));      % 64x1

Wtau = double(P.cell_w_tau_approx_weight);   % 64x4
btau = double(P.cell_w_tau_approx_bias(:));  % 64x1

head_h = tanh(Wh*b + bh);                % 64x1
head_g = tanh(Wg*b + bg);                % 64x1
head_f = Wf*b + bf;                      % 64x1
w_tau  = Wtau*b + btau;                  % 64x1

% -----------------------------
% Time step
% -----------------------------
ts_model=0.05;
ts = double(ts_model);                 % scalar

sigma = sigmoid((w_tau + head_f) * ts);  % 64x1

h_next = head_h .* (1.0 - sigma) + sigma .* head_g;
h_next = double(h_next);
end

function y = sigmoid(x)
y = 1 ./ (1 + exp(-x));
end