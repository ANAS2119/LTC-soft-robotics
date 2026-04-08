function [A,B]=CfCStateFcnFastJacobian(h,u)
% Jacobian of CfCStateFcnFast
% A = dh_next/dh   (64x64)
% B = dh_next/du   (64x3)

persistent P S
if isempty(P)
    P = load("cfc_params.mat");
    S = load("scalars.mat");
end

h = double(h(:));      % 64x1
u = double(u(:));      % 3x1

%Parameters
Wout = double(P.out_proj_weight);   % 6x64
bout = double(P.out_proj_bias(:));  % 6x1

Wb = double(P.cell_backbone_0_weight);   % 4x73
bb = double(P.cell_backbone_0_bias(:));  % 4x1

Wf = double(P.cell_head_f_weight);       % 64x4
bf = double(P.cell_head_f_bias(:));      % 64x1

Wg = double(P.cell_head_g_weight);       % 64x4
bg = double(P.cell_head_g_bias(:));      % 64x1

Wh = double(P.cell_head_h_weight);       % 64x4
bh = double(P.cell_head_h_bias(:));      % 64x1

Wtau = double(P.cell_w_tau_approx_weight);   % 64x4
btau = double(P.cell_w_tau_approx_bias(:));  % 64x1

ts_model=0.05;
ts = double(ts_model);                 % scalar

u_mean  =S.X_mean(1:3)';
u_scale =S.X_scale(1:3)';

y_mean  =S.Y_mean';
y_scale =S.Y_scale';

%forward pass
y = Wout*h + bout;                  % 6x1

u_norm=(u-u_mean) ./u_scale;
y_norm=(y-y_mean) ./y_scale;
inv_scale = 1 ./ u_scale;

inp= [u_norm(:); y_norm(:)];                       % 9x1
r = [inp; h];                       % 73x1

b = tanh(Wb*r + bb);                     % 4x1

head_h = tanh(Wh*b + bh);                % 64x1
head_g = tanh(Wg*b + bg);                % 64x1
head_f = Wf*b + bf;                      % 64x1
w_tau  = Wtau*b + btau;                  % 64x1

sigma = sigmoid((w_tau + head_f) * ts);  % 64x1

% ---------- derivatives ----------
% --- db/dh and db/du ---
Db = (1 - b.^2);               % 4x1

Wb_scaled = Wb .* Db;         

% dr/dh
dr_dh = [zeros(3,64); Wout; eye(64)];

% dr/du
dr_du = [diag(inv_scale); zeros(6,3); zeros(64,3)];

db_dh = Wb_scaled * dr_dh;     % 4x64
db_du = Wb_scaled * dr_du;     % 4x3

% --- heads derivatives ---
dhh_db = Wh .* (1 - head_h.^2);   % row-wise
dhg_db = Wg .* (1 - head_g.^2);

% --- sigma derivative ---
sig_grad = sigma .* (1 - sigma);      % 64x1

dsig_db = (Wtau + Wf) .* (ts * sig_grad);  % row-wise

% --- final dh_next/db ---
term1 = dhh_db .* (1 - sigma);            % 64x4
term2 = dhg_db .* sigma;                  % 64x4
term3 = dsig_db .* (head_g - head_h);     % 64x4

dhnext_db = term1 + term2 + term3;        % 64x4

% Final Jacobians
A = dhnext_db * db_dh;                    % 64x64
B = dhnext_db * db_du;                    % 64x3
end

function y = sigmoid(x)
y = 1 ./ (1 + exp(-x));
end