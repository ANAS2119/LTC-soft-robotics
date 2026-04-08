function h_next  = LTCStateFcnFast(h, u)
% Fast plain-MATLAB LTC one-step function
% h = full neuron state after 10 internal steps, size 8x1
% u = control input 3*1
% output = next full neuron state 8x1

persistent P S
if isempty(P)
    P =load("ltc_params.mat");
    S=load("scalars.mat");
end

% Full recurrent state
h = h(:);   % 8x1
u_raw=double(u(:)); % 3x1

% Physical state is the first 6 neurons
x = h(1:6);   % 6x1

%normalized input
u_norm=(u_raw-S.X_mean(1:3)') ./S.X_scale(1:3)';

% Input format used in training: [u; x]
inp_raw = [u_norm(:); x(:)];   % 9x1
inp=double(inp_raw(:));

% Parameters
gleak = softplus(P.cell_neuron_gleak(:));                % 8x1
vleak = P.cell_neuron_vleak(:);                          % 8x1
cm    = softplus(P.cell_neuron_cm(:));                   % 8x1

w     = softplus(P.cell_neuron_w);                       % 8x8
sigma = P.cell_neuron_sigma;                             % 8x8
mu    = P.cell_neuron_mu;                                % 8x8
erev  = P.cell_neuron_erev;                              % 8x8

sw    = softplus(P.cell_neuron_sensory_w);              % 9x8
ssig  = P.cell_neuron_sensory_sigma;                    % 9x8
smu   = P.cell_neuron_sensory_mu;                       % 9x8
serev = P.cell_neuron_sensory_erev;                     % 9x8

mask  = P.cell_neuron_sparsity_mask;                    % 8x8
smask = P.cell_neuron_sensory_sparsity_mask;            % 9x8

ode_unfolds = double(P.ode_unfolds);
eps_ = double(P.epsilon);

% Full recurrent state
v = double(h(:));   % 8x1


% -------- sensory effects (computed once) --------
% sigmoid(inputs, sensory_mu, sensory_sigma)
% inputs is 9x1, smu/ssig are 9x8
inp_mat = repmat(inp, 1, size(smu,2));   % 9x8
sens_gate = 1 ./ (1 + exp(-(ssig .* (inp_mat - smu))));
sens_act = sw .* sens_gate;
sens_act = sens_act .* smask;

sens_rev_act = sens_act .* serev;

w_num_sens = sum(sens_rev_act, 1)';   % 8x1
w_den_sens = sum(sens_act, 1)';       % 8x1

cm_t = cm ./ (1.0 / ode_unfolds);     % elapsed_time = 1

% -------- recurrent ODE unfolds --------
for k = 1:ode_unfolds
    % sigmoid(v, mu, sigma), where v is 8x1 and mu/sigma are 8x8
    v_mat = repmat(v, 1, size(mu,2));   % 8x8
    rec_gate = 1 ./ (1 + exp(-(sigma .* (v_mat - mu))));
    
    w_act = w .* rec_gate;
    w_act = w_act .* mask;

    rev_act = w_act .* erev;
    w_num = sum(rev_act, 1)' + w_num_sens;   % 8x1
    w_den = sum(w_act, 1)'   + w_den_sens;   % 8x1

    numerator = cm_t .* v + gleak .* vleak + w_num;
    denominator = cm_t + gleak + w_den;

    v = numerator ./ (denominator + eps_);
end
h_next=double(v);
end


function y = softplus(x)
% numerically stable softplus
y = log1p(exp(-abs(x))) + max(x,0);
end

%-----------------------------------------------
%Output Function
%-----------------------------------------------
function y = lnnOutputFcnFast(h, u)
% Output = physical states only
persistent P
if isempty(P)
    P=load("scalars.mat");
end
y = h(1:6).*P.Y_scale(:)+P.Y_mean(:);
end
