function C = cfcOutputFcnFastJacobian(h,u) %#ok<INUSD>

persistent P S
if isempty(P)
    P = load("cfc_params.mat");
    S =  load("scalars.mat");
end

Wout = double(P.out_proj_weight);   % 6x64
y_scaled=double(S.Y_scale(:));


C=y_scaled.*Wout; % dy/dh
end