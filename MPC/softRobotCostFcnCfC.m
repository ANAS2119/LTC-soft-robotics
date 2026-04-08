function J = softRobotCostFcnCfC(stage, h, u) %#ok<INUSD>
% h = 64x1 CfC hidden state
% physical output y = 6x1

y = cfcOutputFcnFast(h, u);

Q = diag([100 80 100 0 0 0]);
R = 0.05*eye(3);

J = y'*Q*y + u'*R*u;
end