function J = softRobotCostFcn(stage, h, u) %#ok<INUSD>

x = h(1:6);

Q = diag([100 50 50 100 50 100]);
R = 0.05*eye(3);

J = x'*Q*x + u'*R*u;

end