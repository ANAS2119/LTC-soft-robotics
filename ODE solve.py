import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt

# Define system matrices in torch
A = torch.tensor([[-3., 1.],
                  [ 0.,-10.]])
B = torch.tensor([[0.],
                  [1.]])
u = 5.0  # constant step input

# Dynamics function
def dynamics(t, x):
    x = x.reshape(-1, 1)                # column vector
    dxdt = A @ x + B * u
    return dxdt.flatten()               # return as 1D tensor

# Initial condition
x0 = torch.tensor([1., 1.])

# Time points for solution
t = torch.linspace(0., 2., 200)   # 0 to 2 seconds, 200 samples

# Solve ODE
x = odeint(dynamics, x0, t)       # shape [200, 2]

# Plot results
plt.plot(t.numpy(), x[:,0].detach().numpy(), label='x1(t)')
plt.plot(t.numpy(), x[:,1].detach().numpy(), label='x2(t)')
plt.xlabel('Time [s]')
plt.ylabel('States')
plt.title('State Response with Step Input u(t)=5')
plt.legend()
plt.grid(True)
plt.show()
