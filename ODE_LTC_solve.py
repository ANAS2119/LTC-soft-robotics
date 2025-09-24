import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from ltc_model import RandomWiring, LTCRNN

# ================================
# 1. Define Mass–Spring ODE
# ================================
A = torch.tensor([[-3., 1.],
                  [ 0., -10.]])
B = torch.tensor([[0.],
                  [1.]])
u = 5.0  # constant step input

def dynamics(t, x):
    x = x.reshape(-1, 1)                
    dxdt = A @ x + B * u
    return dxdt.flatten()               

# Time vector
T = 2.0
N = 200
t = torch.linspace(0., T, N+1)

# ================================
# 2. Generate multiple trajectories
# ================================
init_states = [
    torch.tensor([1., 1.]),
    torch.tensor([0., 1.]),
    torch.tensor([2., 0.]),
    torch.tensor([1., -1.])
]

X_list, Y_list = [], []
for x0 in init_states:
    x_traj = odeint(dynamics, x0, t)  # shape [N+1, 2]
    X_list.append(x_traj[:-1])        # current states
    Y_list.append(x_traj[1:])         # next states

X_all = torch.cat(X_list, dim=0)  # [num_samples, 2]
Y_all = torch.cat(Y_list, dim=0)  # [num_samples, 2]

print("Dataset shapes:", X_all.shape, Y_all.shape)

# Extract solutions
position, velocity = x_traj[:, 0], x_traj[:, 1]

# Plot results
plt.figure(figsize=(7,4))
plt.plot(t.numpy(), position.numpy(), label=' x1(t) position [m]')
plt.plot(t.numpy(), velocity.numpy(), label='x2(t) position [m/s]')
plt.xlabel('Time [s]')
plt.ylabel('States')
plt.title('Spring Mass States Trajectory')
plt.legend()
plt.grid(True)
plt.show()


# ================================
# 3. Turn into sequences
# ================================
def to_sequences(X, Y, seq_len=20):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i:i+seq_len])
        ys.append(Y[i:i+seq_len])
    return torch.stack(xs), torch.stack(ys)

seq_len = 20
X_seq, Y_seq = to_sequences(X_all, Y_all, seq_len)
print("Training input shape:", X_seq.shape)
print("Training target shape:", Y_seq.shape)

# ================================
# 4. Build LTC Model
# ================================
wiring = RandomWiring(input_dim=2, output_dim=2, neuron_count=16)  # use more neurons
ltc_model = LTCRNN(wiring, input_dim=2, hidden_dim=16, output_dim=2)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(ltc_model.parameters(), lr=1e-3)

# ================================
# 5. Training Loop
# ================================
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    preds = ltc_model(X_seq)       # [batch, seq_len, 2]
    loss = criterion(preds, Y_seq)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss={loss.item():.6f}")

# Save model
torch.save(ltc_model.state_dict(), "ltc_mass_spring.pth")
print("✅ Trained LTC saved to ltc_mass_spring.pth")

# ================================
# 6. Rollout Test from New Initial State
# ================================
wiring = RandomWiring(input_dim=2, output_dim=2, neuron_count=16)
ltc_model = LTCRNN(wiring, input_dim=2, hidden_dim=16, output_dim=2)
ltc_model.load_state_dict(torch.load("ltc_mass_spring.pth"))
ltc_model.eval()
steps = 200
state = torch.tensor([[1.0, 1.0]])  # test initial condition
trajectory = [state.squeeze().tolist()]

with torch.no_grad():
    for _ in range(steps):
        inp = state.unsqueeze(0)    # [1,1,2]
        pred = ltc_model(inp)       # [1,1,2]
        state = pred[:, -1, :]      # [1,2]
        trajectory.append(state.squeeze().tolist())

trajectory = torch.tensor(trajectory)  # [steps+1, 2]

# Compare with true ODE
x_true = odeint(dynamics, torch.tensor([1., 1.]), t)

plt.figure(figsize=(7,4))
plt.plot(x_true[:,0], "--", label="ODE x1(t)")
plt.plot(x_true[:,1], "--", label="ODE x2(t)")
plt.plot(trajectory[:,0], label="LTC x1(t)")
plt.plot(trajectory[:,1], label="LTC x2(t)")
plt.xlabel("Step")
plt.ylabel("States")
plt.title("Mass–Spring: ODE vs LTC")
plt.legend()
plt.grid(True)
plt.show()
