import torch
import random
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from ltc_model import RandomWiring, LTCRNN
import numpy as np
import math

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ================================
# 1. Define pendulum dynamics ODE
# ================================
g, l = 9.81, 1.0       # gravity (m/s^2), pendulum length (m)
b     = 0.2            # damping rate k/m (1/s)

# Initial condition: [theta, theta_dot]
theta0 = torch.deg2rad(torch.tensor(30.0))   # 30 degrees
x0 = torch.tensor([theta0, 0.0])              # [theta, theta_dot]

# Define dynamics: dx/dt = f(t, x)
def pendulum_dynamics(t, x):
    theta, dtheta = x[0], x[1]
    dtheta_dt = -(g/l) * torch.sin(theta) - b * dtheta
    return torch.stack([dtheta, dtheta_dt])   # [theta_dot, dtheta_dt]

#Time vector
T = 10
N = 500
t = torch.linspace(0., T, N+1)



# Solve ODE
x = odeint(pendulum_dynamics, x0, t)   # shape [time, 2]

# Extract solutions
theta, dtheta = x[:, 0], x[:, 1]

# Plot results
plt.figure(figsize=(7,4))
plt.plot(t.numpy(), theta.numpy(), label='x1 angular position [rad]')
plt.plot(t.numpy(), dtheta.numpy(), label='x2 angular velocity [rad/s]')
plt.xlabel('Time [s]')
plt.ylabel('States')
plt.title('Damped Pendulum States Trajectory')
plt.legend()
plt.grid(True)
plt.show()

# ================================
# 2. Generate multiple trajectories
# ================================

degrees_tensor = torch.tensor([5, 30, 60, 120, 170, -30, -60])
radians_tensor = torch.deg2rad(degrees_tensor)
init_states = [
    torch.tensor([radians_tensor[0], 0]),
    torch.tensor([radians_tensor[0], 1]),
    torch.tensor([radians_tensor[0], -1.]),
    
    torch.tensor([radians_tensor[1], 0]),
    torch.tensor([radians_tensor[1], 1]),
    torch.tensor([radians_tensor[2], -1]),
    
    torch.tensor([radians_tensor[2], 0]),
    torch.tensor([radians_tensor[2], 1]),
    torch.tensor([radians_tensor[2], -1]),
    
    torch.tensor([radians_tensor[3], 0]),
    torch.tensor([radians_tensor[3], 1]),
    torch.tensor([radians_tensor[3], -1]),
    
    torch.tensor([radians_tensor[4], 0]),
    torch.tensor([radians_tensor[4], 1]),
    torch.tensor([radians_tensor[4], -1]),
    
    torch.tensor([radians_tensor[5], 0]),
    torch.tensor([radians_tensor[5], 1]),
    torch.tensor([radians_tensor[5], -1]),
]

Xs, Ys = [], []
for x0 in init_states:
    xt = odeint(pendulum_dynamics, x0, t)     # [N+1,2]
    Xs.append(xt[:-1])                        # current
    Ys.append(xt[1:])                         # next

def seqify_per_traj(X_list, Y_list, seq_len=80):
    xs, ys = [], []
    for X, Y in zip(X_list, Y_list):
        L = len(X)
        for i in range(L - seq_len):
            xs.append(X[i:i+seq_len])         # [seq_len,2]
            ys.append(Y[i:i+seq_len])         # [seq_len,2]
    return torch.stack(xs), torch.stack(ys)

seq_len = 80
X_seq, Y_seq = seqify_per_traj(Xs, Ys, seq_len)   # [B, L, 2]
print("Seq dataset:", X_seq.shape, Y_seq.shape)

# ================================
# 3. Data preprocessing
# ================================
# normalization
mu  = X_seq.reshape(-1,2).mean(0)
std = X_seq.reshape(-1,2).std(0).clamp_min(1e-6)
Xn, Yn = (X_seq - mu)/std, (Y_seq - mu)/std

#split & loaders
num = len(Xn)
idx = torch.randperm(num)
split = int(0.8 * num)
tr_idx, va_idx = idx[:split], idx[split:]
Xtr, Ytr = Xn[tr_idx], Yn[tr_idx]
Xva, Yva = Xn[va_idx], Yn[va_idx]

train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=64, shuffle=True, drop_last=True)
val_loader   = DataLoader(TensorDataset(Xva, Yva), batch_size=128, shuffle=False)

# ================================
# 4. Build LTC Model
# ================================
wiring = RandomWiring(input_dim=2, output_dim=2, neuron_count=32)
model  = LTCRNN(wiring, input_dim=2, hidden_dim=32, output_dim=2)



# ================================
# 5. Training
# ================================
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_losses, val_losses = [], []

epochs = 2000
horizon = 30   # number of rollout steps to train on (<= seq_len)

# Early stopping setup
patience = 50          # how many epochs to wait for improvement
best_val_loss = float('inf')
patience_counter = 0

for ep in range(1, epochs+1):
    model.train()
    tr_loss = 0.0
    for xb, yb in train_loader:         # xb,yb shapes [B,L,2]
        optimizer.zero_grad()

        B, L, D = xb.shape
        state = xb[:,0,:]               # initial state [B,2]
        preds = []

        # rollout prediction for horizon steps
        for t in range(horizon):
            inp = state.unsqueeze(1)    # [B,1,2]
            out = model(inp)            # [B,1,2]
            state = out[:,-1,:]         # last state [B,2]
            preds.append(state)

        preds = torch.stack(preds, dim=1)   # [B,horizon,2]
        target = yb[:, :horizon, :]         # ground truth [B,horizon,2]

        loss = criterion(preds, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tr_loss += loss.item() * B
    tr_loss /= len(train_loader.dataset)
    train_losses.append(tr_loss)

    # validation
    model.eval()
    with torch.no_grad():
        va_loss = 0.0
        for xb, yb in val_loader:
            B, L, D = xb.shape
            state = xb[:,0,:]
            preds = []
            for t in range(horizon):
                inp = state.unsqueeze(1)
                out = model(inp)
                state = out[:,-1,:]
                preds.append(state)
            preds = torch.stack(preds, dim=1)
            target = yb[:, :horizon, :]
            va_loss += criterion(preds, target).item() * B
        va_loss /= len(val_loader.dataset)
        val_losses.append(va_loss)
        
     # --- Early stopping check ---
    if va_loss < best_val_loss:
        best_val_loss = va_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_ltc_pendulum.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"⏹ Early stopping at epoch {ep} (no improvement for {patience} epochs)")
            break

    # progress
    if ep % 50 == 0:
        print(f"Epoch {ep:4d} | train {tr_loss:.6e} | val {va_loss:.6e}")

print(f"Best val loss: {best_val_loss:.6e}")    


        
plt.figure(figsize=(6,4))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()


model.eval()
with torch.no_grad():
    # pick one batch from validation
    xb, yb = next(iter(val_loader))   # xb, yb shapes [B, L, 2]
    pred = model(xb)                  # [B, L, 2]

# choose a single sequence from the batch
for i in range(8):
    true_seq = yb[i].cpu().numpy()    # [L,2]
    pred_seq = pred[i].cpu().numpy()  # [L,2]

    plt.figure(figsize=(8,4))
    plt.plot(true_seq[:,0], 'b-', label='x1')
    plt.plot(true_seq[:,1], 'g-', label='x2')
    plt.plot(pred_seq[:,0], 'r--', label='x1 LTC')
    plt.plot(pred_seq[:,1], 'm--', label='x2')
    plt.xlabel('Step in sequence')
    plt.ylabel('Normalized states')
    plt.title('One validation sequence: ODE vs LTC prediction')
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
        
# ================================
# Rollout Test from New Initial State

# ----------------------------
# 1. Reload model
# ----------------------------
wiring = RandomWiring(input_dim=2, output_dim=2, neuron_count=32)
model = LTCRNN(wiring, input_dim=2, hidden_dim=32, output_dim=2)
model.load_state_dict(torch.load("best_ltc_pendulum.pth"))
model.eval()
# ================================

rad=math.radians(200) ;  
# Choose a test IC and generate the ground-truth ODE for this exact IC
x0 = torch.tensor([rad, 0.75]) #200°, 0.75 rad/s
x_true = odeint(pendulum_dynamics, x0, t)   # [N+1, 2]

# Rollout with learned next-state map using same dt and normalization
state = x0.view(1, 2)                       # [1,2]
traj = [state.squeeze().tolist()]
with torch.no_grad():
    for _ in range(N):                      # N steps -> N+1 states
        inp  = (state - mu) / std           # normalize
        pred = model(inp.unsqueeze(0))      # [1,1,2]
        state = pred[:, -1, :] * std + mu   # denormalize
        traj.append(state.squeeze().tolist())
traj = torch.tensor(traj)                   # [N+1, 2]

# Plot ODE vs LTC for the SAME initial condition
plt.figure(figsize=(8,4))
plt.plot(t.numpy(), x_true[:,0].numpy(), '--', label='x1 ODE')
plt.plot(t.numpy(), x_true[:,1].numpy(), '--', label='x2 ODE')
plt.plot(t.numpy(), traj[:,0].numpy(), label='x1 LTC')
plt.plot(t.numpy(), traj[:,1].numpy(), label= 'x2 LTC')
plt.xlabel('Time [s]'); plt.ylabel('States'); plt.title('Pendulum: ODE vs LTC')
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()


# Save model
torch.save(model.state_dict(), "ltc_pendulum.pth")
print("✅ Trained LTC saved to ltc_pendulum.pth")
############################################################
import math
import matplotlib.pyplot as plt

# ==================================
# Plot one batch (all sequences)
# ==================================
# x_true: [501, 2] from odeint
# traj:   [501, 2] from LTC rollout
# t:      [501]   time vector

plt.figure(figsize=(10,4))

# θ (angle)
plt.subplot(1,2,1)
plt.plot(t.numpy(), x_true[:,0].numpy(), 'b-', label='θ ODE')
plt.plot(t.numpy(), traj[:,0].numpy(), 'r--', label='θ LTC')
plt.xlabel("Time [s]")
plt.ylabel("θ [rad]")
plt.title("Angle trajectory")
plt.legend(); plt.grid(True)

# θ̇ (angular velocity)
plt.subplot(1,2,2)
plt.plot(t.numpy(), x_true[:,1].numpy(), 'g-', label='θ̇ ODE')
plt.plot(t.numpy(), traj[:,1].numpy(), 'm--', label='θ̇ LTC')
plt.xlabel("Time [s]")
plt.ylabel("θ̇ [rad/s]")
plt.title("Angular velocity trajectory")
plt.legend(); plt.grid(True)

plt.tight_layout()
plt.show()

## plot 5 consequative sequences

# pick a starting index (say 100)
start_idx = 100
num_seq = 5   # number of consecutive sequences to plot

plt.figure(figsize=(10,5))

for k in range(num_seq):
    seq_idx = start_idx + k
    seq = X_seq[seq_idx]   # [L,2] sequence
    
    plt.plot(seq[:,0], label=f'Seq {seq_idx} x1', alpha=0.8)     # angle
    plt.plot(seq[:,1], label=f'Seq {seq_idx} x2', alpha=0.8)    # velocity

plt.xlabel("Step within sequence")
plt.ylabel("States")
plt.title(f"{num_seq} consecutive sequences (starting at {start_idx})")
plt.legend()
plt.grid(True)
plt.show()




## plot trajectories Xs and Ys

# Plot all trajectories stored in Xs and Ys
plt.figure(figsize=(10,5))

# Plot Xs (current states)
for i, X in enumerate(Xs):
    plt.plot(X[:,0].numpy(), label=f'Xs[{i}] θ (angle)')
    plt.plot(X[:,1].numpy(), '--', label=f'Xs[{i}] θ̇ (velocity)')

plt.title("Current states trajectories (Xs)")
plt.xlabel("Time step")
plt.ylabel("State values")
plt.legend()
plt.grid(True)
plt.show()

# Plot Ys (next states)
plt.figure(figsize=(10,5))
for i, Y in enumerate(Ys):
    plt.plot(Y[:,0].numpy(), label=f'Ys[{i}] θ (angle)')
    plt.plot(Y[:,1].numpy(), '--', label=f'Ys[{i}] θ̇ (velocity)')

plt.title("Next states trajectories (Ys)")
plt.xlabel("Time step")
plt.ylabel("State values")
plt.legend()
plt.grid(True)
plt.show()

# Plot all trajectories stored in Xn and Yn (normalized)

# pick some random sequences
num_seq_to_plot = 3
idxs = torch.randint(0, len(Xn), (num_seq_to_plot,))

# Plot Xn
plt.figure(figsize=(10, 5))
for i, idx in enumerate(idxs):
    seq = Xn[idx].numpy()  # shape [80, 2]
    plt.plot(seq[:,0], label=f"x1 Xn seq {i}")     # angle
    plt.plot(seq[:,1], label=f"x2 Xn seq {i}")    # angular velocity
plt.xlabel("Step in sequence")
plt.ylabel("Normalized states")
plt.title("Sample sequences from Xn (inputs)")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()


# Plot Yn
plt.figure(figsize=(10, 5))
for i, idx in enumerate(idxs):
    seq = Yn[idx].numpy()  # shape [80, 2]
    plt.plot(seq[:,0], label=f"y1 Yn seq {i}")     # angle
    plt.plot(seq[:,1], label=f"y1 Yn seq {i}")    # angular velocity
plt.xlabel("Step in sequence")
plt.ylabel("Normalized states")
plt.title("Sample sequences from Yn (targets)")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()

##Visualize trained data
# Grab one batch
xb, yb = next(iter(train_loader))   # xb, yb shapes [B, L, 2]

B, L, D = xb.shape
print("Batch size:", B, "Seq length:", L, "Dim:", D)

# Plot inputs (Xn)
plt.figure(figsize=(10,5))
for i in range(B):   # loop over sequences in this batch
    plt.plot(xb[i,:,0].numpy(), alpha=0.5, color='blue')   # state 1
    plt.plot(xb[i,:,1].numpy(), alpha=0.5, color='green')  # state 2
plt.title("All input sequences (Xn) in one batch")
plt.xlabel("Step")
plt.ylabel("Normalized states")
plt.grid(True)
plt.show()

# Plot targets (Yn)
plt.figure(figsize=(10,5))
for i in range(B):
    plt.plot(yb[i,:,0].numpy(), alpha=0.5, color='red')   # state 1
    plt.plot(yb[i,:,1].numpy(), alpha=0.5, color='orange')# state 2
plt.title("All target sequences (Yn) in one batch")
plt.xlabel("Step")
plt.ylabel("Normalized states")
plt.grid(True)
plt.show()

# plot evaluation versus actual

# Put model in evaluation mode
model.eval()

# Take one full batch from validation loader
xb, yb = next(iter(val_loader))   # xb, yb shapes [B, L, 2]

with torch.no_grad():
    pred = model(xb)              # [B, L, 2]


seq_ids=[5,10,20];

plt.figure(figsize=(10,5))

# x1 (angle)
plt.subplot(1,2,1)
for i in seq_ids:
    true_seq= yb[i].cpu().numpy()
    pred_seq = pred[i].cpu().numpy()
    plt.plot(true_seq[:,0], 'b-', label="ODE x1")
    plt.plot(pred_seq[:,0], 'r--', label="LTC x1")
plt.xlabel("Step")
plt.ylabel("θ [rad]")
plt.title("Angle θ - validation sequences")
plt.grid(True)
    
# Plot θ̇ for all three sequences
plt.subplot(1,2,2)
for i in seq_ids:
    true_seq = yb[i].cpu().numpy()
    pred_seq = pred[i].cpu().numpy()
    plt.plot(true_seq[:,1], 'b-', label="ODE x2")
    plt.plot(pred_seq[:,1], 'r--', label="LTC x2")
plt.xlabel("Step")
plt.ylabel("θ̇ [rad/s]")
plt.title("Angular velocity θ̇ - validation sequences")
plt.grid(True)   
    
    
    
plt.tight_layout()
plt.show()





