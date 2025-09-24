import torch
import random
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from LSTM import ImprovedLSTM
import numpy as np
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")


torch.manual_seed(42) 
np.random.seed(42)
random.seed(42)

# ================================
# 1. Define pendulum dynamics ODE
# ================================
g, l = 9.81, 1.0       # gravity (m/s^2), pendulum length (m)
b     = 0.15            # damping rate k/m (1/s)

# Initial condition: [theta, theta_dot]
theta0 = torch.deg2rad(torch.tensor(30.0))   # 30 degrees
x0 = torch.tensor([theta0, 0.0])              # [theta, theta_dot]

# Define dynamics: dx/dt = f(t, x)
def pendulum_dynamics(t, x):
    theta, dtheta = x[0], x[1]
    dtheta_dt = -(g/l) * torch.sin(theta) - b * dtheta
    return torch.stack([dtheta, dtheta_dt])   # [theta_dot, dtheta_dt]

#Time vector
T = 50
N = 2000
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

degrees_tensor = torch.tensor([5, 30, 60, 180, 170, -30, -60])
radians_tensor = torch.deg2rad(degrees_tensor)
init_states = [
    torch.tensor([radians_tensor[0], 0]),
    torch.tensor([radians_tensor[0], 1]),
    torch.tensor([radians_tensor[0], -1.]),
    
    torch.tensor([radians_tensor[1], 0]),
    torch.tensor([radians_tensor[1], 1]),
    torch.tensor([radians_tensor[1], -1]),
    
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

seq_len = 200
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

mu  = mu.to(device)
std = std.to(device)

train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=64, shuffle=True, drop_last=True)
val_loader   = DataLoader(TensorDataset(Xva, Yva), batch_size=128, shuffle=False)

# ================================
# 4. Load LSTM model
# ================================
input_size = 2   # [theta, theta_dot] only
hidden_size = 50
output_size = 2  # predict next [theta, theta_dot]
num_layers = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ImprovedLSTM(input_size, hidden_size, output_size, num_layers).to(device)



# ================================
# 5. Training
# ================================
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_losses, val_losses = [], []

epochs = 500

# Early stopping setup
patience = 50          # how many epochs to wait for improvement
best_val_loss = float('inf')
patience_counter = 0

for ep in range(1, epochs+1):
    #---Training----
    model.train()
    running_train_loss = 0.0
    for sequences, targets in train_loader:
        sequences, targets = sequences.to(device), targets.to(device)

        y_pred = model(sequences)             # [B,L,2]
        loss = criterion(y_pred, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    train_loss = running_train_loss / len(train_loader)
    train_losses.append(train_loss)
        
        #-----Validiation------
        
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for sequences, targets in val_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            y_pred = model(sequences)         # [B,L,2]
            loss = criterion(y_pred, targets)
            running_val_loss += loss.item()

    val_loss = running_val_loss / len(val_loader)
    val_losses.append(val_loss)
       
        
      # ---- Early Stopping ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_lstm_pendulum.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"⏹ Early stopping at epoch {ep}")
            break

    # ---- Progress ----
    if ep % 50 == 0:
        print(f"Epoch {ep:4d} | Train {train_loss:.6e} | Val {val_loss:.6e}")

print(f"✅ Best val loss: {best_val_loss:.6e}") 


        
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
    xb, yb = next(iter(val_loader))
    xb, yb = xb.to(device), yb.to(device)
    pred_seq = model(xb)    # [batch, seq_len, 2]

# Plot multiple sequences
n_show = min(10, xb.shape[0])             # show up to 10 sequences
seq_len = xb.shape[1]

fig, axes = plt.subplots(n_show, 1, figsize=(10, 2*n_show), sharex=True)

for i in range(n_show):
    true_seq = yb[i].cpu().numpy()        # [L,2]
    pred_seq_i = pred_seq[i].cpu().numpy()
    
    ax = axes[i]
    ax.plot(true_seq[:,0], 'b-', label='θ true' if i==0 else "")
    ax.plot(true_seq[:,1], 'g-', label='θ̇ true' if i==0 else "")
    ax.plot(pred_seq_i[:,0], 'r--', label='θ pred' if i==0 else "")
    ax.plot(pred_seq_i[:,1], 'm--', label='θ̇ pred' if i==0 else "")
    ax.set_ylabel("State")
    ax.grid(True)

axes[0].legend(loc="upper right")
axes[-1].set_xlabel("Step in sequence")
fig.suptitle("Pendulum: True vs LSTM (10 validation sequences)", y=0.92, fontsize=14)
plt.tight_layout()
plt.show()


        
# ================================
# Rollout Test from New Initial State
# ================================
model.eval()

rad = math.radians(200)
x0 = torch.tensor([rad, 0.75], dtype=torch.float32, device=device)  # [2]

# normalize initial state
state = (x0 - mu.to(device)) / std.to(device)   # [2]

# add batch + seq_len dims
state = state.unsqueeze(0).unsqueeze(0)         # [1,1,2]

traj_pred = [x0.cpu().numpy()]
steps = 2000

with torch.no_grad():
    for step in range(steps):
        y_pred = model(state)                   # [1,1,2]
        

        # always collapse time dimension
        y_pred = y_pred[:, -1, :]               # [1,2]

        # denormalize
        y_denorm = y_pred * std.to(device) + mu.to(device)
        traj_pred.append(y_denorm.squeeze(0).cpu().numpy())

        # re-normalize and re-shape for next input
        state = (y_denorm - mu.to(device)) / std.to(device)   # [1,2]
        state = state.unsqueeze(1)                            # [1,1,2]

traj_pred = np.array(traj_pred)   # [steps+1, 2]        

# ground truth ODE rollout (keep on CPU, since it's just for comparison)
x_true = odeint(pendulum_dynamics, x0.cpu(), t.cpu())

# Plot ODE vs LTC for the SAME initial condition
plt.figure(figsize=(10,5))
plt.plot(t.numpy(), x_true[:,0].numpy(), 'b--', label='θ true (ODE)')
plt.plot(t.numpy(), x_true[:,1].numpy(), 'g--', label='θ̇ true (ODE)')
plt.plot(t[:steps+1].numpy(), traj_pred[:,0], 'r-', label='θ pred (LSTM)')
plt.plot(t[:steps+1].numpy(), traj_pred[:,1], 'm-', label='θ̇ pred (LSTM)')
plt.xlabel("Time [s]")
plt.ylabel("States")
plt.title("Pendulum Rollout: ODE vs LSTM (IC: 200°, 0.75 rad/s)")
plt.legend()
plt.grid(True)
plt.show()


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
num_seq_to_plot = 30
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



