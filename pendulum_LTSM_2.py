"""
LSTM , using input noises with one initial conditions

"""
import torch
import random
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from LSTM import ImprovedLSTM
import numpy as np
import math
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error, r2_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")


torch.manual_seed(42) 
np.random.seed(42)
random.seed(42)

# ============================================
# 1. Define pendulum dynamics ODE with inputs
# ============================================
g, l = 9.81, 1.0       # gravity (m/s^2), pendulum length (m)
b     = 0.15            # damping rate k/m (1/s)

# Initial condition: [theta, theta_dot]
theta0 = torch.deg2rad(torch.tensor(90.0))   # 90 degrees
x0 = torch.tensor([theta0, 0.0])              # [theta, theta_dot]

# Define dynamics: dx/dt = f(t,,x u)
def pendulum_dynamics(t, x, u):
    theta, dtheta = x[0], x[1]
    dtheta_dt = -(g/l) * torch.sin(theta) - b * dtheta +u
    return torch.stack([dtheta, dtheta_dt])   # [theta_dot, dtheta_dt]

#Time vector
T = 60
N = 3000
t = torch.linspace(0., T, N+1)
dt=0.03

#Generate Input noise signal
def multi_sine(N, dt, amps=(0.1, 0.2, 0.3), freqs=(0.5, 1.3, 2.1)):
    n = np.arange(N+1)*dt
    u = sum(a*np.sin(2*np.pi*f*n) for a,f in zip(amps, freqs))
    return u

#Input u(t)
input_ = multi_sine(N, dt) # numpy array [N+1]


#Build an interpolation function for the input
input_func = interp1d(np.arange(N+1)*dt, input_, kind='linear', fill_value="extrapolate")

#ODE with inputs
def pendulum_with_input(t, x):
    u_val = float(input_func(t.item()))   # convert to scalar
    return pendulum_dynamics(t, x, torch.tensor(u_val, dtype=torch.float32))


# Solve ODE
x = odeint(pendulum_with_input, x0, t)   # shape [time, 2]

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
# 2. Generate input noises
# ================================

amps=torch.tensor([0.1, 0.2, 0.3, 0.4]);
freqs=torch.tensor([0.5, 1.3, 1.8]);

input_noise = [
    torch.tensor([amps[0], freqs[0]]),
    torch.tensor([amps[0]*2, freqs[1]]),
    torch.tensor([amps[0]*3, freqs[2]]),
    
    torch.tensor([amps[0]*4, freqs[0]]),
    torch.tensor([amps[0]*5, freqs[1]]),
    torch.tensor([amps[0]*6, freqs[2]]),
    
    torch.tensor([amps[0]*7, freqs[0]]),
    torch.tensor([amps[0]*8, freqs[1]]),
    torch.tensor([amps[0]*9, freqs[2]]),
    
    torch.tensor([amps[1], freqs[0]]),
    torch.tensor([amps[1]*2, freqs[1]]),
    torch.tensor([amps[1]*3, freqs[2]]),
    
    torch.tensor([amps[1]*4, freqs[0]]),
    torch.tensor([amps[2], freqs[1]]),
    torch.tensor([amps[2]*2, freqs[2]]),
    
    torch.tensor([amps[2]*3, freqs[0]]),
    torch.tensor([amps[3], freqs[1]]),
    torch.tensor([amps[3]*2, freqs[2]]),
    
]

def simulate_pendulum_with_input(x0, t, N, amps, freqs, dt):

    # generate input
    u_signal = multi_sine(N, dt, amps=amps, freqs=freqs)

    # interpolation function
    u_func = interp1d(np.arange(N+1)*dt, u_signal, kind='linear', fill_value="extrapolate")

    # dynamics wrapper
    def dynamics_with_input(t, x):
        u_val = float(u_func(t.item()))  # get interpolated input at time t
        return pendulum_dynamics(t, x, torch.tensor(u_val, dtype=torch.float32, device=x.device))

    # solve ODE
    x = odeint(dynamics_with_input, x0, t)
    return x


Xs, Ys = [], []
for input in input_noise:
    amp, freq = input[0].item(), input[1].item()
    xt = simulate_pendulum_with_input(x0, t, N, amps=(amp,), freqs=(freq,), dt=dt) # [N+1,2] 
    Xs.append(xt[:-1])                        # current
    Ys.append(xt[1:])                       # next


# ================================
# 3. Create Train,Valid, Test
# ================================

train_trajs = Xs[:12]
val_trajs   = Xs[12:15]
test_trajs  = Xs[15:]

#stack training trajectories
X_train_raw = np.vstack(train_trajs)   # [12*3000, 2]

#compute normalization params on TRAIN only
mu  = X_train_raw.mean(axis=0)
std = X_train_raw.std(axis=0)

#normalize train/val/test

train_norm = (train_trajs - mu) / std
val_norm   = (val_trajs - mu) / std
test_norm  = (test_trajs - mu) / std


# ================================
# 4. create sequences Train,Valid, Test
# ================================

def create_sequences(data, seq_len=300):
    xs, ys = [], []
    for traj in data:
        for i in range(len(traj) - seq_len):
            x = traj[i:i+seq_len]
            y = traj[i+1:i+seq_len+1]
            xs.append(x); ys.append(y)
    return np.array(xs), np.array(ys)

Xtr, Ytr = create_sequences(train_norm, seq_len=300)
Xva, Yva = create_sequences(val_norm,   seq_len=300)
Xte, Yte = create_sequences(test_norm,  seq_len=300)


print("Seq dataset:", Xtr.shape, Ytr.shape)

#convert to tensor
Xtr = torch.tensor(Xtr, dtype=torch.float32)
Ytr = torch.tensor(Ytr, dtype=torch.float32)
Xva = torch.tensor(Xva, dtype=torch.float32)
Yva = torch.tensor(Yva, dtype=torch.float32)
Xte = torch.tensor(Xte, dtype=torch.float32)
Yte = torch.tensor(Yte, dtype=torch.float32)


train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=64, shuffle=True, drop_last=True)
val_loader   = DataLoader(TensorDataset(Xva, Yva), batch_size=128, shuffle=False)
test_loader   = DataLoader(TensorDataset(Xte, Yte), batch_size=128, shuffle=False)

# ================================
# 5. Load LSTM model
# ================================
input_size = 2   # [theta, theta_dot] only
hidden_size = 50
output_size = 2  # predict next [theta, theta_dot]
num_layers = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ImprovedLSTM(input_size, hidden_size, output_size, num_layers).to(device)


# ================================
# 6. Training
# ================================
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_losses, val_losses = [], []

epochs = 200

# Early stopping setup
patience = 40          # how many epochs to wait for improvement
best_val_loss = float('inf')
patience_counter = 0

for ep in range(1, epochs+1):
    #---Training----
    model.train()
    running_train_loss = 0.0
    for sequences, targets in train_loader:
        sequences, targets = sequences.to(device), targets.to(device)

        optimizer.zero_grad()
        
        y_pred = model(sequences)             # [B,L,2]
        loss = criterion(y_pred, targets)

        
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
    if ep % 20 == 0:
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


# ================================
# 7. Inference
# ================================

#--Model Loading
model.load_state_dict(torch.load("best_lstm_pendulum.pth", map_location=device))
model.eval()

all_preds = []
all_targets = []

with torch.no_grad():
    for sequences, targets in test_loader:
        sequences, targets= sequences.to(device), targets.to(device)
        y_pred = model(sequences)
        
        all_preds.append(y_pred.cpu())
        all_targets.append(targets.cpu())

#stack and convert to numpy    
all_preds = torch.cat(all_preds, dim=0).numpy()     # [Ntest, L, 2]
all_targets = torch.cat(all_targets, dim=0).numpy()

#Evaluate MSE and R2 oon flatten sequence
mse = mean_squared_error(all_targets.reshape(-1,2), all_preds.reshape(-1,2))
r2  = r2_score(all_targets.reshape(-1,2), all_preds.reshape(-1,2))

print(f"Test MSE: {mse:.4e}")
print(f"Test R² : {r2:.4f}")

# Plot multiple sequences

# Pick one sequence index
idx = 5000   # you can change to any sequence index from 0 to 8099

# Take from dataset
x_seq = Xte[idx]   # input sequence [300, 2]
y_true = Yte[idx]  # target sequence [300, 2]

# Convert to tensor and run through model
with torch.no_grad():
    seq_tensor = torch.tensor(x_seq, dtype=torch.float32, device=device).unsqueeze(0)  # [1,300,2]
    y_pred = model(seq_tensor).cpu().numpy()  # [1,300,2]
y_pred = y_pred.squeeze(0)   # [300,2]

# --- Plot true vs predicted ---
plt.figure(figsize=(10,5))
plt.plot(y_true[:,0], 'b-', label='x1 true')
plt.plot(y_true[:,1], 'g-', label='x2 true')
plt.plot(y_pred[:,0], 'r--', label='x1 pred')
plt.plot(y_pred[:,1], 'm--', label='x2 pred')

plt.xlabel("Step in sequence")
plt.ylabel("States")
plt.title(f"Test sequence {idx}: True vs Predicted")
plt.legend()
plt.grid(True)
plt.show()


############################################################
