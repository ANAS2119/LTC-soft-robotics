import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from ltc_model import RandomWiring, LTCRNN
from torch.utils.data import DataLoader, TensorDataset

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

Xs, Ys = [], []
for x0 in init_states:
    xt = odeint(dynamics, x0, t)       # [N+1,2]
    Xs.append(xt[:-1])                 # current
    Ys.append(xt[1:])                  # next

def seqify_per_traj(X_list, Y_list, seq_len=80):
    xs, ys = [], []
    for X, Y in zip(X_list, Y_list):
        L = len(X)
        for i in range(L - seq_len):
            xs.append(X[i:i+seq_len])  # [seq_len,2]
            ys.append(Y[i:i+seq_len])  # [seq_len,2]
    return torch.stack(xs), torch.stack(ys)

seq_len = 80
X_seq, Y_seq = seqify_per_traj(Xs, Ys, seq_len)   # [B,L,2]
print("Seq dataset:", X_seq.shape, Y_seq.shape)

# ================================
# 3. Normalize dataset
# ================================
mu  = X_seq.reshape(-1,2).mean(0)
std = X_seq.reshape(-1,2).std(0).clamp_min(1e-6)
Xn, Yn = (X_seq - mu)/std, (Y_seq - mu)/std

# Split train/val
num = len(Xn)
idx = torch.randperm(num)
split = int(0.8 * num)
Xtr, Ytr = Xn[idx[:split]], Yn[idx[:split]]
Xva, Yva = Xn[idx[split:]], Yn[idx[split:]]

train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=64, shuffle=True, drop_last=True)
val_loader   = DataLoader(TensorDataset(Xva, Yva), batch_size=128, shuffle=False)


# ================================
# 4. Build LTC Model
# ================================
wiring = RandomWiring(input_dim=2, output_dim=2, neuron_count=32)  # use more neurons
ltc_model = LTCRNN(wiring, input_dim=2, hidden_dim=32, output_dim=2)

ltc_model.load_state_dict(torch.load("best_ltc_pendulum.pth"))

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(ltc_model.parameters(), lr=1e-3)

# ================================
# 5. Training Loop
# ================================

train_losses, val_losses = [], []

epochs   = 2000
horizon  = 30   # rollout steps for loss
best_val = float("inf")

for ep in range(1, epochs+1):
    # ---- training ----
    ltc_model.train()
    tr_loss = 0.0
    for xb, yb in train_loader:  # [B,L,2]
        optimizer.zero_grad()
        B, L, D = xb.shape
        state = xb[:,0,:]        # initial state [B,2]
        preds = []
        for _ in range(horizon):
            inp = state.unsqueeze(1)    # [B,1,2]
            out = ltc_model(inp)            # [B,1,2]
            state = out[:,-1,:]
            preds.append(state)
        preds = torch.stack(preds, dim=1)
        target = yb[:, :horizon, :]
        loss = criterion(preds, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ltc_model.parameters(), 1.0)
        optimizer.step()
        tr_loss += loss.item() * B
    tr_loss /= len(train_loader.dataset)
    train_losses.append(tr_loss)

    # ---- validation ----
    ltc_model.eval()
    va_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            B, L, D = xb.shape
            state = xb[:,0,:]
            preds = []
            for _ in range(horizon):
                inp = state.unsqueeze(1)
                out = ltc_model(inp)
                state = out[:,-1,:]
                preds.append(state)
            preds = torch.stack(preds, dim=1)
            target = yb[:, :horizon, :]
            va_loss += criterion(preds, target).item() * B
    va_loss /= len(val_loader.dataset)
    val_losses.append(va_loss)

    # ---- save best ----
    if va_loss < best_val:
        best_val = va_loss
        torch.save(ltc_model.state_dict(), "best_ltc_mass_spring.pth")

    if ep % 50 == 0:
        print(f"Epoch {ep:4d} | train {tr_loss:.6e} | val {va_loss:.6e}")

# Plot training curve
plt.figure(figsize=(6,4))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Mass–Spring Training vs Validation Loss")
plt.legend(); plt.grid(True); plt.show()


# ================================
# 6. Rollout Test
# ================================
ltc_model.load_state_dict(torch.load("best_ltc_mass_spring.pth"))
ltc_model.eval()

# Ground truth
x0 = torch.tensor([1., 1.])
x_true = odeint(dynamics, x0, t)   # [N+1, 2]

# Normalize initial state
state = ((x0 - mu) / std).unsqueeze(0)  # [1,2]
traj = [state.squeeze().tolist()]

with torch.no_grad():
    for _ in range(N):
        inp = state.unsqueeze(1)        # [1,1,2] for LTC input
        pred = ltc_model(inp)           # [1,1,2]
        state = pred[:, -1, :]          # [1,2]
        traj.append(state.squeeze().tolist())

# Denormalize
traj = torch.tensor(traj) * std + mu    # [N+1, 2]

# Compare
plt.figure(figsize=(7,4))
plt.plot(t.numpy(), x_true[:,0].numpy(), '--', label="ODE x1(t)")
plt.plot(t.numpy(), x_true[:,1].numpy(), '--', label="ODE x2(t)")
plt.plot(t.numpy(), traj[:,0].numpy(), label="LTC x1(t)")
plt.plot(t.numpy(), traj[:,1].numpy(), label="LTC x2(t)")
plt.xlabel("Time [s]")
plt.ylabel("States")
plt.title("Mass–Spring: ODE vs LTC")
plt.legend()
plt.grid(True)
plt.show()


print("✅ Trained & saved best LTC as best_ltc_mass_spring.pth")
