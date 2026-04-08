# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 12:37:58 2025

@author: anasa
"""
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
from joblib import load
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import sys
import os

#Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Path
model_path = "/home/anasaq/LTC/Modeling/PyTorch"

# Add that folder to Python's module search path
if model_path not in sys.path:
    sys.path.append(model_path)

# Import Model now import normally, as if the file was in the same folder
from   ltc_model import RandomWiring, LTCRNN
#from   ltc_2_model import RandomWiring, LTCRNN

from CFC_model import CFCSolutionCell, CFCImprovedCell, CFCRNN

#

#========================
#load dataset
#========================

def load_soft_robot_dataset(
        file_path,
        to_tensor=True,
        plot_trajectories=True,
        traj_cols=("pos_x", "pos_y", "pos_z"),
        max_points=16000
    ):
    """
    Load soft-robot dataset (no header) and optionally plot some trajectories.

    Parameters
    ----------
    file_path : str
        Path to .csv or .xlsx file without header.
    to_tensor : bool
        If True, returns PyTorch tensors.
    plot_trajectories : bool
        If True, plot selected output columns.
    traj_cols : tuple of str
        Which output columns to plot.
    max_points : int
        Plot at most this many samples.

    Returns
    -------
    X, Y, df
    """
    # Define headers manually
    headers = [
        "P1", "P2", "P3",
        "pos_x", "pos_y", "pos_z",
        "rot_roll", "rot_pitch", "rot_yaw"
    ]

    # --- load file with no header ---
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path, header=None, names=headers)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path, header=None, names=headers)
    else:
        raise ValueError("File must be .csv or .xlsx")

    # Split input/output
    input_cols = ["P1", "P2", "P3"]
    output_cols = ["pos_x", "pos_y", "pos_z", "rot_roll", "rot_pitch", "rot_yaw"]

    X = df[input_cols].to_numpy(dtype=np.float32)
    Y = df[output_cols].to_numpy(dtype=np.float32)

    # ---  plot ---
    if plot_trajectories:
        n = min(len(df), max_points)
        t = np.arange(n)

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # Positions
        axes[0].plot(t, df["pos_x"][:n], label="pos_x")
        axes[0].plot(t, df["pos_y"][:n], label="pos_y")
        axes[0].plot(t, df["pos_z"][:n], label="pos_z")
        axes[0].set_ylabel("Position (cm)")
        axes[0].set_title("Position trajectories (x, y, z)")
        axes[0].legend()
        axes[0].grid(True)

        # Rotations
        axes[1].plot(t, df["rot_roll"][:n], label="rot_roll")
        axes[1].plot(t, df["rot_pitch"][:n], label="rot_pitch")
        axes[1].plot(t, df["rot_yaw"][:n], label="rot_yaw")
        axes[1].set_xlabel("Sample index / Time")
        axes[1].set_ylabel("Rotation (rad)")
        axes[1].set_title("Rotation trajectories (roll, pitch, yaw)")
        axes[1].legend()
        axes[1].grid(True)
        
        # Pressures
        axes[2].plot(t, df["P1"][:n], label="P1")
        axes[2].plot(t, df["P2"][:n], label="P2")
        axes[2].plot(t, df["P3"][:n], label="P3")
        axes[2].set_xlabel("Sample index / Time")
        axes[2].set_ylabel("Pressure")
        axes[2].set_title("Input pressures (P1, P2, P3)")
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()
    # --- convert to tensor ---
    if to_tensor:
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)

    return X, Y, df

file_path = "/home/anasaq/soft_robotics/Dataset/Dataset.xlsx" 

X, Y, df = load_soft_robot_dataset(file_path)

# Combine: 3 pressure + 6 pose = 9 total input features
X_full = np.concatenate([X, Y], axis=1)

#=======================
# Data preprocessing
#=======================

def prepare_soft_robot_dataloaders(
    X, Y,
    seq_len=600,        # sequence length (samples of history)
    batch_size=64,
    seed=42
):
    """
    Prepare DataLoaders for one-step-ahead prediction (Option B).

    Each training example:
        X_seq[i] -> past `seq_len` samples
        Y_seq[i] -> next sample after that window

    Steps:
      1. Normalize inputs/outputs (StandardScaler)
      2. Build sequential windows (stride = 1)
      3. Split chronologically 80/20
      4. Return PyTorch DataLoaders + fitted scalers
    """

    # ---- convert to numpy if tensors ----
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.numpy()

    # ---- normalization ----
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    Xs = scaler_X.fit_transform(X)
    Ys = scaler_Y.fit_transform(Y)

    print("After normalization:")
    print(" X mean:", np.mean(Xs, axis=0))
    print(" X std :", np.std(Xs, axis=0))
    print(" Y mean:", np.mean(Ys, axis=0))
    print(" Y std :", np.std(Ys, axis=0))

    # ---- create sequences ----
    N = len(Xs)
    n_seq = N - seq_len - 1   # minus one for next-step target

    X_seq = np.zeros((n_seq, seq_len, Xs.shape[1]), dtype=np.float32)
    Y_seq = np.zeros((n_seq, Ys.shape[1]), dtype=np.float32)

    for i in range(n_seq):
        X_seq[i] = Xs[i:i+seq_len]       # window of past samples
        Y_seq[i] = Ys[i+seq_len]         # next sample after window

    # ---- chronological 80/20 split ----
    split_idx = int(0.8 * n_seq)
    X_train, Y_train = X_seq[:split_idx], Y_seq[:split_idx]
    X_val,   Y_val   = X_seq[split_idx:], Y_seq[split_idx:]

    # ---- tensors & DataLoaders ----
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    X_val   = torch.tensor(X_val,   dtype=torch.float32)
    Y_val   = torch.tensor(Y_val,   dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=batch_size,
        shuffle=False,
        num_workers=15,
        pin_memory=True,
    )

    val_loader = DataLoader(
        TensorDataset(X_val, Y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=15,
        pin_memory=True,
    )

    # ---- summary ----
    print(f"\nSequence length: {seq_len} samples")
    print(f"Total windows: {n_seq}")
    print(f"Train windows: {len(X_train)}")
    print(f"Val windows:   {len(X_val)}")
    print(f"Input shape: {X_train.shape}, Target shape: {Y_train.shape}")

    return train_loader, val_loader, scaler_X, scaler_Y

train_loader, val_loader, scaler_X, scaler_Y = prepare_soft_robot_dataloaders(
    X_full, Y,
    seq_len=800,
    batch_size=32
)

#Save scalers for inference
joblib.dump(scaler_X, "/home/anasaq/soft_robotics/scaler_X.pkl")
joblib.dump(scaler_Y, "/home/anasaq/soft_robotics/scaler_Y.pkl")

#=======================================
#Pytorch-Lightning RNN training module
#=======================================

#LTC
class SequenceLearner(pl.LightningModule):
    def __init__(self, model, lr=0.005):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        x, y = batch # x:[B, seq_len, in_dim], y:[B, out_dim]
        x = x.permute(1, 0, 2)  # [seq_len, B, input_dim]
        y_hat = self.model(x) # -> [seq_len, B, out_dim]
        y_pred = y_hat[-1]  # last output only [B, out_dim]
        loss = self.criterion(y_pred, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.permute(1, 0, 2)
        y_hat = self.model(x)
        y_pred = y_hat[-1]
        loss = self.criterion(y_pred, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss


    def configure_optimizers(self):
        optimizer=torch.optim.Adam(self.model.parameters(), lr=self.lr)   
        return optimizer
"""
#LTC mixed training
class SequenceLearner_LTC_Rollout(pl.LightningModule):
    def __init__(self, model, lr=5e-4, rollout_steps=5, rollout_weight=0.2):
        super().__init__()
        self.model = model
        self.lr = lr
        
        self.rollout_steps = rollout_steps        # multi-step horizon
        self.rollout_weight = rollout_weight      # weight of rollout loss
        self.criterion = nn.MSELoss()

    def one_step_forward(self, x):
        # x : [B, seq_len, input_dim] 
        x_LTC = x.permute(1,0,2)     # LTC expects [T, B, F]
        y_pred = self.model(x_LTC)
        return y_pred[-1]            # last step only

    def multi_step_rollout(self, x_init):
        
        x_init: [B, seq_len, input_dim]
        Perform autoregressive prediction for rollout_steps.
        
        B, L, F = x_init.shape
        window = x_init.clone().to(self.device)
        preds = []

        for k in range(self.rollout_steps):
            y = self.one_step_forward(window)          # [B, output_dim]
            preds.append(y)

            # next input = concat[pressures_next, predicted pose]
            next_press = x_init[:, k , :3]            # use last known pressure or shift if you want
            next_input = torch.cat([next_press, y], dim=-1).unsqueeze(1)

            # sliding window
            window = torch.cat([window[:,1:,:], next_input], dim=1)

        return torch.stack(preds, dim=1)  # [B, rollout_steps, out_dim]

    def training_step(self, batch, batch_idx):
        x, y = batch  # x: [B, seq_len, 9], y: [B, 6]
        
        # one-step prediction loss
        y_pred = self.one_step_forward(x)
        loss_1 = self.criterion(y_pred, y)

        # rollout loss (every two batches)
        if batch_idx % 2 == 0:    
            rollout_preds = self.multi_step_rollout(x)
            rollout_true = y.unsqueeze(1).repeat(1, self.rollout_steps, 1)
            loss_roll = self.criterion(rollout_preds, rollout_true)
            self.log("train_rollout_loss", loss_roll, on_step=True, on_epoch=False)
            
            #print(f"[batch{batch_idx}] rollout loss = {loss_roll.item():.6f}")
        else:
            loss_roll=torch.tensor(0.0, device=self.device)   

        # combine losses
        loss = loss_1 + self.rollout_weight * loss_roll

        self.log("train_loss", loss, on_step=True, prog_bar=True)
        self.log("train_one_step_loss", loss_1, on_step=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.one_step_forward(x)
        loss = self.criterion(y_pred, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


#CFC
class SequenceLearner(pl.LightningModule):
    def __init__(self, model, lr=0.005, dt=0.05):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.dt=dt

    def training_step(self, batch, batch_idx):
        x, y = batch # x:[B, seq_len, in_dim], y:[B, out_dim]
        batch_size, seq_len, _ = x.shape
        device = x.device
        # build timespan tensor [B, seq_len]
        timespans = torch.full((batch_size, seq_len), self.dt, device=device)
        
        # forward through model
        y_hat = self.model(x, timespans)   # -> [B, seq_len, out_dim]
        y_pred = y_hat[:, -1, :]           # last output only [B, out_dim]
        
        loss = self.criterion(y_pred, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
       x, y = batch # x:[B, seq_len, in_dim], y:[B, out_dim]
       batch_size, seq_len, _ = x.shape
       device = x.device
       # build timespan tensor [B, seq_len]
       timespans = torch.full((batch_size, seq_len), self.dt, device=device)
       
       # forward through model
       y_hat = self.model(x, timespans)   # -> [B, seq_len, out_dim]
       y_pred = y_hat[:, -1, :]           # last output only [B, out_dim]
       
       loss = self.criterion(y_pred, y)
       self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
       return loss


    def configure_optimizers(self):
        optimizer=torch.optim.Adam(self.model.parameters(), lr=self.lr)   
        return optimizer
"""
#==================
# LTC Model
#=================

# fixed tau
#tau=1;
# Instantiate model
input_dim=9
output_dim=6
hidden_dim=8
wiring = RandomWiring(input_dim, output_dim, hidden_dim)
ltc_model = LTCRNN(wiring, input_dim, hidden_dim, output_dim)

"""
#==================
# CFC Model
#=================

#Model Initialization

cell = CFCImprovedCell(input_dim=9, hidden_dim=64)
CFC_model = CFCRNN(cell, input_dim=9, hidden_dim=64, output_dim=6)
"""
#==================
# Model Training
#=================
# Lightning wrapper

#LTC
learn = SequenceLearner(ltc_model, lr=4e-5)

#CFC
#learn = SequenceLearner(CFC_model, lr=1e-5, dt=0.05)

#log root
log_root = "./soft_rob_logs"  
os.makedirs(log_root, exist_ok=True)

# Callbacks

#LTC
checkpoint_cb = ModelCheckpoint(
    dirpath=os.path.join(log_root, "checkpoints"),
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    filename="ltc-{epoch:02d}-{val_loss:.4f}"
)
early_stop_cb = EarlyStopping(monitor="val_loss", patience=20, mode="min")

"""
#CFC
checkpoint_cb = ModelCheckpoint(
    dirpath=os.path.join(log_root, "checkpoints"),
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    filename="cfc-{epoch:02d}-{val_loss:.4f}"
)
early_stop_cb = EarlyStopping(monitor="val_loss", patience=10, mode="min")
"""
# Create a TensorBoard logger

#LTC
tb_logger = TensorBoardLogger(
    save_dir=log_root,
    name="tb_logs/soft_robotics_ltc"
)

csv_logger = CSVLogger(
    save_dir=log_root,
    name="csv_logs/soft_robotics_ltc"
)

"""
#CFC
tb_logger = TensorBoardLogger(
    save_dir=log_root,
    name="tb_logs/soft_robotics_CFC"
)

csv_logger = CSVLogger(
    save_dir=log_root,
    name="csv_logs/soft_robotics_CFC"
)

"""
trainer = pl.Trainer(
    default_root_dir=log_root,
    logger=[tb_logger,csv_logger],
    max_epochs=4000,
    gradient_clip_val=1.0,     # stabilizes LTC training
    accelerator="gpu",
    enable_progress_bar=True,
    precision="16-mixed",
    devices=1,
    callbacks=[checkpoint_cb, early_stop_cb],
)


#To resume training

trainer.fit(
    learn,
    train_loader,
    val_loader,
    ckpt_path="/home/anasaq/soft_rob_logs/checkpoints/ltc-epoch=2346-val_loss=0.0283.ckpt"
)


trainer.fit(learn, train_loader, val_loader)

#print tau
#print("Learned taus:", ltc_model.taus.data)

# save training completes

#LTC
torch.save(learn.model.state_dict(), "soft_robotics_LTC_4.pth")

#CFC
#torch.save(learn.model.state_dict(), "soft_robotics_CFC_model.pth")



