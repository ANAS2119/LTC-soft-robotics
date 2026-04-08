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
from Load_Data import load_soft_robot_dataset

#Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
# Model Path
model_path = "/.........."

# Add that folder to Python's module search path
if model_path not in sys.path:
    sys.path.append(model_path)

"""
from LSTM import ImprovedLSTM
#

#========================
#load dataset
#========================

file_path = r"C:\Users\........\Dataset.xlsx"

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
    print(" X mean:", np.mean(Xs,  axis=0))
    print(" X std :", np.std(Xs,axis=0))
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
        persistent_workers=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        TensorDataset(X_val, Y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=15,
        persistent_workers=True,
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
joblib.dump(scaler_X)
joblib.dump(scaler_Y)

#=======================================
#Pytorch-Lightning RNN training module
#=======================================

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
#==================
# LSTM Model
#=================

# Instantiate model
input_dim=9
output_dim=6
hidden_dim=128
num_layers = 3
LSTM_model =ImprovedLSTM(input_dim, hidden_dim, output_dim, num_layers)   

#==================
# Model Training
#=================
# Lightning wrapper

#LTC
learn = SequenceLearner(LSTM_model, lr=4e-5)

#CFC
#learn = SequenceLearner(CFC_model, lr=1e-5, dt=0.05)

#log root
log_root = "./LSTM_soft_rob_logs"  
os.makedirs(log_root, exist_ok=True)

# Callbacks

checkpoint_cb = ModelCheckpoint(
    dirpath=os.path.join(log_root, "checkpoints"),
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    filename="LSTM-{epoch:02d}-{val_loss:.4f}"
)
early_stop_cb = EarlyStopping(monitor="val_loss", patience=40, mode="min")


# Create a TensorBoard logger

tb_logger = TensorBoardLogger(
    save_dir=log_root,
    name="tb_logs/soft_robotics_LSTM"
)

csv_logger = CSVLogger(
    save_dir=log_root,
    name="csv_logs/soft_robotics_LSTM"
)


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
    ckpt_path=r"C:\Users.....\LSTM-epoch=276-val_loss=0.0014.ckpt"
)

trainer.fit(learn, train_loader, val_loader)


# save training completes


torch.save(learn.model.state_dict(), "soft_robotics_LSTM.pth")


#########################END###################################################


