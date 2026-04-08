# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 12:37:58 2025

@author: anasa
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import joblib
from joblib import load
import torch.nn as nn
from Load_Data import load_soft_robot_dataset
import sys
import os
import numpy
from   ltc_model import RandomWiring, LTCRNN

#Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get directory where inference.py is located
if '__file__' in globals():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    BASE_DIR = os.getcwd()
    
#================================
# Model Prediction
#================================

"""
4 Validation Test:
    - Test1: normal operation , 1 Hz sine wave
    - Test2: normal operation , 2 Hz sine wave
    - Test3: Z operation , 1 Hz sine wave
    - Test4: Z operation , 2 Hz sine wave
"""
#load the scalars

# Scaler paths
scaler_X_path = os.path.join(BASE_DIR, "scaler_X.pkl")
scaler_Y_path = os.path.join(BASE_DIR, "scaler_Y.pkl")

# Load scalers
scaler_X = joblib.load(scaler_X_path)
scaler_Y = joblib.load(scaler_Y_path)

#load validation Test

test_data = {}

# Loop through Test1 → Test4
for i in range(1, 5):
    file_path = os.path.join(BASE_DIR, f"Test{i}.xlsx")

    # Load dataset using your existing function
    X, Y, df = load_soft_robot_dataset(file_path)
       
    # Combine: 3 pressure + 6 pose = 9 total features
    X_full = np.concatenate([X, Y], axis=1)
    
    # Normalized test Data
    X_full_scaled = scaler_X.transform(X_full)   # 9 features

    # Store in dictionary for easy access later
    test_data[f"Test{i}"] = {"X": X, "Y": Y, "df": df, "X_full_scaled": X_full_scaled}

    print(f"Loaded Test{i}: {X.shape[0]} samples, X_full_scaled shape = {X_full_scaled.shape}")

# Create Data:
for i in range(1, 5):
    globals()[f"Data{i}"] = test_data[f"Test{i}"]["X_full_scaled"]
    print(f"Data{i} shape: {globals()[f'Data{i}'].shape}")

## Re-Load the model
input_dim=9
output_dim=6
hidden_dim=8
wiring = RandomWiring(input_dim, output_dim, hidden_dim)
ltc_model = LTCRNN(wiring, input_dim, hidden_dim, output_dim)

# Load LTC weights

state_dict = torch.load("soft_robotics_LTC_3.pth", map_location=device)
ltc_model.load_state_dict(state_dict)
ltc_model.eval()


print("✅ Model loaded successfully and ready for inference!")


def autoregressive_rollout_from_init_pose(
    model, data, device="cuda", scaler_Y=None
):
    """
    Rollout for soft robot starting from *initial 6 pose values only* and given pressures.
    - data: np.array [T, 9] = [P1,P2,P3,pos_x,pos_y,pos_z,roll,pitch,yaw]
    - model: trained LTC model
    - starts from t=0 pose, predicts all future poses autoregressively
    """

    model.to(device)
    model.eval()

    X_full = data.astype(np.float32)
    n_total = len(X_full)
    seq_len = 200
    rollout_horizon = n_total - seq_len  # predict for the remaining steps

    # --- Step 1. Initial state (t=0) ---
    init_pose = X_full[0, 3:]            # 6 pose values
    window_np = X_full[:seq_len].copy()  # take first seq_len pressure samples
    window_np[:, 3:] = init_pose         # constant pose = initial
    window = torch.tensor(window_np, dtype=torch.float32, device=device).unsqueeze(0)

    preds = []
    true_future = X_full[seq_len:seq_len + rollout_horizon, 3:]  # aligned ground truth

            
    with torch.no_grad():
        for t in range(rollout_horizon):
            
            # Forward pass with current window [1, seq_len, 9]
            y_hat = model(window)      # [1, seq_len, 6]
            y_last = y_hat[:, -1, :]              # take last output [1, 6]
            preds.append(y_last.cpu().numpy()[0])

            # Next input = [next pressure, predicted pose]
            if seq_len + t < n_total - 1:
                next_pressure = X_full[seq_len + t, :3]
            else:
                next_pressure = X_full[-1, :3]

            next_input = np.concatenate([next_pressure, y_last.cpu().numpy()[0]], axis=-1)
            next_input = torch.tensor(next_input, dtype=torch.float32, device=device).view(1, 1, -1)

            # Slide the window: drop first step, append new one (keep seq_len constant)
            window = torch.cat([window[:, 1:, :], next_input], dim=1)


    # --- Post-processing ---
    preds = np.stack(preds, axis=0).astype(np.float64)
    true_future = true_future.astype(np.float64)
    
    if scaler_Y is not None:
        preds = scaler_Y.inverse_transform(preds)
        true_future = scaler_Y.inverse_transform(true_future)
    
    # Compute and print MSE
    mse = np.mean((preds - true_future) ** 2)
    print(f"Rollout MSE: {mse:.4e}")

    # --- Plot ---
    labels = ["pos_x","pos_y","pos_z","roll","pitch","yaw"]
    plt.figure(figsize=(10,8))
    for i in range(6):
        plt.subplot(3,2,i+1)
        plt.plot(true_future[:,i],'g-',label="True")
        plt.plot(preds[:,i],'r--',label="Pred")
        plt.title(labels[i]); plt.grid(True)
        if i==0: plt.legend()
    plt.suptitle("Soft Robot Rollout Results")
    plt.tight_layout(); plt.show()
    
    return preds, true_future

preds1, true1 = autoregressive_rollout_from_init_pose(
    model=ltc_model, data=Data1, device="cuda", scaler_Y=scaler_Y)


preds2, true2 = autoregressive_rollout_from_init_pose(
    model=ltc_model, data=Data2, device="cuda", scaler_Y=scaler_Y)

preds3, true3 = autoregressive_rollout_from_init_pose(
    model=ltc_model, data=Data3, device="cuda", scaler_Y=scaler_Y)

preds4, true4 = autoregressive_rollout_from_init_pose(
    model=ltc_model, data=Data4, device="cuda", scaler_Y=scaler_Y)
##############################################################################




