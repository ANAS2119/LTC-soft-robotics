🧠 Liquid Neural Networks for Soft Robotics Modeling & Control

This repository presents a data-driven modeling and control framework for a 6-DoF pneumatically actuated soft robot using Liquid Neural Networks (LNNs), including Liquid Time-Constant (LTC) and 
Closed-form Continuous-time (CfC) models.
The learned models are integrated into a Nonlinear Model Predictive Control (nLMPC) framework for two purposes : trajectory tracking and Stabilization .

Project Overview

This project explores Liquid Neural Networks as a compact, continuous-time alternative that:

- Captures system dynamics via ODE-based neural models
- Requires less data and parameters than deep networks
- Generalizes better to unseen excitation signals
- Enables real-time control integration (MPC)

Models Implemented

1. Liquid Time-Constant (LTC)
- Continuous-time RNN governed by ODEs
- Adaptive time constants
- Strong representation of physical dynamics
2. Closed-form Continuous-time (CfC)
- Faster alternative to LTC
- Closed-form update (no ODE solver required)
- Suitable for real-time MPC
3. Comparison Models
- Koopman-based model (baseline)
- LSTM / RNN variants
