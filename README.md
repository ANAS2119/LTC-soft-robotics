**Liquid Neural Networks for Soft Robotics Modeling & Control**

This repository presents a data-driven modeling and control framework for a 6-DoF pneumatically actuated soft robot using Liquid Neural Networks (LNNs), including Liquid Time-Constant (LTC) and 
Closed-form Continuous-time (CfC) models.
The learned models are integrated into a Nonlinear Model Predictive Control (nLMPC) framework for two purposes : trajectory tracking and Stabilization .

**Project Overview**

This project explores Liquid Neural Networks as a compact, continuous-time alternative that:

- Captures system dynamics via ODE-based neural models
- Requires less data and parameters than deep networks
- Generalizes better to unseen excitation signals
- Enables real-time control integration (MPC)

**Models Implemented**

1. Liquid Time-Constant (LTC) , [GitHub Page](https://github.com/KPEKEP/LTCtutorial)
- Continuous-time RNN governed by ODEs
- Adaptive time constants
- Strong representation of physical dynamics
2. Closed-form Continuous-time (CfC) [GitHub Page](https://github.com/KPEKEP/CfCTutorial)
- Faster alternative to LTC
- Closed-form update (no ODE solver required)
- Suitable for real-time MPC
3. Comparison Models
- Koopman-based model (baseline)
- LSTM / RNN variants

**Training Setup**

- Input format: [u_k;x_k]
- Output: x_k+1

Dataset:
Chirp excitation signals (training)
, Sine waves (validation – unseen data)

Normalization:
- scaler_X.pkl → input scaling
- scaler_Y.pkl → output scaling


**Evaluation Metrics**

Model performance is evaluated using Mean Squared Error (MSE).

Validation is performed on:
- Test 1: 1 Hz excitation
- Test 2: 2 Hz excitation
- Different operating modes

**Environment Setup**

Python Requirements:
- Python 3.12.9
- PyTorch 2.5.1
- numpy 2.2.4
- Joblib 1.5.2
- matplotlib 3.10.6

MATLAB Requirements:
- Model Predictive Control Toolbox
- Deep Learning Toolbox
- (Optional) Parallel Computing Toolbox

**MPC Integration**
The trained LNN is embedded into MATLAB nLMPC using:

- StateFcn → LNN state update
- OutputFcn → predicted output

**Future Work**
- Real-time deployment on physical robot
- Hardware-in-the-loop MPC
- Comparison with PINNs / Neural ODEs
- FORCESPRO / CasADi acceleration

**Notes**
- Ensure normalization is consistent between training and MATLAB inference
- Hidden states must be initialized properly before rollout
- Sampling time (Ts) must match training data

**Authors**
- A. Qutah and A. M. Boker, with Bradley Department of
Electrical and Computer Engineering, Virginia Tech, Blacksburg, VA,
USA. 

For questions or collaboration:
-Email: anasaq@vt.edu
