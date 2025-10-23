# -*- coding: utf-8 -*-
"""
Impoved LTC from https://github.com/ShilongZong/LNN-code/blob/master/LNN_RNN_training9.ipynb

@author: anasa
"""
import torch
import torch.nn as nn

class LiquidCell(nn.Module):
    """
    Liquid Cells with shared parameters: All layers share the same ODE function.
    The input and hidden states are processed by a concatenated single linear
    layer, reducing the number of parameters by 50%.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.combined_linear = nn.Linear(hidden_size*2, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, t, hidden):
        x, h = hidden
        
        #After splicing, the single linear layer replaces the original
        #two linear layers
        combined = torch.cat([x, h], dim=-1)
        dhdt = self.tanh(self.combined_linear(combined))
        return dhdt
class MultiScaleLiquidRNN(nn.Module):
    """
    Improved Multi-Scale Liquid Networks:
        1. Added an input uniform conversion layer to ensure consistent input dimensions across all layers
        2. Shared ODE function parameters ensure no increase in parameters as the number of layers increases
        3. Dynamic time scaling coefficients enhance time series modeling
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dt=0.05,
                 time_span=1.0, num_time_steps=20, learn_tau=True, tau=1.0): # Add tau parameter
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.time_span = time_span
        self.num_time_steps = num_time_steps
        self.dt = dt

        self.cell = LiquidCell(hidden_size)

        # Use tau as a fixed time scale
        if learn_tau:
            self.taus = nn.Parameter(torch.tensor([tau] * num_layers, dtype=torch.float32))
        else:
            self.register_buffer('taus', torch.tensor([tau] * num_layers, dtype=torch.float32))

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
    def forward(self, x, h_0=None):
        seq_len, batch_size, _ = x.size()
        x = self.input_proj(x)  # Unified input projection

        # Initialize hidden state
        device = x.device
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers, batch_size,
                             self.hidden_size, device=device)
        hidden = h_0


        outputs = []
        for t_step in range(seq_len):
            input_t = x[t_step]
            new_hidden = []

            for layer in range(self.num_layers):
                tau = self.taus[layer] + 1e-3  # avoid div-by-zero
                h = hidden[layer]

                # Euler-style update
                dhdt = self.cell(input_t.detach(), h)
                h_next = h + (self.dt / tau) * dhdt

                new_hidden.append(h_next)
                input_t = h_next

            hidden = torch.stack(new_hidden)
            outputs.append(self.output_layer(hidden[-1]))

        return torch.stack(outputs), hidden
