"""
CFC_model.py - This module defines the CFCModel class obtained from raminmh github repository.Linked below:
https://github.com/raminmh/CfC/blob/main/torch_cfc.py
"""
# %%  <-- import library cell
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

# %%  <-- CFC Model Definition

# Define the CFC Solution Cell class
class CFCSolutionCell(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=4, activation_type=nn.Tanh, num_backbone_neurons=4, num_backbone_layers=1, dropout=None):
        super(CFCSolutionCell, self).__init__()

        # Set the parameters for the CFC Solution Cell
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation_type = activation_type
        self.num_backbone_neurons = num_backbone_neurons

        # Define extra layers for the backbone network
        extra_layers = [
            nn.Linear(num_backbone_neurons, num_backbone_neurons),
            self.activation_type(),
            nn.Dropout(dropout) if dropout is not None else nn.Identity()
        ] * (num_backbone_layers - 1)

        # Define the backbone network
        self.backbone = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, num_backbone_neurons),
            self.activation_type(),
            *extra_layers
        )

        # Define the linear layer for f
        self.f = nn.Linear(num_backbone_neurons, hidden_dim)

        # Define parameters for w_tau and A
        self.w_tau = torch.nn.Parameter(data=torch.zeros(1, self.hidden_dim), requires_grad=True)
        self.A = torch.nn.Parameter(data=torch.ones(1, self.hidden_dim), requires_grad=True)

    def forward(self, inputs, hx, ts):
        batch_size = inputs.size(0)  # Get batch size
        ts = ts.view(batch_size, 1)  # Reshape the timespan to match batch size
        x = torch.cat([inputs, hx], 1)  # Concatenate inputs and hidden state

        x = self.backbone(x)  # Pass through the backbone network
        
        # Calculate the value of f
        f_value = self.f(x)
        
        # Compute the new hidden state using the solution formula 2 from the article
        new_hidden = (-self.A * torch.exp(-ts * (torch.abs(self.w_tau) + torch.abs(f_value))) * f_value + self.A)

        return new_hidden
    def __init__(self, input_dim=2, hidden_dim=4, activation_type=nn.Tanh, num_backbone_neurons=4, num_backbone_layers=1, dropout=None):
        super(CFCSolutionCell, self).__init__()

        # Set the parameters for the CFC Solution Cell
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation_type = activation_type
        self.num_backbone_neurons = num_backbone_neurons

        # Define extra layers for the backbone network
        extra_layers = [
            nn.Linear(num_backbone_neurons, num_backbone_neurons),
            self.activation_type(),
            nn.Dropout(dropout) if dropout is not None else nn.Identity()
        ] * (num_backbone_layers - 1)

        # Define the backbone network
        self.backbone = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, num_backbone_neurons),
            self.activation_type(),
            *extra_layers
        )

        # Define the linear layer for f
        self.f = nn.Linear(num_backbone_neurons, hidden_dim)

        # Define parameters for w_tau and A
        self.w_tau = torch.nn.Parameter(data=torch.zeros(1, self.hidden_dim), requires_grad=True)
        self.A = torch.nn.Parameter(data=torch.ones(1, self.hidden_dim), requires_grad=True)

    def forward(self, inputs, hx, ts):
        batch_size = inputs.size(0)  # Get batch size
        ts = ts.view(batch_size, 1)  # Reshape the timespan to match batch size
        x = torch.cat([inputs, hx], 1)  # Concatenate inputs and hidden state

        x = self.backbone(x)  # Pass through the backbone network
        
        # Calculate the value of f
        f_value = self.f(x)
        
        # Compute the new hidden state using the solution formula 2 from the article
        new_hidden = (-self.A * torch.exp(-ts * (torch.abs(self.w_tau) + torch.abs(f_value))) * f_value + self.A)

        return new_hidden
    

# Define the CFC Improved Cell class
class CFCImprovedCell(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=4, activation_type=nn.Tanh, num_backbone_neurons=4, num_backbone_layers=1, dropout=None):
        super(CFCImprovedCell, self).__init__()

        # Set the parameters for the CFC Improved Cell
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation_type = activation_type
        self.num_backbone_neurons = num_backbone_neurons

        # Define extra layers for the backbone network
        extra_layers = [
            nn.Linear(num_backbone_neurons, num_backbone_neurons),
            self.activation_type(),
            nn.Dropout(dropout) if dropout is not None else nn.Identity()
        ] * (num_backbone_layers - 1)

        # Define the backbone network
        self.backbone = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, num_backbone_neurons),
            self.activation_type(),
            *extra_layers
        )

        # Define activation functions and linear layers for the improved cell
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.head_f = nn.Linear(num_backbone_neurons, hidden_dim)
        self.head_g = nn.Linear(num_backbone_neurons, hidden_dim)
        self.head_h = nn.Linear(num_backbone_neurons, hidden_dim)
        self.w_tau_approx = nn.Linear(num_backbone_neurons, hidden_dim)

    def forward(self, inputs, hx, ts):
        batch_size = inputs.size(0)  # Get batch size
        ts = ts.view(batch_size, 1)  # Reshape the timespan to match batch size
        x = torch.cat([inputs, hx], 1)  # Concatenate inputs and hidden state

        x = self.backbone(x)  # Pass through the backbone network
        
        # Compute the values for h, g, and f using the head layers
        head_h = self.tanh(self.head_h(x))
        head_g = self.tanh(self.head_g(x))
        head_f = self.head_f(x)

        # Compute the approximation for w_tau
        w_tau_approx = self.w_tau_approx(x)
        
        # Compute sigma using the sigmoid function
        # Note: this place is different from the reference implementation:
        # instead of just adding w_tau_approx, we multiply it by ts as well, as in original formula.
        # I have asked authors for comments on that matter. Will update here.
        sigma = self.sigmoid((w_tau_approx + head_f)* ts)
        
        # Compute the new hidden state using the improved solution formula
        new_hidden = head_h * (1.0 - sigma) + sigma * head_g

        return new_hidden
    
    # Define the CFC RNN class
class CFCRNN(nn.Module):
    def __init__(self, cfc_cell, input_dim, hidden_dim, output_dim):
        super(CFCRNN, self).__init__()
        self.cell = cfc_cell  # Set the CFC cell
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.out_proj = nn.Linear(hidden_dim, output_dim)  # Define the output projection layer

    def forward(self, inputs, timespans):
        batch_size, seq_len, _ = inputs.size()  # Get batch size and sequence length from input dimensions
        
        device = inputs.device

        states = torch.zeros(batch_size, self.hidden_dim, device=device)  # Initialize hidden states with zeros
        outputs = []  # List to store outputs for each time step

        for t in range(seq_len):
            # Compute output and next state for each time step
            states = self.cell(inputs[:, t, :], states, timespans[:, t])
            outputs.append(self.out_proj(states))  # Append the output to the list

        result = torch.stack(outputs, dim=1)  # Stack the outputs along the sequence dimension
        return result