import torch
import torch.nn as nn

# Define the LSTM model class
class ImprovedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(ImprovedLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [B, L, input_size]
        out, _ = self.lstm(x)     # [B, L, hidden]
        out = self.fc(out)        # [B, L, output_size]
        return out                # just return predictions
    
    
"""
# Example initialization
input_size = 9   # [u;x] , u:3x1 , x:6x1
hidden_size = 128
output_size = 6  # predict next x
num_layers = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lstm_model = ImprovedLSTM(input_size, hidden_size, output_size, num_layers).to(device)
"""
