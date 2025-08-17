import torch.nn as nn

class LSTMRegressor(nn.Module):
    def __init__(self, n_mels, hidden=128, layers=1, bidir=False, dropout=0.2):
        super().__init__()
        self.rnn = nn.LSTM(input_size=n_mels, hidden_size=hidden, num_layers=layers,
                           batch_first=True, bidirectional=bidir, dropout=0.0 if layers==1 else dropout)
        out_dim = hidden * (2 if bidir else 1)
        self.head = nn.Sequential(nn.Linear(out_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64,1))
    def forward(self, x):  # x: (B,T,n_mels)
        out, _ = self.rnn(x)           # (B,T,H or 2H)
        last = out[:, -1, :]           # take last time step
        return self.head(last).squeeze(-1)
