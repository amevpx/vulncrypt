import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class VulnCryptModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0):
        super(VulnCryptModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if x.size(1) == 0:  # Check for empty sequences
            raise ValueError("Input tensor contains empty sequences.")
        
        embeds = self.embedding(x)  # Shape: (batch_size, seq_length, embedding_dim)
        lstm_out, _ = self.lstm(embeds)  # Shape: (batch_size, seq_length, hidden_dim)
        out = self.fc(lstm_out[:, -1, :])  # Last LSTM output
        return torch.sigmoid(out)

    def forward(self, x):
        """
        Forward pass of the model.
        """
        # Embedding Layer
        embeds = self.embedding(x)
        
        # LSTM Layer
        lstm_out, _ = self.lstm(embeds)
        
        # Take the output from the last LSTM cell
        final_output = lstm_out[:, -1, :]
        
        # Fully Connected Layer
        output = self.fc(final_output)
        
        # Sigmoid Activation
        return self.sigmoid(output)
