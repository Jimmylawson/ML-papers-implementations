import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=5000):
        super().__init__()

        # Create positional encoding matrix
        self.embedding = nn.Embedding(max_len,d_model)
        # Calculate sine and cosine values
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0,max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position *div_term)
        # Register the positional encodings as a buffer
        self.register_buffer("p_e",pe)

    def forward(self, x):
        # Add positional encodings to token embeddings
        seq_len = x.shape[1]
        x = x + self.p_e[:seq_len]
        return x
