from typing import Any

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, seq_len=500,dropout:float =0.1):
        super().__init__()

        # Create positional encoding matrix

        self.dropout = nn.Dropout(dropout)
        # Calculate sine and cosine values
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position *div_term)

        pe = pe.unsqueeze(0) #(1,seq_Len,d_model)
        # Register the positional encodings as a buffer
        #register_buffer will keep them out of the optimization trainable parameters
        #IT will move them to the GPU when you move the model
        self.register_buffer("p_e",pe)

    def forward(self, x):
        # Add positional encodings to token embeddings
        x = x + self.p_e[:, :x.shape[1], :]
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self,eps:float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Added

    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True)
        var = x.var(dim=-1, keepdim=True)

        return self.alpha  * (x - mean) / torch.sqrt(var + self.eps) +  self.bias
