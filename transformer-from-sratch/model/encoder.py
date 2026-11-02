import torch.nn as nn
import torch

from model.FeedForward import FeedForwardNetwork
from model.attention import MultiHeadAttention
from model.embedding import InputEmbedding
from model.positional_encoding import PositionalEncoding


class ResidualConnection(nn.Module):
    def __init__(self,d_model=512,dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)


    def forward(self,x,sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


class EncoderLayer(nn.Module):
    def __init__(self,d_model=512,num_heads=8,d_ff=2048,dropout=0.1):
        super().__init__()

        self.attention = MultiHeadAttention(d_model=d_model,num_heads=num_heads)
        self.ffn = FeedForwardNetwork(d_model=d_model,d_ff=d_ff)
        self.residual1 = ResidualConnection(d_model=d_model,dropout=dropout)
        self.residual2 = ResidualConnection(d_model=d_model,dropout=dropout)


    def forward(self,x,mask=None):
        # 1. Self-attention + residual connection + LayerNorm
        x = self.residual1(x,
            lambda x: self.attention(x, x, x, mask)[0])
        # 2. Feed-forward + residual connection + LayerNorm
        x = self.residual2( x, self.ffn)
        return x


class Encoder(nn.Module):
    def __init__(self,vocab_size,d_model=512,num_heads=8,d_ff=2048,num_layers=6,dropout=0.1):
        super().__init__()
        #Step 1: Input embedding(vocab_size,d_model)
        self.d_model = d_model
        self.embedding = InputEmbedding(vocab_size,d_model)
        #Step 2: Positional encoding(d_model)
        self.pos_encoding = PositionalEncoding(d_model=d_model)
        #Step 3: Dropout
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model,num_heads=num_heads,d_ff=d_ff,dropout=dropout)
            for _ in range(num_layers)
        ])


    def forward(self,x,mask=None):
        #Pass x through all the encoder layers
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x,mask)
        return x
