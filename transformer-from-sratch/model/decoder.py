import torch
import torch.nn as nn

from model.FeedForward import FeedForwardNetwork
from model.attention import MultiHeadAttention
from model.embedding import InputEmbedding
from model.encoder import ResidualConnection
from model.positional_encoding import PositionalEncoding


class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8,d_ff=2048, dropout=0.1):
        super().__init__()
        #Initialize masked self-attention
        self.self_attention = MultiHeadAttention(d_model=d_model,num_heads=num_heads)

        # Initialize cross-attention
        self.cross_attention = MultiHeadAttention(d_model=d_model,num_heads=num_heads)
        #Initial feed-forward attention
        self.ffn = FeedForwardNetwork(d_model=d_model,d_ff=d_ff)
        #Initialize residual connections
        self.residual1 = ResidualConnection(d_model=d_model,dropout=dropout)
        self.residual2 = ResidualConnection(d_model=d_model,dropout=dropout)
        self.residual3 = ResidualConnection(d_model=d_model,dropout=dropout)

    def forward(self,y,encoder_output,causal_mask=None,src_mask=None):
        #1. Masked self-attention + residual
        y= self.residual1(y,
                         lambda y: self.self_attention(y,y,y,causal_mask)[0])
        #Cross-attention + residual
        y = self.residual2(y,
                          lambda y: self.cross_attention(

                              y,encoder_output,encoder_output,src_mask)[0])
        #Feed-forward + residual
        y = self.residual3(y,self.ffn)

        return y




class Decoder(nn.Module):
    def __init__(self,vocab_size,d_model=512,num_heads=8,d_ff=2048,num_layers=6,dropout=0.1):
        super().__init__()
        #Embedding
        self.embedding = InputEmbedding(vocab_size,d_model)
        #Positional encoding
        self.pos_encoding = PositionalEncoding(d_model=d_model)
        # Step 3: Dropout
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model,num_heads=num_heads,d_ff=d_ff,dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self,y,encoder_output,causal_mask=None,src_mask=None):
        # Step 1: Embedding target-language tokens
        y = self.embedding(y)
        # Step 2: Add positional encoding
        y = self.pos_encoding(y)
        # Step 3: Apply dropout
        y = self.dropout(y)
        # Step 4: Pass through all decoder layers
        for layer in self.layers:
            y = layer(y, encoder_output, causal_mask, src_mask)

        return y




