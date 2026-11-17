import torch.nn as nn
import torch

from model.FeedForward import FeedForwardNetwork
from model.attention import MultiHeadAttentionBlock
from model.positional_encoding import LayerNormalization
from model.embedding import InputEmbedding
from model.positional_encoding import PositionalEncoding


class ResidualConnection(nn.Module):
    def __init__(self,dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()


    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self,self_attention_block:MultiHeadAttentionBlock,feed_forward_block:FeedForwardNetwork,dropout:float ):
        super().__init__()

        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])


    def forward(self,x,src_mask):
        # 1. Self-attention + residual connection + LayerNorm
        x = self.residual_connections[0](x,lambda x: self.self_attention_block(x,x,x,src_mask))
        # 2. Feed-forward + residual connection + LayerNorm
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm   = LayerNormalization()



    def forward(self,x,mask):
        #Pass x through all the encoder layers
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)
