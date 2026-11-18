import torch
import torch.nn as nn

from model.FeedForward import FeedForwardNetwork
from model.attention import MultiHeadAttentionBlock
from model.embedding import InputEmbedding
from model.encoder import ResidualConnection
from model.positional_encoding import PositionalEncoding, LayerNormalization


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttentionBlock,cross_attention_block:MultiHeadAttentionBlock, feed_forward_block:FeedForwardNetwork,dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x,lambda  x:self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x = self.residual_connections[2](x,self.feed_forward_block)

        return x




class Decoder(nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers =  layers
        self.norm = LayerNormalization()

    def forward(self,x,encoder_output,src_mask,tgt_mask):

        for layer in self.layers:
            x = layer(x,encoder_output,src_mask,tgt_mask) #calling the forward of the Decoder block
        return  self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.proj = nn.Linear(d_model,vocab_size)

    def forward(self,x):
        #(Batch,Seq_len,d_model) --> (Batch,Seq_len,Vocab_Size)
        return self.proj(x)






