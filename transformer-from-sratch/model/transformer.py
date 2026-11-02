import torch.nn as nn
import torch
from model.encoder import Encoder
from model.decoder import Decoder



class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            tgt_vocab_size,
            d_model=512,
            num_heads=8,
            d_ff=2048,
            num_layers=6,
            dropout=0.1
    ):
        super().__init__()

        self.encoder = Encoder(
            src_vocab_size, d_model, num_heads,
            d_ff, num_layers, dropout
        )

        self.decoder = Decoder(
            tgt_vocab_size, d_model, num_heads,
            d_ff, num_layers, dropout
        )

        self.output_prediction = nn.Linear(
            d_model, tgt_vocab_size
        )

    def forward(self,src_token,tgt_token,src_mask=None,tgt_mask=None):
        encoder = self.encoder(src_token,src_mask)
        decoder = self.decoder(tgt_token,encoder,causal_mask=tgt_mask,src_mask=src_mask)
        output = self.output_prediction(decoder)
        return output # Gives us  raw logits -> a linear transformation L = HW + b

