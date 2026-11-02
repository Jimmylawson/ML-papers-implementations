import torch
import torch.nn as nn
import math


class InputEmbedding(nn.Module):
    def __init__(self,vocab_size, d_model =512):
        super().__init__()

        #store the d_model
        self.d_model = d_model
        #Create an embedded layer
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        #Look up token embedd
        #Multiply by sqrt(d_model)
        x = self.embedding(x) * math.sqrt(self.d_model)
        #return the embeddings
        return x
