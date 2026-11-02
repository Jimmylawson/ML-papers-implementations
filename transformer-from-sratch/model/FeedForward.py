import torch.nn as nn
import torch

class FeedForwardNetwork(nn.Module):
    def __init__(self,d_model=512,d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ff)
        self.linear2 = nn.Linear(d_ff,d_model)
        self.relu = nn.ReLU()

    def forward(self,x):
        return self.linear2(self.relu(self.linear1(x)))