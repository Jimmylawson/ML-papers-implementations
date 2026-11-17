import torch.nn as nn
import torch

class FeedForwardNetwork(nn.Module):
    def __init__(self,d_model:int,d_ff:int,dropout:float):
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ff) #W1 and b1
        self.dropout = dropout
        self.linear2 = nn.Linear(d_ff,d_model) #W2 and b2
        self.relu = nn.ReLU()

    def forward(self,x):
        #(Batch,Seq_len,d_model) ---> (Batch,Seq_len,d_ff) -> (Batch, Seq_len,d_model)
        return self.linear2(self.dropout(self.relu(self.linear1(x))))