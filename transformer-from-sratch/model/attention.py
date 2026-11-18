import torch
import math
import torch.nn as nn


def scaled_dot_product_attention(Q,K,V,mask=None):
    #attention scores QK.T/ sqrt(d_k)
    d_k = K.shape[-1]
    scores = torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(d_k)
    if mask is not None:
        # Masks mark padding and future tokens with True.
        scores = scores.masked_fill(mask.bool(),float("-inf"))
    attention = torch.softmax(scores,dim=-1)
    output = torch.matmul(attention,V)

    return output, attention


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model:int,num_heads:int,dropout:float):
        super().__init__()

        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model is not divisible by h"


        self.d_model = d_model
        self.W_q = nn.Linear(d_model,d_model) # W_q
        self.W_k = nn.Linear(d_model,d_model) #W_k
        self.W_v = nn.Linear(d_model,d_model) #W_v
        self.W_o = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod # call this class without having an instance of the class
    def attention(query,key,value,mask,dropout:nn.Dropout):
        d_k = query.shape[-1]
        #(Batch,h, Seq_len,d_k ---> (Batch,h,Seq_len,Seq_len)
        attention_scores = (query@key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
           attention_scores =  attention_scores.masked_fill(mask.bool(),float("-inf"))
        attention_scores = attention_scores.softmax(dim=-1) #Batch,Seq_len,h,d_k
        if dropout is not None:
            attention_scores = dropout(attention_scores)


        return (attention_scores @value), attention_scores

    def forward(self,q,k,v,mask):
        batch_size,seq_len,_ = q.shape
        _, key_len, _ = k.shape
        _, value_len, _ = v.shape
        query = self.W_q(q) #(Batch,Seq_len,d_model) -> (Batch,Seq_len,d_model)
        key = self.W_k(k) #(Batch,Seq_len,d_model) -> (Batch,Seq_len,d_model)
        value = self.W_v(v) #(Batch,Seq_len,d_model) -> (Batch,Seq_len,d_model)
        #(Batch,Seq_len,d_model) --> (Batch,Seq_len,h,d_k) --> (Batch, h,Seq_len,d_k)
        #The purpose of view() is to split the embedding dimension into multiple attention heads
        query = query.view(query.shape[0],query.shape[1],self.num_heads,self.d_head).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.num_heads,self.d_head).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.num_heads,self.d_head).transpose(1,2)

        x,self.attention_scores = MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)

        # (Batch, h, seq_len, d_k) -> (B, seq_len, h, d_k) ->
        # Concatenate heads: (B, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.d_model)
        #contiguous() → Ensures the tensor has a contiguous memory layout.
        #(Batch,Seq_len, d_model) --> (Batch,Seq_len,d_model)
        return self.W_o(x)






