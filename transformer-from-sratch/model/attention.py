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


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512,num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model
        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model,d_model)

    def forward(self,query,key,value,mask):
        batch_size,seq_len,_ = query.shape
        _, key_len, _ = key.shape
        _, value_len, _ = value.shape
        query = self.W_q(query)
        query = query.reshape(batch_size, seq_len, self.num_heads, self.d_head)
        query = query.transpose(1, 2)
        key = self.W_k(key)
        key = key.reshape(batch_size, key_len, self.num_heads, self.d_head)
        key = key.transpose(1, 2)
        value = self.W_v(value)
        value = value.reshape(batch_size, value_len, self.num_heads, self.d_head)
        value = value.transpose(1, 2)
        output, attention = scaled_dot_product_attention(query, key, value, mask)

        # (B, heads, seq_len, d_head)
        # -> (B, seq_len, heads, d_head)
        output = output.transpose(1, 2)

        # Concatenate heads: (B, seq_len, d_model)
        #contiguous() → Ensures the tensor has a contiguous memory layout.
        output = output.contiguous().reshape(
            batch_size, seq_len, self.d_model
        )

        # Final linear projection
        output = self.W_o(output)

        return output, attention






