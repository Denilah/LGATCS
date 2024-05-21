#!/usr/bin/env python
# encoding: utf-8


from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module, ABC):
    """
    Multi-heads attention layer.
    """

    def __init__(self, n_heads, d_model, dk, dv, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.dk = dk
        self.dv = dv

        self.w_q = nn.Linear(d_model, n_heads * dk)
        self.w_k = nn.Linear(d_model, n_heads * dk)
        self.w_v = nn.Linear(d_model, n_heads * dv)
        self.fc = nn.Linear(n_heads * dv, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, q, k, v, padding_mask=None):

        n_heads, dk, dv = self.n_heads, self.dk, self.dv
        batch_size, seq_q, seq_k, seq_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = k

        q = self.w_q(q).view(batch_size, seq_q, n_heads, dk)
        k = self.w_k(k).view(batch_size, seq_k, n_heads, dk)
        v = self.w_v(v).view(batch_size, seq_v, n_heads, dv)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1)

        U = torch.randn(self.d_model, batch_size * q.size(3)).view(batch_size, self.n_heads, -1, self.dk)
        U = U.to(self.device)
        attn_weight = torch.matmul(q, U)

        attn_weight = torch.matmul(attn_weight, k.transpose(-1, -2))

        attn_weight = self.dropout(F.softmax(attn_weight, dim=-1))

        context = torch.matmul(attn_weight.transpose(-1,-2), v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_k, -1)

        context = self.dropout(context)

        output = context + residual

        return output


class PositionalWiseFeedForward(nn.Module, ABC):
    """
    Position-wise feed forward network.
    """

    def __init__(self, d_model, d_ffn, dropout=0.1):
        super(PositionalWiseFeedForward, self).__init__()

        self.w_1 = nn.Linear(d_model, d_ffn)
        self.w_2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):

        x = self.layer_norm(x)
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)

        output = x + residual

        return output


if __name__ == "__main__":
    inputs = torch.randint(0, 10, (16, 40, 64)).float()
    print(inputs.size())
    enc = MultiHeadAttention(8, 64, 8, 8)
    outputs = enc(inputs, inputs, inputs)
    print(outputs.size())
    pos = PositionalWiseFeedForward(64, 256)
    output = pos(outputs)
    print(output.size())






