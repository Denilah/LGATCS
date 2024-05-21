#!/usr/bin/env python
# encoding: utf-8

from abc import ABC

import torch
import torch.nn as nn
from attention.sublayers import MultiHeadAttention, PositionalWiseFeedForward


class JointEncoderLayer(nn.Module, ABC):
    """
    Encoder layer composed with a multi-heads attention layer
    and a position-wise feed forward network.
    """

    def __init__(self, d_model, d_ffn, n_heads, d_k, d_v, dropout=0.1):
        super(JointEncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionalWiseFeedForward(d_model, d_ffn, dropout=dropout)

    def forward(self, repr1, repr2, padding_mask=None):

        enc_output = self.slf_attn(repr2, repr1, repr2, padding_mask=padding_mask)

        enc_output = self.pos_ffn(enc_output)

        return enc_output

    def code_joint(self, repr1, repr2, repr3, padding_mask=None):
        enc_output = self.slf_attn(repr1, repr2, repr3, padding_mask=padding_mask)

        enc_output = self.pos_ffn(enc_output)

        return enc_output

if __name__ == "__main__":
    input1 = torch.randint(0, 10, (256, 50, 128)).float()
    input2 = torch.randint(0, 10, (256, 6, 128)).float()
    enc = JointEncoderLayer(128, 512, 8, 16, 16)
    outputs = enc(input1, input2)
    print(outputs.size())
