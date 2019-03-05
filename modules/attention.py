# -*- coding: utf-8 -*-
# @Time    : 2019/3/1 上午10:56
# @Author  : Xiachong Feng
# @File    : attention.py
# @Software: PyCharm

import torch.nn as nn
import torch.nn.functional as F
import torch


class Attention(nn.Module):

    def __init__(self, method, batch_size, decoder_hidden_size, encoder_total_size, encoder_hidden_size):
        """

        :param method: attention方法
        :param batch_size: batch大小
        :param decoder_hidden_size: decoder的hidden维度
        :param encoder_total_size: 如果是双向的，则是encoder_hidden_size * 2
        :param encoder_hidden_size: encoder的hidden维度

        """
        super(Attention, self).__init__()
        self.method = method
        self.batch_size = batch_size

        if method == "bilinear":
            self.bilinear = nn.Bilinear(decoder_hidden_size, encoder_hidden_size, 1)
        elif method == "bahdanau":
            # for encoder state
            self.key_layer = nn.Linear(encoder_total_size, encoder_hidden_size, bias=False)
            # for decode state
            self.query_layer = nn.Linear(decoder_hidden_size, encoder_hidden_size, bias=False)
            #
            self.energy_layer = nn.Linear(encoder_hidden_size, 1, bias=False)
        else:
            print("Attention method should be bilinear or bahdanau!")

    def forward(self, decoder_hidden, encoder_states, mask):
        """

        :param decoder_hidden: decoder hidden  (1, batch size, decoder hidden size)
        :param encoder_states: encoder hidden  (src_len, batch, encoder total size)
        :param mask: mask for the source input (batch_size, src_len)
        :return:
        """
        if self.method == 'bilinear':
            src_len = encoder_states.size(0)
            enc_energy = self.bilinear(decoder_hidden.expand(src_len, self.batch_size, -1).contiguous(),
                                       encoder_states)  # enc_energy: (src_len, batch, 1)
        elif self.method == 'bahdanau':
            query = self.query_layer(decoder_hidden)
            key = self.key_layer(encoder_states)
            enc_energy = self.energy_layer(torch.tanh(query + key))  # (src_len, batch, 1)

        mask = mask.permute(1, 0).unsqueeze(-1)
        # Mask out invalid positions.
        enc_energy.data.masked_fill_(mask == 0, -float('inf'))
        enc_attn = F.softmax(enc_energy, dim=0).transpose(0, 1)  # (batch, src_len, 1)

        return enc_attn
