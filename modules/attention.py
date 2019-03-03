# -*- coding: utf-8 -*-
# @Time    : 2019/3/1 上午10:56
# @Author  : Xiachong Feng
# @File    : attention.py
# @Software: PyCharm

import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self, method, batch_size, decoder_hidden_size, encoder_hidden_size):
        """

        :param method: attention方法
        :param batch_size: batch大小
        :param decoder_hidden_size: decoder的hidden维度
        :param encoder_hidden_size: encoder的hidden维度
        """
        super(Attention, self).__init__()
        self.method = method
        self.batch_size = batch_size

        if method == "bilinear":
            self.bilinear = nn.Bilinear(decoder_hidden_size, encoder_hidden_size, 1)
        # elif method == "bahdanau":

    def forward(self, decoder_hidden, encoder_states):
        """

        :param decoder_hidden: decoder hidden  (1, batch size, hidden size)
        :param encoder_states: encoder 编码状态 (src_len, batch, hidden)
        :return:
        """
        if self.method == 'bilinear':
            src_len = encoder_states.size(0)
            enc_energy = self.bilinear(decoder_hidden.expand(src_len, self.batch_size, -1).contiguous(),
                                       encoder_states)  # enc_energy: (src_len, batch, 1)
        # elif self.method == 'bahdanau':

        enc_attn = F.softmax(enc_energy, dim=0).transpose(0, 1)  # (batch, src_len, 1)

        return enc_attn
