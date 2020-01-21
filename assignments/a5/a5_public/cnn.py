#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, e_char, e_word, k=5):
        super(CNN, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=e_word, kernel_size=k)

    def forward(self, x):
        x_conv = F.relu(self.conv1d(x))
        x_convout = F.max_pool1d(x_conv, kernel_size=x.size()[1:])
        return x_convout

### END YOUR CODE

