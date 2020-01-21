#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    def __init__(self, e_word, bias=True):
        super(Highway, self).__init__()
        self.proj = nn.Linear(e_word, e_word, bias=bias)
        self.gate = nn.Linear(e_word, e_word, bias=bias)
    
    def forward(self, x):
        x_proj = F.relu(self.proj(x))
        x_gate = F.sigmoid(self.gate(x))
        x_highway = x_gate * x_proj + (1. - x_gate) * x
        return x_highway

### END YOUR CODE 

