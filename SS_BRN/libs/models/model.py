#!/usr/bin/python2.7

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
from loguru import logger

from typing import Any, Optional, Tuple


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out

class ActionSegmentRefinementFramework(nn.Module):

    def __init__(
        self,
        in_channel: int,
        n_features: int,
        n_layers: int,
        **kwargs: Any
    ) -> None:

        super().__init__()
        self.conv_in = nn.Conv1d(in_channel, n_features, 1)
        multi_dilated_layers = [
            DilatedResidualLayer(2 ** i, n_features, n_features)
            for i in range(n_layers)
        ]
        self.multi_dilated_layers = nn.ModuleList(multi_dilated_layers)
        self.conv_bound = nn.Conv1d(n_features, 1, 1)

        self.activation_brb = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.conv_in(x)
        for layer in self.multi_dilated_layers:
            out = layer(out)

        out_bound = self.conv_bound(out)

        if self.training:
            outputs_bound = [out_bound]

            outputs_bound.append(out_bound)

            return outputs_bound
        else:
            return out_bound
