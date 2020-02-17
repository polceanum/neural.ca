import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class NCA(nn.Module):
    def __init__(self, cell_state_size, hidden_size, num_layers, device):
        super(NCA, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=cell_state_size, 
            out_channels=cell_state_size, 
            kernel_size=3,
            padding=1,
            bias=False)

        self.fcFirst = nn.Linear(cell_state_size, hidden_size)

        nbInterimLayers = num_layers
        self.interimLayers = []
        for i in range(nbInterimLayers):
            self.interimLayers.append(nn.Linear(hidden_size, hidden_size).to(device))
        
        self.fcLast = nn.Linear(hidden_size, cell_state_size, bias=False)
        with torch.no_grad():
            self.fcLast.weight.zero_()

        self.inputDropout = nn.Dropout(p=0.2)
        self.stochasticOutput = nn.Dropout(p=0.5)

    def alive(self, x):
        return (F.max_pool2d(x[:, :, :, 3:4], kernel_size=3, stride=1, padding=1) > 0.1).float()


    def perceive(self, x):
        # apply convolution over input (transposed version, since PyTorch requires C,H,W instead of H,W,C)
        percepts = self.conv(x.transpose(-3, -1)).transpose(-3, -1).contiguous()

        # only keep alive cells
        # percepts = percepts * self.alive(x)
        
        return percepts

    def update(self, x):
        ds = self.fcFirst(x)
        ds = F.relu(ds)
        for i in range(len(self.interimLayers)):
            ds = self.interimLayers[i](ds)
            ds = F.relu(ds)
        ds = self.fcLast(ds)
        ds = self.stochasticOutput(ds) # stochastic update
        return ds # deltas

    def forward(self, x, steps):
        # keep size for future use
        normal_size = x.size()
        # initialize history list
        history = [x.detach().cpu()]

        for step in range(steps):
            # get perception for each pixel, from damaged previous state
            percepts = self.perceive(self.inputDropout(x))
            # speed optimization by considering each pixel separately by flattening
            out = self.update(percepts.flatten(start_dim=0, end_dim=-2))
            # add network output to state
            x = x + out.view(normal_size)
            # save history
            history.append(x.detach().cpu())

        return x, history
