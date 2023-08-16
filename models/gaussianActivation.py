import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianActivation(nn.Module):
    def __init__(self, mu=0, sigma=1):
        super(GaussianActivation, self).__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        # truncation bounds
        return torch.exp(-0.5 * ((x - self.mu) / self.sigma)**2)