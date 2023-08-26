import torch
import torch.nn as nn

class GaussianActivation(nn.Module):
    def __init__(self, mu=0, sigma=1, min=-1, max=1):
        super(GaussianActivation, self).__init__()
        self.mu = mu
        self.sigma = sigma
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.exp(-0.5 * ((x - self.mu) / self.sigma)**2)