import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianActivation(nn.Module):
    def __init__(self, mu=0, sigma=1, min=-1, max=1):
        super(GaussianActivation, self).__init__()
        self.mu = mu
        self.sigma = sigma
        self.min = min
        self.max = max

    def forward(self, x):
        # max = torch.full(x.shape, self.max).to(torch.device('cuda:0'))
        # min = torch.full(x.shape, self.min).to(torch.device('cuda:0'))
        # return torch.where(x > max or x < min, torch.tensor(0.0), torch.exp(-0.5 * ((x - self.mu) / self.sigma)**2))
        return torch.exp(-0.5 * ((x - self.mu) / self.sigma)**2)