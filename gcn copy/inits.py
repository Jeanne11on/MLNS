import torch
import numpy as np

def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = torch.empty(shape).uniform_(-scale, scale)
    return torch.nn.Parameter(initial, name=name)

def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = torch.empty(shape).uniform_(-init_range, init_range)
    return torch.nn.Parameter(initial, name=name)

def zeros(shape, name=None):
    """All zeros."""
    initial = torch.zeros(shape)
    return torch.nn.Parameter(initial, name=name)

def ones(shape, name=None):
    """All ones."""
    initial = torch.ones(shape)
    return torch.nn.Parameter(initial, name=name)