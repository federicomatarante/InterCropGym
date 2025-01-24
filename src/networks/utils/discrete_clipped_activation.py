import torch
import torch.nn as nn


class DiscreteParameterizedActivation(nn.Module):
    def __init__(self, max_val=1):
        super().__init__()
        self.max_val = max_val

    def forward(self, x):
        # Use sigmoid to get values between 0 and 1, then multiply and round
        return torch.round(self.max_val * torch.sigmoid(x))

    def extra_repr(self):
        return f'max_val={self.max_val}'