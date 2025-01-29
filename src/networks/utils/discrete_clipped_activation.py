import functorch.dim
import torch
import torch.nn as nn


class DiscreteParameterizedActivation(nn.Module):
    def __init__(self, max_val=1):
        super().__init__()
        self.max_val = max_val

    def forward(self, x):
        # Instead of sigmoid lets use softmax and sample
        probs = nn.functional.softmax(x, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)

    def extra_repr(self):
        return f'max_val={self.max_val}'