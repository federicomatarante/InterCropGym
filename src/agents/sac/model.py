import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()
        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_actions)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def forward(self, state, action):
        x1 = F.relu(self.linear1(state))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(state))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        # Take respective actions

        x1_action = x1.gather(1, action.long().unsqueeze(-1))
        x2_action = x2.gather(1, action.long().unsqueeze(-1))
        return x1_action, x2_action


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        action_logits = self.linear3(x)
        return action_logits

    def sample(self, state):
        action_logits = self.forward(state)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action).unsqueeze(-1)
        probs = F.softmax(action_logits, dim=-1)
        return action, log_prob, probs

    def to(self, device):
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.logits = nn.Linear(hidden_dim, num_actions)
        self.softmax = nn.Softmax(dim=-1)
        # Small exploration noise for training
        self.exploration_eps = 0.1

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        action_logits = self.logits(x)
        return self.softmax(action_logits)

    def sample(self, state, evaluate=False):
        probs = self.forward(state)

        # During training, add small epsilon exploration
        if self.training or not evaluate:
            # Mix between argmax policy and uniform random
            random_probs = torch.ones_like(probs) / probs.shape[-1]
            mixed_probs = (1 - self.exploration_eps) * probs + self.exploration_eps * random_probs
            action_dist = torch.distributions.Categorical(mixed_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
        else:
            # During evaluation, just take argmax
            action = torch.argmax(probs, dim=-1)
            log_prob = torch.log(probs.gather(1, action.unsqueeze(-1)).squeeze(-1))

        # For deterministic policy, log_prob is 0 since it's not used in updates
        # Return action, log_prob (0), and raw probabilities
        return action, log_prob, probs

    def to(self, device):
        return super(DeterministicPolicy, self).to(device)
