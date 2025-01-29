from typing import Tuple, Dict, Any

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from src.networks.baseNetwork import BaseNetwork
from src.networks.utils.discrete_clipped_activation import DiscreteParameterizedActivation


class ActorNetwork(BaseNetwork):
    """
    Actor network for PPO with discrete action space.
    Outputs action probabilities for a Categorical distribution.
    
    :param input_dim: Dimension of input (state) space
    :param output_dim: Dimension of output (action) space
    :param hidden_sizes: List of hidden layer sizes
    :param activation: Activation function to use
    :param Device: Device to place the network on 

    Usage:
        actor = ActorNetwork(
            input_dim=4,
            action_dim=2, # Number of possible discrete actions
            hidden_sizes=[64, 64],
            activation='ReLU'
        )
        
        # Get action distribution for a state
        state = torch.FloatTensor([1.0, 2.0, 3.0, 4.0])
        action_dist = actor.get_distribution(state)
        
        # Sample action and get log probability
        action, log_prob = actor.get_action_and_log_prob(state)    
    """

    def __init__(self, input_dim: int, action_dim: int, max_val: int, **kwargs):
        super().__init__(input_dim=input_dim,
                         output_dim=action_dim,
                         name='DiscreteActor',
                         **kwargs)

        self.action_dim = action_dim
        self.act = DiscreteParameterizedActivation(
            max_val=max_val
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        :param state: State tensor of shape (batch_size, input_dim)
        :return: Action logits tensor of shape (batch_size, action_dim)
        """
        return self.network(state)

    def get_distribution(self, state: torch.Tensor) -> Categorical:
        """
        Get discrete action distribution for a given state.
        
        :param state: State tensor
        :return: Categorical distribution over actions
        """
        logits = self(state)
        # Convert logits to probabilities using softmax
        probs = F.softmax(logits, dim=-1)
        return Categorical(probs)

    def get_action_and_log_prob(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action and get its log probability using categorical distribution.
        
        :param state: State tensor
        :return: Tuple of (sampled_action, log_probability)
        """
        logits = self(state)
        # Get distribution
        dist = self.get_distribution(state)
        # Sample action
        action = dist.sample()
        # Get log prob of the sampled action
        log_prob = dist.log_prob(action)
        return action, log_prob

    def get_config(self) -> Dict[str, Any]:
        """Get network configuration."""
        config = super().get_config()
        config.update({
            'action_dim': self.action_dim
        })
        return config


class CriticNetwork(BaseNetwork):
    """
    Critic network for PPO that estimates state values.
    
    :param input-dim: Dimension of input (state) space
    :param hidden_sizes: List of hidden layer sizes
    :param activation: Activation function to use
    :param Device: Device to place the network on
    
    Usage:
        critic = CriticNetwork(
            input_dim=4,
            hidden_sizes=[64, 64],
            activation='ReLU'
        )
        
        state = torch.FloatTensor([1.0, 2.0, 3.0, 4.0])
        value = critic(state)
    """

    def __init__(self, input_dim: int, **kwargs):
        super().__init__(input_dim=input_dim,
                         output_dim=1,
                         name='Critic',
                         **kwargs)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        :param state: State tensor of shape (batch_size, input_dim)
        :return: Value tensor of shape (batch_size, 1)
        """
        value = self.network(state)
        return value.squeeze(-1)  # Remove last dimension
