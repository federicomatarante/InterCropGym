import abc
from typing import List, Dict, Any, Optional, Union

import torch
import torch.nn as nn

class BaseNetwork(nn.Module, abc.ABC):
    """
    Base neural network class that implements common functionalities and defines the interface for all neural networks.
    
    :param input_dim: Dimension of the input space.
    :param output_dim: Dimension of the output space.
    :param hidden_sized: List of hidden layer sizes.
    :param activation: Activation function to use between layers (e.g. 'ReLU', 'Tanh', 'Sigmoid').
    :param device: Device to place the network on.
    :param name: Name of the network.

    :ivar network: Sequential container of network layers.
    :ivar name: Network name for identification.
    :ivar device: Device the network is placed on.

    Usage:
        class CustomNetwork(BaseNetwork):
            def __init__(self, input_dim: int, output_dim: int, **kwargs):
                super().__init__(input_dim, output_dim, **kwargs)
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.network(x)

            def get_config(self) -> Dict[str, Any]:
                config = super().get_config()
                config.update({
                    'custom_param': self.custom_param
                })
                return config

    Example with training:
        config = {
            'input_dim': 4,
            'output_dim': 2,
            'hidden_sizes': [64, 64],
            'activation': 'relu',
        }
        network = CustomNetwork(**config)
        output = network(input_tensor)
    """

    def __init__(self,
                input_dim: int,
                output_dim: int,
                hidden_sizes: List[int],
                activation: str = "ReLU",
                device: torch.device = torch.device("gpu" if torch.cuda.is_available() else "cpu"),
                name: Optional[str] = None):
        super().__init__()
        self.device = device
        print(f"Using device: {self.device}")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name or self.__class__.__name__

        # Build network layers
        layers = []
        prev_size = input_dim

        # Get activation function
        activation_map = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'leakyrelu': nn.LeakyReLU
        }
        
        activation_lower = activation.lower()
        if activation_lower in activation_map:
            activation_fn = activation_map[activation_lower]
        else:
            raise ValueError(f"Activation function {activation} not supported. Choose from: {list(activation_map.keys())}")

        # Build hidden layers
        for size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, size),
                activation_fn()
            ])
            prev_size = size

        # Add output layer
        layers.append(nn.Linear(prev_size, output_dim))

        self.network = nn.Sequential(*layers)
        self.to(self.device)

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        Must be implemented by child classes.

        :param x: Input tensor
        :return: Output tensor
        """
        pass

    def save(self, path:str) -> None:
        """
        Save network parameters and configuration.

        :param path: Path to save location
        :raises OSError: If saving fails
        """
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.get_config()
        }, path)

    def load(self, path: str) -> None:
        """
        Load network parameters and configuration.

        :param path: Path to load location
        :raises FileNotFoundError: If file doesn't exist
        :raises ValueError: If saved configuration doesn't match
        """
        if not torch.cuda.is_available():
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(path)

        # Verify configuration matches
        saved_config = checkpoint['config']
        current_config = self.get_config()

        # Check critical parameters match
        for key in ['input_dim', 'output_dim']:
            if saved_config.get(key) != current_config.get(key):
                raise ValueError(
                    f"Saved model configuration mismatch: {key} "
                    f"saved: {saved_config.get(key)} current: {current_config.get(key)}"
                )
            
        self.load_state_dict(checkpoint['state_dict'])

    def get_config(self) -> Dict[str, Any]:
        """
        Get network configuration for saving/loading.
        Override in child classes to add additional parameters.

        :return: Dictionary containing network configuration
        """
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "device": self.device,
            "name": self.name
        }
    
    @property
    def num_parameters(self) -> int:
        """
        Get the total number of parameters in the network.

        :return: Total number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __str__(self) -> str:
        """String representation of the network architecture"""
        return (f"{self.name} Network:\n"
                f" Input dim: {self.input_dim}\n"
                f" Output dim: {self.output_dim}\n"
                f" Parameters: {self.num_parameters}\n"
                f" Device: {self.device}")