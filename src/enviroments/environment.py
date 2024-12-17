from typing import Tuple, Dict, Any, Union

import gymnasium as gym
import numpy
import numpy as np
import torch
from gymnasium import Space


class Environment:
    """
    A wrapper for gymnasium environments\
    :param env: The gymnasium environment
    :param use_tensor: Whether to return tensors instead of numpy arrays
    :param device: The device to place the tensors on if use_tensor is True

    Usage:
        import gymnasium as gym
        import torch

        # Using numpy arrays
        with Environment(gym.make('CartPole-v1')) as env:
            obs, info = env.reset()
            action = env.sample_action()
            obs, reward, terminated, truncated, info = env.step(action)

        # Using tensors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with Environment(gym.make('CartPole-v1'), use_tensor=True, device=device) as env:
            obs, info = env.reset()  # obs will be a tensor
            action = env.sample_action()  # action will be a tensor
            obs, reward, terminated, truncated, info = env.step(action)
    """

    def __init__(self, env: gym.Env, use_tensor: bool = False, device: torch.device = torch.device('cpu')):
        self.env = env
        self.use_tensor = use_tensor
        self.device = device

    # Methods useful to automatically opening and closing the enviroment with "with"

    def __enter__(self):
        """
        Enter the context manager\
        :return: self
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager\
        :param exc_type: Exception type if an exception was raised
        :param exc_val: Exception value if an exception was raised
        :param exc_tb: Exception traceback if an exception was raised
        """
        self.close()

    def close(self):
        """
        Closes the environment
        """
        self.env.close()

    # Enviroment exploration methods

    def reset(self) -> Tuple[Union[numpy.ndarray, torch.Tensor], Dict[str, Any]]:
        """
        Resets the environment to initial state\
        :return: A tuple containing (observation, info dictionary). Observation is tensor if use_tensor=True, numpy array otherwise
        """
        state, info = self.env.reset()
        if self.use_tensor:
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        return state, info

    def step(self, action: Union[numpy.ndarray, torch.Tensor]) -> Tuple[
        Union[numpy.ndarray, torch.Tensor], float, bool, bool, Dict[str, Any]]:
        """
        Steps the environment\
        :param action: The action to take in the environment (tensor or numpy array)
        :return: A tuple containing (observation, reward, terminated, truncated, info). Observation is tensor if use_tensor=True, numpy array otherwise
        """
        if isinstance(action, torch.Tensor):
            cpu = torch.device('cpu')
            action = action.to(cpu).numpy().astype(np.float32)

        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.use_tensor:
            observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
        return observation, reward, terminated, truncated, info

    def sample_action(self) -> Union[numpy.ndarray, torch.Tensor]:
        """
        Samples a random action from the environment's action space\
        :return: A random action (tensor if use_tensor=True, numpy array otherwise)
        """
        action = self.env.action_space.sample()
        if self.use_tensor:
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        return action

    # Enviroment properties

    @property
    def observation_space(self) -> Space[Any]:
        """
        Gets the observation space of the environment\
        :return: The observation space
        """
        return self.env.observation_space

    @property
    def action_space(self) -> Space[Any]:
        """
        Gets the action space of the environment\
        :return: The action space
        """
        return self.env.action_space
