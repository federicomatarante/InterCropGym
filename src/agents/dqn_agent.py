from pathlib import Path

import numpy as np
from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
import gymnasium as gym
from gym.spaces import Discrete

from src.agents.agent import Agent
from src.agents.sac.replay_memory import ReplayMemory
from src.utils.configs.config_reader import ConfigReader
from src.utils.configs.ini_config_reader import INIConfigReader


class QNetwork(nn.Module):
    """
    A neural network that approximates the Q-function, mapping states to Q-values for each possible action.

    :param state_dim: Dimension of the input state space
    :param num_actions: Number of possible actions in the environment
    :param hidden_dim: Size of hidden layers in the network
    """

    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        nn.init.uniform_(self.network[-1].weight, -1e-3, 1e-3)
        nn.init.uniform_(self.network[-1].bias, -1e-3, 1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Batch of state observations. Tensor of shape: [batch_size, state_size]
        :return: Q values for each action for each state. Tensor of shape: [batch_size, num_actions]
        """
        return self.network(x)


class DQNAgent(Agent):
    def __init__(self, num_inputs: int, action_space: Discrete, config_reader: ConfigReader, seed: int = None):
        """
        Initialize DQN agent using configuration parameters similar to SAC implementation.

        Args:
            num_inputs: Dimension of state space (renamed from state_dim for consistency)
            action_space: Gymnasium Discrete action space (replacing direct num_actions)
            config_reader: Configuration reader for loading parameters
            seed: Random seed for reproducibility
        """
        super().__init__()

        # Store dimensions
        self.state_dim = num_inputs
        self.num_actions = action_space.n

        # Load network configuration
        hidden_dim = config_reader.get_param('network.hidden_dim', v_type=int)
        learning_rate = config_reader.get_param('optimizer.learning_rate', v_type=float)

        # Load DQN-specific parameters
        self.gamma = config_reader.get_param('algorithm.gamma', v_type=float)
        self.epsilon_start = config_reader.get_param('exploration.epsilon_start', v_type=float)
        self.epsilon_end = config_reader.get_param('exploration.epsilon_end', v_type=float)
        self.epsilon_decay = config_reader.get_param('exploration.epsilon_decay', v_type=float)
        self.target_update_freq = config_reader.get_param('algorithm.target_update_freq', v_type=int)

        # Initialize current epsilon
        self.epsilon = self.epsilon_start
        self.update_count = 0

        # Setup device and networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(self.state_dim, self.num_actions, hidden_dim).to(self.device)
        self.target_network = QNetwork(self.state_dim, self.num_actions, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Setup optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.max_norm = config_reader.get_param('optimizer.max_norm',v_type=float)
        # Initialize replay buffer with configuration
        buffer_capacity = config_reader.get_param('memory.capacity', v_type=int)
        self.batch_size = config_reader.get_param('memory.batch_size', v_type=int)
        self.replay_buffer = ReplayMemory(buffer_capacity, seed)

        # Loss function
        self.loss_fn = nn.MSELoss()

        # Initialize metrics dictionary similar to SAC
        self.prev_logs = {
            'loss': None,
            'epsilon': self.epsilon,
            'avg_q_value': None
        }

    def act(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """
        Select a discrete action using epsilon-greedy policy.

        :param state: Current state observation
        :param explore: Whether to use exploration

        :return: action. Selected discrete action as a numpy array with a single integer
        """
        if explore and random.random() < self.epsilon:
            action = random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                action = torch.argmax(q_values).item()

        # Return action as numpy array to match interface
        return np.array([action], dtype=np.int64)

    def update(self, state: np.ndarray, action: np.ndarray,
               reward: float, next_state: np.ndarray, done: bool) -> Dict[str, float]:
        """
        Update the agent's policy based on a single transition.

        :param state: Current state
        :param action: Action taken (as numpy array with single integer)
        :param reward: Reward received
        :param next_state: Next state
        :param done: Whether episode terminated
        :return Dictionary of training metrics
        """
        # Store experience in replay buffer (convert action array to integer)
        self.replay_buffer.push(state, action[0], reward, next_state, done)

        # Only update if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0, "epsilon": self.epsilon, "avg_q_value": 0.0}

        # Sample batch and convert to tensors
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device).squeeze()
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        # Compute current Q values
        current_q_values = self.q_network(states_tensor).gather(1, actions_tensor)
        current_q_mean = current_q_values.mean().item()
        # Compute target Q values
        with torch.no_grad():
            next_actions = self.q_network(next_states_tensor).max(1)[1].unsqueeze(1)

            next_q_values = self.target_network(next_states_tensor).gather(1, next_actions).squeeze()
            target_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values
        # Compute loss and update
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=self.max_norm)
        self.optimizer.step()

        # Update target network if needed
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Track metrics
        metrics = {
            "loss": loss.item(),
            "epsilon": self.epsilon,
            "avg_q_value": current_q_mean
        }

        return metrics

    def _load_agent_checkpoint(self, path: Path):
        state_dict = torch.load(path)
        self.q_network.load_state_dict(state_dict['current_network'])
        self.target_network.load_state_dict(state_dict['target_network'])
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def load(self, path: str) -> None:
        base_path = str(path).removesuffix(".pt")
        agent_path = base_path / Path("dqn.pt")
        memory_path = base_path / Path("buffer.b")
        self.replay_buffer.load_buffer(memory_path)
        self._load_agent_checkpoint(agent_path)

    def _save_agent_checkpoint(self, path: Path):
        torch.save({'current_network': self.q_network.state_dict(),
                    'target_network': self.target_network.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, path)

    def save(self, path: str) -> None:
        base_path = path.removesuffix(".pt")
        agent_path = base_path / Path("dqn.pt")
        memory_path = base_path / Path("buffer.b")
        os.makedirs(base_path, exist_ok=True)
        self._save_agent_checkpoint(agent_path)
        self.replay_buffer.save_buffer(memory_path)
