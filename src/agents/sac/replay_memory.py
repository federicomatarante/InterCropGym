import os
import pickle
import random
from pathlib import Path
from typing import Tuple

import numpy as np


class ReplayMemory:
    """
    A circular buffer implementation for experience replay in reinforcement learning.
    Stores transitions (state, action, reward, next_state, done) and allows random sampling
    for training.

    :param capacity: Maximum number of transitions to store in the buffer
    :param seed: Random seed for reproducibility

    :ivar buffer (List): List storing the transitions
    :ivar position (int): Current position in the circular buffer
    :ivar capacity (int): Maximum buffer size
    """

    def __init__(self, capacity: int, seed: int = None):
        if seed is None:
            seed = random.randint(0, 1000)
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        """
        Add a new transition to the buffer. If buffer is full, oldest transitions
        are overwritten first.

        :param state: Current state observation
        :param action: Action taken
        :param reward: Reward received
        :param next_state: Next state observation
        :param done: Whether episode ended after this transition
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
         Randomly sample a batch of transitions from the buffer.
         :param batch_size: Number of transitions to sample
         :return: Tuple of (states, actions, rewards, next_states, dones) as numpy arrays
         :raises ValueError: If batch_size is larger than current buffer size
         """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, save_path: str | Path):
        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity
