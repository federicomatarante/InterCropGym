from typing import Dict, List

import numpy as np
import torch 

class PPOBuffer:
    """
    Buffer for storing trajectories experienced by a PPO agent.
    
    :param size: Maximum size of the buffer
    :param state_dim: Dimensions of obvservation space
    :param Device: Device to store tensors on

    Usage:
        buffer = PPOBuffer(
            size=2048,  # Number of steps to collect before update
            state_dim=4,  # State space dimension
            device=torch.device('cuda')
        )
        
        # During rollout:
        state = env.reset()
        for t in range(max_steps):
            buffer.store(
                state=state,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done
            )
            
        # At update time:
        data = buffer.get()  # Returns all stored data as tensors
        buffer.clear()  # Clear buffer after update
    """

    def __init__(self, size: int, state_dim: int, device: torch.device):
        self.device = device

        # Core storage
        self.states = np.zeros((size, state_dim), dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.int32)               # Discrete actions
        self.values = np.zeros(size, dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.bool_)
        self.raw_rewards = np.zeros(size, dtype=np.float32)
        self.normalized_rewards = np.zeros(size, dtype=np.float32)
        
        self.size = size
        self.ptr = 0             # Current insertion pointer
        self.reward_mean = 0
        self.reward_std = 1
        self.reward_count = 0
        self.epsilon = 1e-8
        self.reward_var = 0

    def update_reward_stats(self, reward: float):
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        self.reward_var = (self.reward_var * (self.reward_count - 1) + delta ** 2) / self.reward_count
        self.reward_std = np.sqrt(self.reward_var + 1e-8)

    def normalize_reward(self, reward: float) -> float:
        normalized = (reward - self.reward_mean) / (self.reward_std + self.epsilon)
        return np.clip(normalized, -10.0, 10.0)

    def store(self, state: np.ndarray, action: int, reward: float, value: float, log_prob: float, done: bool) -> None:
        """
        Store a transition in the buffer.
        
        :param state: Environment state/observation
        :param action: Action taken
        :param reward: Reward received
        :param value: Value estimate
        :param log_prob: Log probability of the action
        :param done: Whether the episode has terminated
        :raises ValueError: If the buffer is full
        """
        if self.ptr >= self.size:
            raise ValueError("Buffer is full. Call get() and clear() before adding more.")

        self.update_reward_stats(reward)
        self.raw_rewards[self.ptr] = reward
        self.normalized_rewards[self.ptr] = self.normalize_reward(reward)

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done

        self.ptr += 1

    def clear(self) -> None:
        """Clear the buffer."""
        self.ptr = 0

    def get(self) -> Dict[str, torch.Tensor]:
        """
        Get all stored data as tensors on the correct device.
        
        :return: Dictionary containing all buffer data
        :raises ValueError: If the buffer is empty
        """
        if self.ptr == 0:
            raise ValueError("Buffer is empty.")
        
        # Convert everything to tensors
        data = dict(
            states=torch.as_tensor(self.states[:self.ptr], device=self.device),
            actions=torch.as_tensor(self.actions[:self.ptr], device=self.device),
            normalized_rewards=torch.as_tensor(self.normalized_rewards[:self.ptr], device=self.device),
            raw_rewards=torch.as_tensor(self.raw_rewards[:self.ptr], device=self.device),
            values=torch.as_tensor(self.values[:self.ptr], device=self.device),
            log_probs=torch.as_tensor(self.log_probs[:self.ptr], device=self.device),
            dones=torch.as_tensor(self.dones[:self.ptr], device=self.device)
        )

        return data
    
    def __len__(self) -> int:
        """
        Get current number of stored transitions.
        
        :return: Number of transitions stored
        """
        return self.ptr

    def compute_gae(self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, next_value: float,
                    gamma: float = 0.99, gae_lambda: float = 0.95) -> np.ndarray:
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae

        returns = advantages + values
        return advantages, returns