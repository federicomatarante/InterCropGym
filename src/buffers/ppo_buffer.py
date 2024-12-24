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
        self.rewards = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.bool_)
        
        self.size = size
        self.ptr = 0             # Current insertion pointer

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
        
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done

        self.ptr += 1

    # def finish_path(self, last_value: float) -> None:
    #     """
    #     Compute returns and advantages for a trajectory when episode terminates.

    #     :param last_value: Value estimate for the last state (for bootstraping)
    #     """
    #     pass # Implement when needed GAE

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
            rewards=torch.as_tensor(self.rewards[:self.ptr], device=self.device),
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