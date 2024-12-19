import abc
from typing import Dict, List, SupportsFloat

import numpy as np


class Agent(abc.ABC):
    """Abstract base class for a general reinforcement learning agent.

    The Agent class provides a template for implementing reinforcement learning agents.
    It handles core functionality like action selection, learning from experience, and
    tracking episode statistics.

    Example usage:
        class DQNAgent(Agent):
            def __init__(self, state_dim: int, action_dim: int):
                super().__init__(state_dim, action_dim)
                self.q_network = create_network(state_dim, action_dim)

            def act(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
                if explore and np.random.random() < self.epsilon:
                    return np.random.randint(self.action_dim)
                return np.argmax(self.q_network.predict(state))

            def update(self, state, action, reward, next_state, done):
                target = reward + self.gamma * np.max(self.q_network.predict(next_state))
                self.q_network.update(state, action, target)

        # Usage
        env = gym.make('CartPole-v1')
        agent = DQNAgent(state_dim=4, action_dim=2)
        state = env.reset()
        for _ in range(1000):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
    """

    def __init__(self):
        self.episode_rewards: List[float] = []
        self.episode_steps = 0

    @abc.abstractmethod
    def act(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """Select an action given the current state.

        :param state: Current environment state
        :param explore: Whether to use exploration (e.g., epsilon-greedy) or pure exploitation
        :return: Selected action
        """
        pass

    @abc.abstractmethod
    def update(self, state: np.ndarray, action: np.ndarray,
               reward: SupportsFloat, next_state: np.ndarray, done: bool) -> Dict[str, float]:
        """Update the agent's policy based on a single transition.

        :param state: Current state
        :param action: Action taken
        :param reward: Reward received
        :param next_state: Next state
        :param done: Whether episode terminated
        :return: Dict containing relevant training metrics (e.g., loss values)
        """
        pass

    def reset(self) -> None:
        """Reset the agent's episode-specific variables."""
        self.episode_rewards = []
        self.episode_steps = 0

    def add_experience(self, reward: float) -> None:
        """Track episode rewards.

        :param reward: Reward received from environment
        """
        self.episode_rewards.append(reward)
        self.episode_steps += 1

    @property
    def episode_return(self) -> float:
        """Calculate total reward for current episode.

        :return: Sum of rewards for current episode
        """
        return sum(self.episode_rewards)

    def save(self, path: str) -> None:
        """Save agent parameters to disk.

        :param path: Path to save location
        :raises NotImplementedError: If save functionality is not implemented
        """
        raise NotImplementedError

    def load(self, path: str) -> None:
        """Load agent parameters from disk.

        :param path: Path to load location
        :raises NotImplementedError: If load functionality is not implemented
        """
        raise NotImplementedError
