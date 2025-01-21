import gymnasium as gym
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Type

from agents.agent import Agent
from agents.PPOAgent import PPOAgent
from enviroments.environment import Environment
from utils.ini_config_reader import INIConfigReader


class AgentTester:
    """
    Generic tester class for RL agents.

    :param env_id: The id of the environment to test the agent on.
    :param agent_type: Type of agent to test (PPOAgent, DQNAgent, etc.)
    :param config_path: Path to agent configuration file.
    :param device: Device to run the agent on (cpu or cuda).

    Usage:
        # Test PPO Agent
        tester = AgentTester(
            env_id="CartPole-v1",
            agent_type=PPOAgent,
            config_path="../data/configs/ppo.ini"
        )
        results = tester.run_test(num_episodes=5)

        # Test another agent (e.g. DQNAgent)
        tester.setup_agent(
            agent_type=DQNAgent,
            config_path="../data/configs/dqn.ini"
        )
        results = tester.run_test(num_episodes=5)
    """

    def __init__(self,
                 env_id: str,
                 agent_type: Type[Agent],
                 config_path: str,
                 device: Optional[torch.device] = None):
        
        # setup device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = None
        self.env_id = env_id
        self.agent_type = agent_type
        self.config_path = config_path

    def setup_agent(self, agent_type: Type[Agent], config_path: str) -> None:
        """
        Setup a new agent for testing.

        :param agent_type: Type of agent to create
        :param config_path: Path to agent configuration file
        """
        config = INIConfigReader(config_path)
        self.agent = agent_type(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            config=config,
            device=self.device
        )
        self.agent_name = agent_type.__name__

    def run_episode(self, max_steps: int=500, render : bool = False) -> Dict[str, Any]:
        """
        Run a single test episode.

        :param max_steps: Maximum number of steps per episode.
        :param render: Whether to render the environment.
        :return: Dictionary containing episode metrics.
        """
        state, _ = self.env.reset()
        episode_reward = 0
        metrics = {}

        # First step (for PPO-like algorithms that need to store transitions)
        if hasattr(self.agent, "act"):
            action = self.agent.act(state)[0] # returning a tuple of (action, ...)
        else:
            action = self.agent.select_action(state) # For other agents

        for step in range(max_steps):
            if render:
                self.env.env.render()

            # Take a step in the environment
            next_state, reward, done, truncated, _ = self.env.step(action)
            episode_reward += reward

            # Get next action 
            if hasattr(self.agent, "act"):
                # for PPO-like algorithms
                action = self.agent.act(next_state, reward, done or truncated)[0]
            else:
                # For other algorithms
                action = self.agent.select_action(next_state)

            # Update if episode is done
            if done or truncated:
                step_metrics = self.agent.update(state, action, reward, next_state, done)
                if step_metrics:
                    metrics.update(step_metrics)
                break

            state = next_state

        return {
            "reward": episode_reward,
            "steps": step + 1,
            **metrics
        }
    
    def run_test(self,
                 num_episodes: int = 5,
                 max_steps: int = 500,
                 render: bool = False,
                 verbose: bool = True) -> Dict[str, Any]:
        """
        Run multiple test episodes.

        :param num_episodes: Number of episodes to run.
        :param max_steps: Maximum number of steps per episode.
        :param render: Whether to render the environment.
        :param verbose: Whether to print test results.
        :return: Dictionary containing test results.
        """
        env_kwargs = {'render_mode': 'human'} if render else {}
        gym_env = gym.make(self.env_id, **env_kwargs)
        self.env = Environment(gym_env)

        # Setup agent if not done
        if not hasattr(self, 'agent'):
            self.state_dim = self.env.observation_space.shape[0]
            self.action_dim = (self.env.action_space.n
                               if isinstance(self.env.action_space, gym.spaces.Discrete)
                               else self.env.action_space.shape[0])
            self.setup_agent(self.agent_type, self.config_path)
            
        if verbose:
            print(f"\nTesting {self.agent_name} on {self.env.env.unwrapped.spec.id}")
            print(f"Episodes: {num_episodes}, Max Steps: {max_steps}")

        episode_results = []

        for episode in range(num_episodes):
            results = self.run_episode(max_steps, render)
            episode_results.append(results)

            if verbose:
                print(f"Episode {episode} - Reward: {results['reward']:.2f}, "
                      f"Steps: {results['steps']}")
                if 'metrics' in results:
                    print(f"Metrics: {results['metrics']}")

        # Compute summary statistics
        rewards = [r['reward'] for r in episode_results]
        steps = [r['steps'] for r in episode_results]

        summary = {
            'avg_reward': sum(rewards) / len(rewards),
            'min_reward': min(rewards),
            'max_reward': max(rewards),
            'avg_steps': sum(steps) / len(steps),
            'episode_results': episode_results
        }

        if verbose:
            print("\nTest Results:")
            print(f"Average Reward: {summary['avg_reward']:.2f}")
            print(f"Min/Max Reward: {summary['min_reward']:.2f}/{summary['max_reward']:.2f}")
            print(f"Average Steps: {summary['avg_steps']:.2f}")
        
        return summary

    def close(self):
        """Clean up resources."""
        self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()