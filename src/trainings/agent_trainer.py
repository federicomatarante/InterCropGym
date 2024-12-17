import json
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
from numpy import floating
from tqdm import tqdm
import matplotlib.pyplot as plt


from src.agents.agent import Agent
from src.enviroments.environment import Environment
from src.utils.ConfigReader import ConfigReader


class AgentTrainer:
    """A training framework for reinforcement learning agents.

    :param agent: The RL agent to train
    :param env: The training environment
    :param config_path: Path to the INI configuration file

    :ivar agent: The reinforcement learning agent being trained
    :ivar env: The environment the agent interacts with
    :ivar config_data: Dictionary containing the complete configuration data
    :ivar train_episodes: Number of training episodes to run
    :ivar eval_episodes: Number of episodes used for each evaluation
    :ivar eval_frequency: How often to run evaluation (in episodes)
    :ivar max_steps_per_episode: Maximum number of steps allowed per episode
    :ivar save_frequency: How often to save checkpoints (in episodes)
    :ivar save_path: Directory path where checkpoints are saved
    :ivar log_path: Directory path where logs are saved
    :ivar early_stop_patience: Number of evaluations without improvement before early stopping
    :ivar early_stop_min_improvement: Minimum improvement required to reset early stopping counter
    :ivar train_returns: List of returns from training episodes
    :ivar eval_returns: List of average returns from evaluation periods
    :ivar train_steps: Total number of training steps taken

    :raises ValueError: If config file format is not an INI file or if config file format is invalid
    :raises KeyError: If config file is missing required parameters
    :raises FileNotFoundError: If config file does not exist

    Example:
        Create a configuration file (config.ini):
        ```ini
        [episodes]
        train_episodes = 1000
        eval_episodes = 20
        eval_frequency = 10
        max_steps_per_episode = 500

        [checkpoints]
        save_frequency = 100
        save_path = ./checkpoints
        log_path = ./logs

        [early_stopping]
        early_stop_patience = 20
        early_stop_min_improvement = 0.01
        ```

        Basic usage:
        ```python
        # Import necessary components
        from src.agents import MyAgent
        from src.environments import MyEnvironment
        from src.trainers import AgentTrainer

        # Create environment and agent
        env = MyEnvironment()
        agent = MyAgent(state_dim=env.state_dim, action_dim=env.action_dim)

        # Initialize trainer
        trainer = AgentTrainer(
            agent=agent,
            env=env,
            config_path='config.ini'
        )

        # Train the agent
        training_metrics = trainer.train()

        # Evaluate trained agent
        final_performance = trainer.evaluate(num_episodes=100)
        print(f"Final average return: {final_performance}")
        ```

        Loading from checkpoint:
        ```python
        # Initialize components
        env = MyEnvironment()
        agent = MyAgent(state_dim=env.state_dim, action_dim=env.action_dim)

        # Load agent from checkpoint
        checkpoint_path = "checkpoints/agent_ep500.pt"
        agent.load(checkpoint_path)

        # Create trainer with loaded agent
        trainer = AgentTrainer(agent=agent, env=env, config_path='config.ini')

        # Continue training or evaluate
        trainer.train()  # Continue training
        eval_score = trainer.evaluate(num_episodes=50)  # Evaluate performance
        ```

        Custom training loop with early stopping:
        ```python
        trainer = AgentTrainer(agent=agent, env=env, config_path='config.ini')

        best_return = float('-inf')
        patience_counter = 0

        for episode in range(trainer.train_episodes):
            # Run training episode
            returns = trainer._run_episode(training=True)

            # Periodic evaluation
            if episode % trainer.eval_frequency == 0:
                eval_return = trainer.evaluate(trainer.eval_episodes)

                # Early stopping check
                if eval_return > best_return + trainer.early_stop_min_improvement:
                    best_return = eval_return
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= trainer.early_stop_patience:
                    print("Early stopping triggered!")
                    break

            # Save checkpoint
            if episode % trainer.save_frequency == 0:
                trainer._save_checkpoint(episode)
        ```

        Training with live plotting:
        ```python
        # Train with live progress plotting
        trainer = AgentTrainer(agent=agent, env=env, config_path='config.ini')
        metrics = trainer.train(plot_progress=True)  # Will show plot during training

        # Or plot after training is complete
        trainer.plot_progress()  # Plot using collected metrics
        ```
    """

    def __init__(self, agent: Agent, env: Environment, config_path: Optional[str] = None):
        self.agent = agent
        self.env = env
        # Load config file
        self._load_config(config_path)
        # Training metrics
        self.train_returns = []
        self.eval_returns = []
        self.train_steps = 0
        # Plotting
        self._fig = None
        self._ax = None

    def _load_config(self, config_path: str) -> None:
        """Load and validate configuration from an INI file.

        :param config_path: Path to the INI configuration file
        :raises ValueError: If config file format is not an INI file or if config file format is invalid
        :raises KeyError: If config file is missing required parameters
        :raises FileNotFoundError: If config file does not exist
        """
        config_reader = ConfigReader(config_path)
        self.config_data = config_reader.config_data
        # Episodes
        self.train_episodes = config_reader.get_param('episodes.train_episodes')
        self.eval_episodes = config_reader.get_param('episodes.eval_episodes')
        self.eval_frequency = config_reader.get_param('episodes.eval_frequency')
        self.max_steps_per_episode = config_reader.get_param('episodes.max_steps_per_episode')
        # Checkpoints
        self.save_frequency = config_reader.get_param('checkpoints.save_frequency')
        self.save_path = config_reader.get_param('checkpoints.save_path')
        self.log_path = config_reader.get_param('checkpoints.log_path')
        # Hyperparameters
        self.early_stop_patience = config_reader.get_param('early_stop_patience')
        self.early_stop_min_improvement = config_reader.get_param('early_stop_patience')

    def plot_progress(self) -> None:
        """Plot the training and evaluation returns.

        Creates a figure showing the training returns and evaluation returns
        over episodes.
        """
        plt.close('all')  # Close any existing figures
        self._fig, self._ax = plt.subplots(figsize=(10, 5))

        # Plot training returns
        self._ax.plot(self.train_returns, label='Training Returns', alpha=0.6)

        # Plot evaluation returns at correct episodes
        if self.eval_returns:
            eval_episodes = range(0, len(self.eval_returns) * self.eval_frequency,
                                  self.eval_frequency)
            self._ax.plot(eval_episodes, self.eval_returns,
                          label='Evaluation Returns', linewidth=2)

        self._ax.set_xlabel('Episode')
        self._ax.set_ylabel('Return')
        self._ax.legend()
        self._ax.set_title('Training Progress')
        plt.show()

    def _update_plot(self) -> None:
        """Update the live training plot.

        Called during training when plot_progress=True to update the plot
        in real-time.
        """
        if self._fig is None or self._ax is None:
            self.plot_progress()
        else:
            self._ax.clear()
            self._ax.plot(self.train_returns, label='Training Returns', alpha=0.6)

            if self.eval_returns:
                eval_episodes = range(0, len(self.eval_returns) * self.eval_frequency,
                                      self.eval_frequency)
                self._ax.plot(eval_episodes, self.eval_returns,
                              label='Evaluation Returns', linewidth=2)

            self._ax.set_xlabel('Episode')
            self._ax.set_ylabel('Return')
            self._ax.legend()
            self._ax.set_title('Training Progress')
            self._fig.canvas.draw()
            plt.pause(0.01)  # Small pause to update the plot

    def train(self, plot_progress: bool = False) -> Dict[str, list]:
        """Train the agent using the specified configuration.

        :param plot_progress: Whether to show and update a plot during training
        :return: Dictionary containing training metrics including:
                - 'train_returns': List of returns from training episodes
                - 'eval_returns': List of average returns from evaluation periods
                - 'train_steps': Total number of training steps taken
        """
        best_eval_return = float('-inf')
        episodes_without_improvement = 0

        for episode in tqdm(range(self.train_episodes)):
            # Training episode
            episode_return = self._run_episode(training=True)
            self.train_returns.append(episode_return)

            # Periodic evaluation
            if episode % self.eval_frequency == 0:
                eval_return = self.evaluate(self.eval_episodes)
                self.eval_returns.append(eval_return)

                # Update plot if requested
                if plot_progress:
                    self._update_plot()

                # Early stopping check
                if eval_return > best_eval_return + self.early_stop_min_improvement:
                    best_eval_return = eval_return
                    episodes_without_improvement = 0
                else:
                    episodes_without_improvement += 1

                if episodes_without_improvement >= self.early_stop_patience:
                    print(f"Early stopping triggered at episode {episode}")
                    break

            # Save checkpoint
            if episode % self.save_frequency == 0:
                self._save_checkpoint(episode)

        # Final plot update if plotting was enabled
        if plot_progress:
            self._update_plot()

        return {
            'train_returns': self.train_returns,
            'eval_returns': self.eval_returns,
            'train_steps': self.train_steps
        }

    def evaluate(self, num_episodes: int) -> floating[Any]:
        """Evaluate the agent's performance by running multiple episodes without training.

        :param num_episodes: Number of evaluation episodes to run
        :return: Average return across all evaluation episodes
        """
        eval_returns = []
        for _ in range(num_episodes):
            episode_return = self._run_episode(training=False)
            eval_returns.append(episode_return)
        return np.mean(eval_returns)

    def _run_episode(self, training: bool = True) -> float:
        """Run a single episode in the environment.

        :param training: Whether to update the agent during the episode
        :return: Total reward accumulated during the episode
        """
        state, _ = self.env.reset()
        self.agent.reset()
        episode_return = 0

        for step in range(self.max_steps_per_episode):
            # Select action
            action = self.agent.act(state, explore=training)

            # Take step in environment
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # Update agent if training
            if training:
                self.agent.update(state, action, reward, next_state, done)
                self.train_steps += 1

            episode_return += reward
            state = next_state

            if done:
                break

        return episode_return

    def _save_checkpoint(self, episode: int) -> None:
        """Save the current state of training to disk.

        :param episode: Current episode number
        """
        save_dir = Path(self.save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'episode': episode,
            'train_returns': self.train_returns,
            'eval_returns': self.eval_returns,
            'train_steps': self.train_steps,
            'config': self.config_data
        }

        # Save agent
        agent_path = save_dir / f"agent_ep{episode}.pt"
        self.agent.save(str(agent_path))

        # Save training state
        state_path = save_dir / f"trainer_ep{episode}.json"
        with open(state_path, 'w') as f:
            json.dump(checkpoint, f, indent=4)