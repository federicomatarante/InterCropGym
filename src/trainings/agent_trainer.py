import json
import os
import pickle
from json import JSONDecodeError
from typing import Dict, Any, Tuple
from pathlib import Path
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from numpy import floating
from tqdm import tqdm
import matplotlib.pyplot as plt
import tkinter as tk
from src.agents.agent import Agent
from src.enviroments.environment import Environment
from src.utils.configs.ini_config_reader import ConfigReader

import matplotlib

matplotlib.use('TkAgg')


class AgentTrainer:
    """A training framework for reinforcement learning agents.

    :param agent: The RL agent to train
    :param env: The training environment
    :param config_reader: ConfigReader for the training configuration. See examples for structure to have.

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


        Basic usage:

            # Create environment and agent
            cart_pole_env = CartPoleEnv()
            env = Environment(cart_pole_env)
            agent = DQNAgent(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n,
                learning_rate=0.001,
                gamma=0.99
            )

            # Initialize config reader with path to config file
            config_reader = INIConfigReader('training_config.ini')

            # Initialize trainer
            trainer = AgentTrainer(
                agent=agent,
                env=env,
                config_reader=config_reader
            )

            # Train the agent
            training_metrics = trainer.train()

            # Evaluate trained agent
            final_performance = trainer.evaluate(num_episodes=100)
            print(f"Final average return: {final_performance}")


        Starting from checkpoint:
        # Initialize components
            cart_pole_env = CartPoleEnv()
            env = Environment(cart_pole_env)
            agent = DQNAgent(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n,
                learning_rate=0.001,
                gamma=0.99
            )

            # Load agent from checkpoint
            agent_path, env_path, training_state_path = AgentTrainer.get_checkpoint_paths('./checkpoints',10)
            agent.load(agent_path)
            env.load(env_path)
            trainer = AgentTrainer.from_checkpoint(
                agent=agent,
                env=env,
                checkpoint_file=training_state_path
            )

            # Continue training or evaluate
            training_metrics = trainer.train()  # Continue training
            eval_score = trainer.evaluate(num_episodes=50)  # Evaluate performance
            print(f"Evaluation score: {eval_score}")

        Training with live plotting:
            # Setup environment and agent
            cart_pole_env = CartPoleEnv()
            env = Environment(cart_pole_env)
            agent = DQNAgent(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n,
                learning_rate=0.001,
                gamma=0.99
            )

            # Initialize trainer
            config_reader = ConfigReader('config.ini')
            trainer = AgentTrainer(
                agent=agent,
                env=env,
                config_reader=config_reader
            )

            # Train with live progress plotting
            metrics = trainer.train(plot_progress=True)  # Will show plot during training

            # Plot final results
            trainer.plot_progress()  # Plot using collected metrics

            print("Training metrics:", metrics)
"""

    def __init__(self, agent: Agent, env: Environment, config_reader: ConfigReader):
        self.agent = agent
        self.env = env
        # Load config file
        self._load_config(config_reader)
        # Training metrics
        self.train_returns = []
        self.eval_returns = []
        self.train_steps = 0
        self.episode = 0
        # Plotting
        self._fig = None
        self._ax = None
        self._root = None
        self._canvas = None

    def _load_config(self, config_reader: ConfigReader) -> None:
        """Load and validate configuration from an INI file.

        :param config_reader: A ConfigReader object with contains all the configuration of the training.
        :raises ValueError: If config file format is not an INI file or if config file format is invalid
        :raises KeyError: If config file is missing required parameters
        :raises FileNotFoundError: If config file does not exist
        """
        self.config_data = config_reader.config_data
        # Episodes
        self.train_episodes = config_reader.get_param('episodes.train_episodes', v_type=int)
        self.eval_episodes = config_reader.get_param('episodes.eval_episodes', v_type=int)
        self.eval_frequency = config_reader.get_param('episodes.eval_frequency', v_type=int)
        self.max_steps_per_episode = config_reader.get_param('episodes.max_steps_per_episode', v_type=int)
        # Checkpoints
        self.save_frequency = config_reader.get_param('checkpoints.save_frequency', v_type=int)
        self.save_path = config_reader.get_param('checkpoints.save_path', v_type=Path)
        self.log_path = config_reader.get_param('checkpoints.log_path', v_type=Path)
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.save_path / 'agents', exist_ok=True)
        os.makedirs(self.save_path / 'environments', exist_ok=True)
        os.makedirs(self.save_path / 'trainings', exist_ok=True)

        os.makedirs(self.log_path, exist_ok=True)

        # Hyperparameters
        self.early_stop_patience = config_reader.get_param('early_stopping.early_stop_patience', v_type=int)
        self.early_stop_min_improvement = config_reader.get_param('early_stopping.early_stop_min_improvement',
                                                                  v_type=float)

    @staticmethod
    def from_checkpoint(agent: Agent, env: Environment, checkpoint_file: str): # TODO shoudl test this before definitive!
        """
        Creates an AgentTrainer instance from the given checkpoint.
        :param agent: the agent to use for the training. Careful! Load it before starting the training with proper  methods.
        :param env: the environment to use for the training. Careful! Load it before starting the training with proper
            methods.
        :param checkpoint_file: the file that contains the training state.
        :raises FileNotFoundError: if file doesn't exist-
        :raises ValueError: if file is not in the valid format.
        :return: AgentTrainer loaded object.
        """
        with open(checkpoint_file, 'w') as f:
            try:
                checkpoint = json.load(f)
            except JSONDecodeError | TypeError | UnicodeDecodeError as e:
                raise ValueError(f"Invalid file format: \n{e}")

        trainer = AgentTrainer(
            agent,
            env,
            ConfigReader(checkpoint)
        )
        trainer.train_returns = checkpoint['train_returns']
        trainer.eval_returns = checkpoint['eval_returns']
        trainer.train_steps = checkpoint['train_steps']
        trainer.episode = checkpoint['episode']

    def plot_progress(self) -> None:
        """Plot the training and evaluation returns.

        Creates a figure showing the training returns and evaluation returns
        over episodes.
        """
        if self._root is None:
            self._root = tk.Tk()
            self._root.protocol('WM_DELETE_WINDOW', self._on_closing)

        plt.close('all')
        self._fig, self._ax = plt.subplots(figsize=(10, 5))

        # Plot training returns
        self._ax.plot(self.train_returns, label='Training Returns', alpha=0.6)

        # Plot evaluation returns
        if self.eval_returns:
            eval_episodes = range(0, len(self.eval_returns) * self.eval_frequency, self.eval_frequency)
            self._ax.plot(eval_episodes, self.eval_returns, label='Evaluation Returns', linewidth=2)

        self._ax.set_xlabel('Episode')
        self._ax.set_ylabel('Return')
        self._ax.legend()
        self._ax.set_title('Training Progress')

        if self._canvas is None:
            self._canvas = FigureCanvasTkAgg(self._fig, master=self._root)
            self._canvas.draw()
            self._canvas.get_tk_widget().pack()

        self._root.update()

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
                eval_episodes = range(0, len(self.eval_returns) * self.eval_frequency, self.eval_frequency)
                self._ax.plot(eval_episodes, self.eval_returns, label='Evaluation Returns', linewidth=2)

            self._ax.set_xlabel('Episode')
            self._ax.set_ylabel('Return')
            self._ax.legend()
            self._ax.set_title('Training Progress')

            if self._canvas:
                self._canvas.draw()
                self._root.update()

    def _on_closing(self):
        if self._root:
            self._root.quit()
            self._root.destroy()
            self._root = None
            self._canvas = None
            self._fig = None
            self._ax = None

    def train(self, plot_progress: bool = False, verbosity: str = "INFO", allowed_exceptions: tuple = ()) -> Dict[
        str, list]:
        """Train the agent using the specified configuration.
        Some features:
            - Periodic evaluation
            - Early stopping
            - Tracking of checkpoints
            - Configurable verbosity levels
            - Exception handling for specified exceptions

        :param plot_progress: Whether to show and update a plot during training
        :param verbosity: Print verbosity level ('DEBUG', 'INFO', 'WARNING', 'NONE')
        :param allowed_exceptions: Tuple of exception types that should be caught and ignored during training
        :return: Dictionary containing training metrics including:
                - 'train_returns': List of returns from training episodes
                - 'eval_returns': List of average returns from evaluation periods
                - 'train_steps': Total number of training steps taken
        """
        # Set verbosity level
        verbosity = verbosity.upper()
        verbosity_levels = {"DEBUG": 3, "INFO": 2, "WARNING": 1, 'NONE': 0}
        self.verbosity_level = verbosity_levels.get(verbosity, 0)  # Default to NONE

        if self.verbosity_level >= 2:
            print(f"Starting training with {self.train_episodes} episodes")
        if self.verbosity_level >= 3:
            print(f"Configuration - Eval frequency: {self.eval_frequency}, "
                  f"Early stop patience: {self.early_stop_patience}, "
                  f"Save frequency: {self.save_frequency}")

        best_eval_return = float('-inf')
        episodes_without_improvement = 0

        for self.episode in tqdm(range(self.train_episodes)):
            try:

                train_return_item, eval_return_item = None, None
                # Training episode
                if self.verbosity_level >= 3:
                    print(f"Starting episode {self.episode + 1}/{self.train_episodes}")
                episode_return = self._run_episode(training=True)
                train_return_item = episode_return.item()

                if self.verbosity_level >= 2:
                    print(f"Episode {self.episode} completed with reward: {float(episode_return):.2f}")

                # Periodic evaluation
                if self.episode % self.eval_frequency == 0:
                    if self.verbosity_level >= 2:
                        print(f"Running evaluation at episode {self.episode}")
                    eval_return = self.evaluate(self.eval_episodes,verbosity)
                    eval_return_item = eval_return.item()
                    if self.verbosity_level >= 2:
                        print(f"Evaluation return: {eval_return:.2f}")

                    # Update plot if requested
                    if plot_progress:
                        if self.verbosity_level >= 3:
                            print("Updating training progress plot")
                        self._update_plot()

                    # Early stopping check
                    if eval_return > best_eval_return + self.early_stop_min_improvement:
                        best_eval_return = eval_return
                        episodes_without_improvement = 0
                        if self.verbosity_level >= 2:
                            print(f"New best evaluation return: {best_eval_return:.2f}")
                    else:
                        episodes_without_improvement += 1
                        if self.verbosity_level >= 3:
                            print(f"Episodes without improvement: {episodes_without_improvement}")

                    if episodes_without_improvement >= self.early_stop_patience:
                        if self.verbosity_level >= 1:
                            print(f"Early stopping triggered at episode {self.episode}")
                        break

                # Save checkpoint
                if self.episode % self.save_frequency == 0:
                    if self.verbosity_level >= 2:
                        print(f"Saving checkpoint at episode {self.episode}")
                    self._save_checkpoint()

            except allowed_exceptions as e:
                if self.verbosity_level >= 1:
                    print(f"Caught allowed exception in episode {self.episode}: {str(e)}")
                continue
            if train_return_item:
                self.train_returns.append(train_return_item)
            if eval_return_item:
                self.eval_returns.append(eval_return_item)

        # Final plot update if plotting was enabled
        if plot_progress:
            if self.verbosity_level >= 3:
                print("Updating final training progress plot")
            self._update_plot()

        if self.verbosity_level >= 2:
            print("Training completed")
            print(f"Final training steps: {self.train_steps}")
        if self.verbosity_level >= 3:
            print(f"Final training returns: {self.train_returns[-1]:.2f}")
            print(f"Final evaluation returns: {self.eval_returns[-1]:.2f}")

        return {
            'train_returns': self.train_returns,
            'eval_returns': self.eval_returns,
            'train_steps': self.train_steps
        }

    def evaluate(self, num_episodes: int, verbosity: str = "INFO") -> floating[Any]:
        """Evaluate the agent's performance by running multiple episodes without training.

        :param num_episodes: Number of evaluation episodes to run
        :param verbosity: Print verbosity level ('DEBUG', 'INFO', 'WARNING', 'NONE')
        :return: Average return across all evaluation episodes
        """
        verbosity_levels = {"DEBUG": 3, "INFO": 2, "WARNING": 1, 'NONE': 0}
        self.verbosity_level = verbosity_levels.get(verbosity, 0)
        eval_returns = []
        for i in range(num_episodes):
            if self.verbosity_level >= 3:
                print(f"Starting episode {i + 1}/{num_episodes}")
            episode_return = self._run_episode(training=False)
            if self.verbosity_level >= 3:
                print(f"Reward of episode {i + 1}: {episode_return}")
            eval_returns.append(episode_return.item())
        return np.mean(eval_returns)

    def _run_episode(self, training: bool = True) -> float:
        """Run a single episode in the environment.

        :param training: Whether to update the agent during the episode
        :return: Total reward accumulated during the episode
        """
        state = self.env.reset()
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
                metrics = self.agent.update(state, action, reward, next_state, done)
                self.train_steps += 1
                if metrics and self.verbosity_level >=2:
                    print("\nTraining metrics")
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"{key}: {value:.4f}")
                        else:
                            print(f"{key}: {value}")

            episode_return += reward
            state = next_state

            if done:
                break
        return episode_return

    def _save_checkpoint(self) -> None:
        """Save the current state of training to disk.
        """
        save_dir = Path(self.save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'episode': self.episode,
            'train_returns': self.train_returns,
            'eval_returns': self.eval_returns,
            'train_steps': self.train_steps,
            'config': self.config_data
        }

        if hasattr(self.agent, 'actor_scheduler') and hasattr(self.agent, 'critic_scheduler'):
            checkpoint.update({
                'actor_scheduler': self.agent.actor_scheduler.state_dict(),
                'critic_scheduler': self.agent.critic_scheduler.state_dict()
            })

        agent_path, env_path, state_path = AgentTrainer.get_checkpoint_paths(save_dir, self.episode)
        # Save agent
        self.agent.save(str(agent_path))
        # Save environment
        self.env.save(str(env_path))
        # Save training state
        with open(state_path, 'wb') as f:
            pickle.dump(checkpoint, f)

    @staticmethod
    def get_checkpoint_paths(save_dir: str | Path, episode: int) -> Tuple[Path, Path, Path]:
        """
        Computes the paths of the agent, environment and training state files for the checkpoint.
        :param save_dir: the directory which contains all the checkpoints.
        :param episode: the episode of the checkpoints.
        :return: (agent_path,env_path,training_state_path)
        """

        save_dir_path = Path(save_dir)
        return (
            save_dir_path / "agents" / f"agent_ep{episode}.pt",
            save_dir_path / "environments" / f"agent_ep{episode}.pt",
            save_dir_path / "trainings" / f"agent_ep{episode}.pt")
