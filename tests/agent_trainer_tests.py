import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import json
import tempfile
from pathlib import Path

from src.agents.agent import Agent
from src.enviroments.environment import Environment
from src.trainings.agent_trainer import AgentTrainer
from src.utils.configs.ini_config_reader import ConfigReader


class TestAgentTrainer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock dependencies
        self.mock_agent = Mock(spec=Agent)
        self.mock_env = Mock(spec=Environment)

        # Setup mock environment responses
        self.mock_env.reset.return_value = (np.zeros(4), {})  # state, info
        self.mock_env.step.return_value = (
            np.ones(4),  # next_state
            1.0,  # reward
            False,  # terminated
            False,  # truncated
            {}  # info
        )

        # Create temporary directory for saving checkpoints
        self.temp_dir = tempfile.mkdtemp()

        # Create mock config
        self.mock_config = {
            'episodes': {
                'train_episodes': 100,
                'eval_episodes': 5,
                'eval_frequency': 10,
                'max_steps_per_episode': 200
            },
            'checkpoints': {
                'save_frequency': 20,
                'save_path': self.temp_dir,
                'log_path': self.temp_dir
            },
            'early_stopping': {
                'early_stop_patience': 5,
                'early_stop_min_improvement': 0.1
            }
        }

        self.mock_config_reader = Mock(spec=ConfigReader)
        self.mock_config_reader.config_data = self.mock_config
        self.mock_config_reader.get_param = lambda x, v_type: self._mock_get_param(x, v_type)

        # Initialize trainer
        self.trainer = AgentTrainer(
            agent=self.mock_agent,
            env=self.mock_env,
            config_reader=self.mock_config_reader
        )

    def _mock_get_param(self, param, v_type):
        """Helper to mock ConfigReader.get_param."""
        section, key = param.split('.')
        value = self.mock_config[section][key]
        if v_type == Path:
            return Path(value)
        return v_type(value)

    def test_initialization(self):
        """Test proper initialization of AgentTrainer."""
        self.assertEqual(self.trainer.train_episodes, 100)
        self.assertEqual(self.trainer.eval_episodes, 5)
        self.assertEqual(self.trainer.eval_frequency, 10)
        self.assertEqual(self.trainer.max_steps_per_episode, 200)
        self.assertEqual(self.trainer.save_frequency, 20)
        self.assertEqual(str(self.trainer.save_path), self.temp_dir)
        self.assertEqual(str(self.trainer.log_path), self.temp_dir)
        self.assertEqual(self.trainer.early_stop_patience, 5)
        self.assertEqual(self.trainer.early_stop_min_improvement, 0.1)
        self.assertEqual(self.trainer.train_returns, [])
        self.assertEqual(self.trainer.eval_returns, [])
        self.assertEqual(self.trainer.train_steps, 0)

    def test_run_episode_training(self):
        """Test running a single training episode."""
        episode_return = self.trainer._run_episode(training=True)

        # Verify episode execution
        self.mock_env.reset.assert_called_once()
        self.assertTrue(self.mock_env.step.called)
        self.mock_agent.act.assert_called()
        self.mock_agent.update.assert_called()
        self.assertGreater(self.trainer.train_steps, 0)
        self.assertIsInstance(episode_return, float)

    def test_run_episode_evaluation(self):
        """Test running a single evaluation episode."""
        episode_return = self.trainer._run_episode(training=False)

        # Verify evaluation behavior
        self.mock_env.reset.assert_called_once()
        self.assertTrue(self.mock_env.step.called)
        self.mock_agent.act.assert_called()
        self.mock_agent.update.assert_not_called()
        self.assertIsInstance(episode_return, float)

    def test_evaluate(self):
        """Test evaluation over multiple episodes."""
        eval_return = self.trainer.evaluate(num_episodes=3)

        self.assertEqual(self.mock_env.reset.call_count, 3)
        self.assertIsInstance(eval_return, np.floating)

    @patch('matplotlib.pyplot.show')
    def test_plot_progress(self, mock_show):
        """Test plotting functionality."""
        self.trainer.train_returns = [1.0, 2.0, 3.0]
        self.trainer.eval_returns = [1.5, 2.5]

        self.trainer.plot_progress()
        mock_show.assert_called_once()

    def test_save_checkpoint(self):
        """Test checkpoint saving functionality."""
        self.trainer.episode = 20
        self.trainer.train_returns = [1.0, 2.0]
        self.trainer.eval_returns = [1.5]
        self.trainer.train_steps = 1000

        self.trainer._save_checkpoint()

        # Verify agent and environment were saved
        self.mock_agent.save.assert_called_once()
        self.mock_env.save.assert_called_once()

        # Verify paths were created
        save_dir = Path(self.temp_dir)
        self.assertTrue((save_dir / "agents").exists())
        self.assertTrue((save_dir / "environments").exists())
        self.assertTrue((save_dir / "trainings").exists())

    def test_from_checkpoint(self):
        """Test loading trainer from checkpoint."""
        # Create mock checkpoint file
        checkpoint_data = {
            'episode': 50,
            'train_returns': [1.0, 2.0],
            'eval_returns': [1.5],
            'train_steps': 1000,
            'config': self.mock_config
        }

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            json.dump(checkpoint_data, f)
            checkpoint_file = f.name

        # Test loading
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value = open(checkpoint_file)
            loaded_trainer = AgentTrainer.from_checkpoint(
                self.mock_agent,
                self.mock_env,
                checkpoint_file
            )

        # Cleanup
        Path(checkpoint_file).unlink()

    def test_training_with_early_stopping(self):
        """Test training process with early stopping."""
        # Mock evaluate to return decreasing values to trigger early stopping
        eval_returns = [2.0, 1.9, 1.8, 1.7, 1.6, 1.5]
        self.trainer.evaluate = Mock(side_effect=eval_returns)

        metrics = self.trainer.train(plot_progress=False, verbosity="NONE")

        self.assertIn('train_returns', metrics)
        self.assertIn('eval_returns', metrics)
        self.assertIn('train_steps', metrics)
        self.assertLess(len(metrics['eval_returns']), len(eval_returns))

    def test_get_checkpoint_paths(self):
        """Test checkpoint path generation."""
        save_dir = Path(self.temp_dir)
        episode = 100

        agent_path, env_path, state_path = AgentTrainer.get_checkpoint_paths(save_dir, episode)

        self.assertEqual(agent_path, save_dir / "agents" / f"agent_ep{episode}.pt")
        self.assertEqual(env_path, save_dir / "environments" / f"agent_ep{episode}.pt")
        self.assertEqual(state_path, save_dir / "trainings" / f"agent_ep{episode}.pt")

    def tearDown(self):
        """Clean up after each test."""
        # Remove temporary directory and all its contents
        import shutil
        shutil.rmtree(self.temp_dir)


if __name__ == '__main__':
    unittest.main()