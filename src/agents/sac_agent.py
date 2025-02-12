import os
from pathlib import Path
from typing import SupportsFloat, Dict

import numpy as np
from gym import Space
from gym.spaces import Discrete

from src.agents.agent import Agent
from src.agents.sac.replay_memory import ReplayMemory
from src.agents.sac.discretesac import DiscreteSAC
from src.agents.utils.frequency_updater import FrequencyUpdater, FixedUpdater, DecayingUpdater
from src.utils.configs.config_reader import ConfigReader


def _get_frequency_updater(config_reader: ConfigReader):
    strategy = config_reader.get_param('memory.strategy', v_type=str, domain={"Fixed", "Decay"})
    update_frequency = config_reader.get_param('memory.update_frequency', v_type=int)
    if strategy == 'Fixed':
        return FixedUpdater(update_frequency)
    else:
        min_frequency = config_reader.get_param('memory.min_update_frequency', v_type=int)
        decay_rate = config_reader.get_param('memory.decay_rate', v_type=float)
        return DecayingUpdater(
            start_freq=update_frequency, min_freq=min_frequency, decay_rate=decay_rate
        )


class SACAgent(Agent):

    def __init__(self, num_inputs: int, action_space: Discrete, config_reader: ConfigReader, seed: int = None):
        super().__init__()
        self.prev_logs = {
            'qf1_loss': None,
            'qf2_loss': None,
            'policy_loss': None,
            'alpha_loss': None,
            'alpha_value': None,
        }
        self.agent = DiscreteSAC(num_inputs, action_space, config_reader)
        self.batch_size = config_reader.get_param('memory.batch_size', v_type=int)
        capacity = config_reader.get_param('memory.capacity', v_type=int)
        self.memory = ReplayMemory(capacity, seed)
        self.frequency_updater: FrequencyUpdater = _get_frequency_updater(config_reader)

    def load(self, path: str | Path) -> None:
        base_path = str(path).removesuffix(".pt")
        agent_path = base_path / Path("sac.pt")
        memory_path = base_path / Path("buffer.b")
        self.agent.load_checkpoint(agent_path)
        self.memory.load_buffer(memory_path)

    def save(self, path: str | Path) -> None:
        base_path = path.removesuffix(".pt")
        agent_path = base_path / Path("sac.pt")
        memory_path = base_path / Path("buffer.b")
        os.makedirs(base_path, exist_ok=True)
        self.agent.save_checkpoint(agent_path)
        self.memory.save_buffer(memory_path)

    def update(self, state: np.ndarray, action: np.ndarray, reward: SupportsFloat, next_state: np.ndarray,
               done: bool) -> Dict[str, float]:
        self.memory.push(state, action, float(reward), next_state, done)
        self.frequency_updater.step()
        if len(self.memory) > self.batch_size and self.frequency_updater.update():
            log_values = self.agent.update_parameters(self.memory, self.batch_size, self.frequency_updater.updates)
            self.prev_logs = {
                'qf1_loss': log_values[0],
                'qf2_loss': log_values[1],
                'policy_loss': log_values[2],
                'alpha_loss': log_values[3],
                'alpha_value': log_values[4],
            }

        return self.prev_logs

    def act(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        return self.agent.select_action(state, not explore)
