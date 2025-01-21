from typing import Tuple, Dict, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.enviroments.gymintercrop.intercropping.base_intercropping_system import BaseIntercroppingSystem
from src.enviroments.gymintercrop.intercropping.intercropping_system import StandardIntercroppingSystem
from src.enviroments.gymintercrop.open_crop_gym_env import OpenCropGymEnv


class IntercroppingFertilizationEnv(gym.Env):
    """
    A gym environment for simulating intercropping fertilization decisions.

    This environment combines two FertilizationEnv instances to simulate
    the concurrent management of two different crops. Actions affect both crops'
    fertilization levels, and rewards are combined from both environments.

    :param env_1_files: Dictionary containing paths for the first crop environment with keys:
                       'crop': path to crop file
                       'site': path to site file
                       'soil': path to soil file
    :param env_2_files: Dictionary containing paths for the second crop environment with keys:
                       'crop': path to crop file
                       'site': path to site file
                       'soil': path to soil file
    :param intercropping_system: Class implementing BaseIntercroppingSystem interface for
                                calculating intercropping effects (default: StandardIntercroppingSystem)

    :example:
        env_1_files = {'crop': 'wheat.crop', 'site': 'site.yaml', 'soil': 'soil.yaml'}
        env_2_files = {'crop': 'maize.crop', 'site': 'site.yaml', 'soil': 'soil.yaml'}
        env = IntercroppingFertilizationEnv(env_1_files, env_2_files)
    """

    def __init__(self, env_1_files: Dict, env_2_files: Dict,
                 intercropping_system: BaseIntercroppingSystem = StandardIntercroppingSystem(),
                 ) -> None:
        super(IntercroppingFertilizationEnv, self).__init__()
        self._fertilization_env_1: OpenCropGymEnv = OpenCropGymEnv(
            crop_file=env_1_files['crop'],
            site_file=env_1_files['site'],
            soil_file=env_1_files['soil']
        )
        self._fertilization_env_2: OpenCropGymEnv = OpenCropGymEnv(
            crop_file=env_2_files['crop'],
            site_file=env_2_files['site'],
            soil_file=env_2_files['soil']
        )
        # Define action and observation spaces
        self.action_space = self._fertilization_env_1.action_space

        self.observation_space: spaces.Box = spaces.Box(
            low=np.concatenate([
                self._fertilization_env_1.observation_space.low,
                self._fertilization_env_2.observation_space.low
            ]),
            high=np.concatenate([
                self._fertilization_env_1.observation_space.high,
                self._fertilization_env_2.observation_space.high
            ]),
            dtype=np.float32
        )
        self.intercropping_system = intercropping_system
        self.steps: int = 0
        self.max_steps: int = 300
        self.state: np.ndarray = np.array([])

    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.

        :return: Initial observation state as a numpy array combining both environments' states.

        :example:
            initial_state = env.reset()
            print(f"Initial state shape: {initial_state.shape}")
        """
        self.steps = 0
        state_1 = self._fertilization_env_1.reset()
        state_2 = self._fertilization_env_2.reset()
        self.state = np.concatenate([state_1, state_2])
        return self.state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step for both crops.

        The step process includes:
        1. Pre-step calculation for both environments
        2. Intercropping effects calculation
        3. Parameter updates for both environments
        4. Final step execution

        :param action: Combined action vector for both environments representing
                      fertilization amounts

        :return: Tuple containing:
            - next_state (numpy.ndarray): Combined observation of both environments' states
            - reward (float): Combined reward from both environments (sum of individual rewards)
            - done (bool): Whether one of the two crops has reached maximum growth
            - truncated (bool): Whether the episode has been terminated before either of the crops reached maximum
            growth
            - info (dict): Additional information including:
                - env1 (dict): Information from first environment
                - env2 (dict): Information from second environment
                - steps (int): Current step count
                - done_1: Whether the first crop has reached maximum growth
                - done_2: Whether the second crop has reached maximum growth
                - truncated_1: Whether the episode has been terminated before the first crop reached maximum growth
                - truncated_2: Whether the episode has been terminated before the second crop reached maximum growth

        :example:
            action = np.array([0.5])  # Same fertilization for both crops
            state, reward, done, info = env.step(action)
            print(f"Combined reward: {reward}, Steps: {info['steps']}")
        """
        self.steps += 1

        # Pre step
        output_1, crop_state_1 = self._fertilization_env_1.pre_step(action)
        output_2, crop_state_2 = self._fertilization_env_2.pre_step(action)

        # Mixing results for intercropping

        new_output_1, new_output_2, new_crop_state_1, new_crop_state_2 = self.intercropping_system.calculate_intercropping_effects(
            output_1, output_2, crop_state_1, crop_state_2
        )

        self._fertilization_env_1.update_env(new_output_1, new_crop_state_1)
        self._fertilization_env_2.update_env(new_output_2, new_crop_state_2)

        # Step through each environment
        state_1, reward_1, done_1, truncated_1, info_1 = self._fertilization_env_1.step(action)
        state_2, reward_2, done_2, truncated_2, info_2 = self._fertilization_env_2.step(action)

        self.state = np.concatenate([state_1, state_2])
        reward = reward_1 + reward_2
        done = done_1 or done_2 or self.steps >= self.max_steps
        truncated = truncated_1 or truncated_2
        info = {
            'env1': info_1,
            'env2': info_2,
            'done_1': done_1,
            'done_2': done_2,
            'truncated_1': truncated_1,
            'truncated_2': truncated_2,
            'steps': self.steps
        }

        return self.state, reward, done, truncated, info

    def render(self, mode: str = 'human') -> None:
        """
        Render environment visualization.
        Not implemented.
        """
        raise NotImplementedError()

    def close(self) -> None:
        """
        Clean up environment resources.

        Closes both underlying fertilization environments and frees associated resources.
        Should be called when the environment is no longer needed.

        :example:
            env.close()
        """
        self._fertilization_env_1.close()
        self._fertilization_env_2.close()
