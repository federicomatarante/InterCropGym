import os
from datetime import timedelta
from typing import Tuple

import numpy as np
import pandas as pd
import pcse
from pcse.input import PCSEFileReader

from gym_crop.envs import FertilizationEnv
from gym_crop.envs.fertilization_env import DATA_DIR
from src.enviroments.gymintercrop.utils.crop_state import CropState
from src.enviroments.gymintercrop.utils.lintul3_parameters import LINTUL3Parameters


class OpenCropGymEnv(FertilizationEnv):
    """
    An environment for simulating crop fertilization with open-loop control capabilities.
    Extends the base FertilizationEnv class by adding methods for pre-step simulation
    and parameter updates.

    :param crop_file: Path or name to the crop parameters file relative to DATA_DIR/crop
    :param site_file: Path or name to the site parameters file relative to DATA_DIR/site
    :param soil_file: Path or name to the soil parameters file relative to DATA_DIR/soil
    :param intervention_interval: Number of days between possible fertilization interventions
    :param beta: Penalty coefficient for fertilizer use in reward calculation
    :param seed: Random seed for reproducibility
    :param fixed_year: If provided, uses this specific year for simulation
    :param fixed_location: If provided, uses this specific location for simulation

    :example:
        env = OpenFertilizationEnv("wheat.crop", "site.yaml", "soil.yaml", intervention_interval=7)
        metrics, params = env.pre_step(action=0.5)
        env.update_params(new_metrics, new_params)
        obs, reward, done, info = env.step(action=0.5)
    """

    def __init__(self, crop_file: str, site_file: str, soil_file: str, intervention_interval: int = 7, beta: float = 10,
                 seed: int = 0, fixed_year: int = None, fixed_location: int = None):
        super().__init__(intervention_interval, beta, seed, fixed_year, fixed_location)
        self._last_fertilizer = None
        self._last_output = None
        self._pre_step_done = False
        self._update_done = False
        crop = PCSEFileReader(crop_file)
        soil = PCSEFileReader(soil_file)
        site = PCSEFileReader(site_file)
        self.parameterprovider = pcse.base.ParameterProvider(soildata=soil, cropdata=crop,
                                                             sitedata=site)
        self.model = pcse.models.LINTUL3(self.parameterprovider, self.weatherdataprovider,
                                         self.agromanagement)

        self.baseline_model = pcse.models.LINTUL3(self.parameterprovider, self.weatherdataprovider,
                                                  self.agromanagement)

    def pre_step(self, action: np.ndarray) -> Tuple[CropState, LINTUL3Parameters]:
        """
        Performs the preliminary step of applying an action and running the simulation,
        without updating the environment state.

        :param action: The fertilization action to take.
        :return: A tuple (CropMetrics, SimulationParameters) containing the simulation
                results and current model parameters

        :example:
            metrics, params = env.pre_step(0.5)
        """

        self._last_fertilizer = self._take_action(action)
        self._last_output = self._run_simulation(self.model)

        self._pre_step_done = True
        return CropState.from_dataframe(self._last_output), LINTUL3Parameters.from_model(self.model)

    def update_env(self, new_crop_state: CropState, new_env_state: LINTUL3Parameters):
        """
        Updates the environment's internal state with new metrics and parameters.
        Must be called after pre_step and before step.

        :param new_crop_state: Updated crop metrics to apply to the environment.
        :param new_env_state: Updated simulation parameters to apply to the model.

        :example:
            env.update_params(metrics, params)
        """
        self._update_done = True
        new_output = new_crop_state.to_dataframe()
        self._last_output = self._last_output.iloc[:-1]
        self._last_output = pd.concat([self._last_output, new_output])
        new_env_state.update_model(self.model)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Executes a step in the environment, must be called after pre_step and update_params.
        Computes the reward based on growth comparison with baseline model and fertilizer cost.

        :param action: The fertilization action to take (typically a float representing
                      the amount of fertilizer to apply)
        :return: A tuple (observation, reward, done, info) where:
                - observation is the current state
                - reward is the calculated reward for the action
                - done is a boolean indicating if the plant has reached maximum growth is finished.
                - truncated is a boolean indicating if the episode is terminated without reaching the maximum growth.
                - info is a dictionary containing additional information

        :raises BrokenPipeError: If called before pre_step and update_params

        :example:
            obs, reward, done, info = env.step(action)
        """
        if not (self._update_done or self._pre_step_done):
            raise BrokenPipeError("You must call 'pre_step' and 'update_params' before calling 'step'!")

        baseline_output = self._run_simulation(self.baseline_model)
        observation = self._process_output(self._last_output)
        self.date = self._last_output.index[-1]

        last_idx = len(self._last_output) - 1
        interval_idx = max(0, last_idx - self.intervention_interval)

        growth = self._last_output['WSO'].iloc[last_idx] - self._last_output['WSO'].iloc[interval_idx]
        growth = growth if not np.isnan(growth) else 0

        baseline_last_idx = len(baseline_output) - 1
        baseline_interval_idx = max(0, baseline_last_idx - self.intervention_interval)
        baseline_growth = baseline_output['WSO'].iloc[baseline_last_idx] - baseline_output['WSO'].iloc[
            baseline_interval_idx]
        baseline_growth = baseline_growth if not np.isnan(baseline_growth) else 0

        reward = growth - baseline_growth - self.beta * self._last_fertilizer
        truncated = self.date >= self.crop_end_date - timedelta(days=self.intervention_interval)
        done = self.model.get_variable('DVS') >= 2.0
        self._log(growth, baseline_growth, self._last_fertilizer, reward)

        info = {**self._last_output.to_dict(), **self.log}
        self._pre_step_done, self._update_done = False, False

        return observation, reward, done, truncated, info
