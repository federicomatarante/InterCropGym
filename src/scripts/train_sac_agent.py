import os
import random
from pathlib import Path

import numpy as np
from pcse.exceptions import WeatherDataProviderError

from src.agents.sac_agent import SACAgent
from src.enviroments.environment import Environment
from src.enviroments.gymintercrop.intercropping_fertilization_env import IntercroppingFertilizationEnv
from src.trainings.agent_trainer import AgentTrainer
from src.utils.configs.ini_config_reader import INIConfigReader


def main():
    base_dir = Path(__file__).parent.parent.parent
    training_config_path = base_dir / 'data' / 'configs' / 'trainingConfig.ini'
    env_config_path = base_dir / 'data' / 'configs' / 'environment.ini'
    trainings_info_dir = base_dir / 'trainings' / 'SAC_agent'
    sac_config_path = base_dir / 'data' / 'configs' / 'sac.ini'
    os.makedirs(trainings_info_dir, exist_ok=True)
    training_config_reader = INIConfigReader(
        config_path=training_config_path,
        base_path=trainings_info_dir
    )
    env_config_reader = INIConfigReader(
        config_path=env_config_path,
        base_path=base_dir
    )
    sac_config_reader = INIConfigReader(
        config_path=sac_config_path,
        base_path=base_dir
    )

    env = IntercroppingFertilizationEnv(
        env_1_files={
            'crop': env_config_reader.get_param('files.env1_crop', v_type=Path),
            'site': env_config_reader.get_param('files.env1_site', v_type=Path),
            'soil': env_config_reader.get_param('files.env1_soil', v_type=Path),
        },
        env_2_files={
            'crop': env_config_reader.get_param('files.env2_crop', v_type=Path),
            'site': env_config_reader.get_param('files.env2_site', v_type=Path),
            'soil': env_config_reader.get_param('files.env2_soil', v_type=Path),
        },

    )

    agent = SACAgent(
        action_space=env.action_space,
        num_inputs=env.observation_space.shape[0],
        config_reader=sac_config_reader
    )


    trainer = AgentTrainer(
        agent=agent,
        env=Environment(env),
        config_reader=training_config_reader
    )

    trainer.train(
        plot_progress=training_config_reader.get_param("debug.plot_progress", v_type=bool, default=True),
        verbosity=training_config_reader.get_param("debug.verbosity", v_type=str, default="INFO",
                                                   domain={'INFO', 'DEBUG', 'WARNING', 'NONE'}),
        allowed_exceptions=(WeatherDataProviderError,)
    )


if __name__ == '__main__':
    import torch

    # Get current Python random seed
    python_seed = random.getstate()[1][0]
    print(f"Current Python random seed: {python_seed}")

    # Get current NumPy seed
    numpy_seed = np.random.get_state()[1][0]
    print(f"Current NumPy random seed: {numpy_seed}")

    # Get current PyTorch seed
    torch_seed = torch.initial_seed()
    print(f"Current PyTorch seed: {torch_seed}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
    main()
