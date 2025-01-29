import os
from pathlib import Path

import torch
from pcse.exceptions import WeatherDataProviderError

from src.agents.PPOAgent import PPOAgent
from src.enviroments.environment import Environment
from src.enviroments.gymintercrop.intercropping_fertilization_env import IntercroppingFertilizationEnv
from src.trainings.agent_trainer import AgentTrainer
from src.utils.configs.ini_config_reader import INIConfigReader


def main():
    base_dir = Path(__file__).parent.parent.parent
    training_config_path = base_dir / 'data' / 'configs' / 'trainingConfig.ini'
    env_config_path = base_dir / 'data' / 'configs' / 'environment.ini'
    trainings_info_dir = base_dir / 'trainings' / 'PPO_agent'
    ppo_config_path = base_dir / 'data' / 'configs' / 'ppo.ini'
    os.makedirs(trainings_info_dir, exist_ok=True)
    training_config_reader = INIConfigReader(
        config_path=training_config_path,
        base_path=trainings_info_dir
    )
    env_config_reader = INIConfigReader(
        config_path=env_config_path,
        base_path=base_dir
    )
    ppo_config_reader = INIConfigReader(
        config_path=ppo_config_path,
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

    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=1,
        max_val=env.action_space.n,
        config=ppo_config_reader,
        device=torch.device('cuda')
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
    main()
