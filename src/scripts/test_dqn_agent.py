import os
from pathlib import Path

from pcse.exceptions import WeatherDataProviderError

from src.agents.dqn_agent import DQNAgent
from src.enviroments.environment import Environment
from src.enviroments.gymintercrop.intercropping_fertilization_env import IntercroppingFertilizationEnv
from src.trainings.agent_trainer import AgentTrainer
from src.utils.configs.ini_config_reader import INIConfigReader


def main():
    base_dir = Path(__file__).parent.parent.parent
    training_config_path = base_dir / 'data' / 'configs' / 'trainingConfig.ini'
    env_config_path = base_dir / 'data' / 'configs' / 'environment.ini'
    trainings_info_dir = base_dir / 'trainings' / 'DQN_agent'
    ppo_config_path = base_dir / 'data' / 'configs' / 'dqn.ini'
    os.makedirs(trainings_info_dir, exist_ok=True)

    training_config_reader = INIConfigReader(
        config_path=training_config_path,
        base_path=trainings_info_dir
    )

    env_config_reader = INIConfigReader(
        config_path=env_config_path,
        base_path=base_dir
    )

    dqn_config_reader = INIConfigReader(
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

    agent = DQNAgent(
        action_space=env.action_space,
        num_inputs=env.observation_space.shape[0],
        config_reader=dqn_config_reader
    )

    checkpoint_episode = training_config_reader.get_param("checkpoints.load_episode", v_type=int)
    checkpoints_dir = training_config_reader.get_param("checkpoints.save_path", v_type=str)
    agent_ckp, env_ckp, training_ckp = AgentTrainer.get_checkpoint_paths(save_dir=trainings_info_dir / checkpoints_dir,
                                                                         episode=checkpoint_episode)
    agent.load(agent_ckp)
    training_env = Environment(env)
    training_env.load(env_ckp)
    trainer = AgentTrainer.from_checkpoint(agent, training_env, training_ckp)
    eval_episodes = training_config_reader.get_param("episodes.eval_episodes", v_type=int)

    trainer.plot_training_history(block=True)
    print(f"Evaluating agent over {eval_episodes} episodes")
    avg_return, avg_results = trainer.evaluate(eval_episodes, 'INFO', allowed_exceptions=(WeatherDataProviderError,))
    print("Average return: ", avg_return)
    print("Weight of storage organs of crop 1: ", avg_results["WSO1"], "g/m^2")
    print("Weight of storage organs of crop 2: ", avg_results["WSO2"], "g/m^2")
    print("Total fertilizer used: ", avg_results["NTOT"], "kg/ha")


if __name__ == '__main__':
    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
    main()
