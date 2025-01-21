from pathlib import Path
import sys
import os

# Add directory to python path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
sys.path.append(str(src_dir))

import torch
from agents.PPOAgent import PPOAgent
from trainings.AgentTester import AgentTester

def main():
    # Test PPO
    tester = AgentTester(
        env_id='CartPole-v1',
        agent_type=PPOAgent,
        config_path=str(Path('data/configs/ppo.ini'))
    )

    # First train the agent
    print("\nTraining phase:")
    train_results = tester.run_test(
        num_episodes=100,
        max_steps=200,
        render=False, # No render during training
        verbose=True
    )

    print("\nTraining results:")
    print(f"Average Training Reward: {train_results['avg_reward']:.2f}")
    print(f"Final Training Reward: {train_results['episode_results'][-1]['reward']:.2f}")

    # Then test the agent
    print("\nTesting phase:")
    test_results = tester.run_test(
        num_episodes=10,
        max_steps=200,
        render=True, # Render during testing
        verbose=True
    )

    print("\nTesting results:")
    print(f"Average Testing Reward: {test_results['avg_reward']:.2f}")
    print(f"Min/Max Testing Reward: {test_results['min_reward']:.2f}/{test_results['max_reward']:.2f}")

    # Other eventual tests

if __name__== "__main__":
    main()