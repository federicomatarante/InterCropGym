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
    with AgentTester(
        env_id='CartPole-v1',
        agent_type=PPOAgent,
        config_path=str(Path('data/configs/ppo.ini'))
    ) as tester:
        # Run test with rendering
        results = tester.run_test(
            num_episodes=5,
            render=True,
            verbose=True
        )

    # Other eventual tests

if __name__== "__main__":
    main()