# Regenerative Agricolture RL

#### This repository shows how Reinforcement Learning technologies can be used in intercropping to optimize the nitrogen usage in crop fields.
#### In particular, 3 agents have been implemented and compared: a SAC agent, PPO agent and a classical DQN agent.
## Installation
### Prerequisites
#### Ensure you have the following installed:
1. Python (>= 3.x)
2. pip
### Setup
#### Clone the repository:
`git clone https://github.com/federicomatarante/RegenerativeAgricoltureRL.git`

`cd yourproject`
#### Create a virtual environment (optional but recommended):
`python -m venv venv`

source `venv/bin/activate`  # On Windows use `venv\Scripts\activate`
### Install dependencies:
`pip install -r requirements.txt`
## Testing agents
### Test the DQN agent:
`python ./src/scripts/train_dqn_agent.py`
### Test the SAC agent:
`python ./src/scripts/train_sac_agent.py`
### Test the PPO agent:
`python ./src/scripts/train_ppo_agent.py`

## How to cite me
`@software{InterCropGym,
   author = {Sandri Leonardo and Matarante Federico},
   title = {RegenerativeAgricoltureRL: Intercropping expansion of CropGym environment for simulation with Reinforcement Learning techniques},
   year = {2024},
   url = {https://github.com/federicomatarante/RegenerativeAgricoltureRL},
   version = {1.0.0}
}`
