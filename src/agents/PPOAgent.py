from importlib.metadata import distribution
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.optim as optim
from pandas.io.stata import stata_epoch
from sympy.physics.quantum.density import entropy

from src.agents.agent import Agent
from src.buffers.ppo_buffer import PPOBuffer
from src.networks.ppo_networks import ActorNetwork, CriticNetwork
from src.utils.configs.config_reader import ConfigReader


class PPOAgent(Agent):
    """
    PPO Agent implementation.
    
    :param state_dim: Dimensions of state space
    :param action_dim: Number of possible actions
    :param config: Configuration reader instance
    :param device: Device to run the agent on

    Usage:
        config = ConfigReader(config_dict)
        agent = PPOAgent(
            state_dim=4,
            action_dim=2,
            config=config,
            device=torch.device('cuda')
        )
        
        # Training loop
        state = env.reset()
        # First step
        action, value, log_prob = agent.act(state)  # First step without storing
        
        for step in range(max_steps):
            next_state, reward, done, _ = env.step(action)
            # Get next action and store current transition
            next_action, next_value, next_log_prob = agent.act(next_state, reward, done)
            
            if done:
                metrics = agent.update(state, action, reward, next_state, done)
                state = env.reset()
            else:
                state = next_state
            action = next_action
            value = next_value
            log_prob = next_log_prob
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_val: int,
                 config: ConfigReader,
                 device: torch.device):
        super().__init__()
        self.device = device

        # Get network configurations
        actor_hidden = eval(config.get_param('network.actor_hidden_sizes'))
        critic_hidden = eval(config.get_param('network.critic_hidden_sizes'))
        activation = config.get_param('network.activation')
        lr = float(config.get_param('network.learning_rate'))

        # Get PPO parameters or default values
        self.gamma = float(config.get_param('ppo.gamma'))
        self.clip_range = float(config.get_param('ppo.clip_range'))
        self.ent_coef = float(config.get_param('ppo.ent_coef'))
        self.vf_coef = float(config.get_param('ppo.vf_coef'))
        self.max_grad_norm = float(config.get_param('ppo.max_grad_norm'))

        # Get training parameters
        buffer_size = int(config.get_param('training.buffer_size'))
        self.num_epochs = int(config.get_param('training.num_epochs'))
        self.batch_size = int(config.get_param('training.batch_size'))

        # Buffer handling
        self.minimum_required_samples = int(self.batch_size * 2) # threshold for buffer update

        # Initialize networks
        self.actor = ActorNetwork(
            input_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=actor_hidden,
            activation=activation,
            device=device,
            max_val=max_val
        )

        self.critic = CriticNetwork(
            input_dim=state_dim,
            hidden_sizes=critic_hidden,
            activation=activation,
            device=device
        )

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Initialize buffer
        self.buffer = PPOBuffer(
            size=buffer_size,
            state_dim=state_dim,
            device=device
        )

    def act(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """
        Select an action given the current state.
        
        :param state: Current environment state
        :param explore: Whether to explore (ignored in PPO as it always samples from policy)
        :return: Selected action as numpy array
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, _ = self.actor.get_action_and_log_prob(state_tensor)
            action_out = action if isinstance(action, (int, float)) else action.cpu().numpy()
            return action_out

    def update(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> Dict[
        str, float]:
        """
        Store transition in buffer and update networks if episode is done.
        
        :param state: Current state
        :param action: Action taken
        :param reward: Reward received
        :param next_state: Next state
        :param done: Whether episode terminated
        :return: Dictionary of training metrics if update performed, empty dict otherwise
        """
        # Store the transition inside the buffer
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            value = self.critic(state_tensor)
            # Convert action to tensor for log prob calculation
            action_tensor = torch.FloatTensor([action]) if isinstance(action, (int, float)) else torch.FloatTensor(action)
            action_tensor = action_tensor.to(self.device)
            _, log_prob = self.actor.get_action_and_log_prob(state_tensor)
            log_prob = log_prob.cpu().numpy().item()

            # Store transition in buffer
            self.buffer.store(
                state=state,
                action=action,
                reward=reward,
                value=value.cpu().numpy().item(),
                log_prob=log_prob,
                done=done
            )

        # If episode is done perform PPO update
        if done and len(self.buffer) >= self.minimum_required_samples:
            print(f"Episode ended with {len(self.buffer)} samples in buffer")
            metrics = self.update_networks(next_state)
            return metrics
        return {}

    def _perform_update(self, final_state: np.ndarray, current_batch_size: int) -> Dict[str, float]:
        """
        Performs a PPO update with the given batch size.

        :param final_state: Final state of episode for value estimation
        :param current_batch_size: Batch size to use for this update
        :return: Dictionary containing update metrics
        """
        data = self.buffer.get()
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        update_count = 0
        total_value_mean = 0
        total_value_std = 0
        total_advantage_mean = 0
        total_advantage_std = 0

        for _ in range(self.num_epochs):
            indices = torch.randperm(len(self.buffer))

            for start in range(0, len(self.buffer), current_batch_size):
                end = min(start + current_batch_size, len(self.buffer))
                if end - start < current_batch_size * 0.8:   # skip if batch size too small
                    continue

                update_count += 1
                batch_indices = indices[start:end]

                # Get batch data
                states = data['states'][batch_indices]
                actions = data['actions'][batch_indices]
                old_values = data['values'][batch_indices]
                old_log_probs = data['log_probs'][batch_indices]
                rewards = data['rewards'][batch_indices]
                dones = data['dones'][batch_indices]

                # Get current policy distribution and value estimates
                action_distribution = self.actor.get_distribution(states)
                values = self.critic(states)
                log_probs = action_distribution.log_prob(actions)
                entropy = action_distribution.entropy().mean()
                # probs_manual = action_distribution.probs # TODO remove these lines when entropy manual not needed anymore
                # entropy_manual = -torch.sum(probs_manual * torch.log(probs_manual + 1e-8), dim=-1).mean()
                # print(entropy_manual)

                # Calculate advantages and returns
                with torch.no_grad():
                    final_value = self.critic(torch.FloatTensor(final_state).unsqueeze(0).to(self.device))
                    next_values = torch.zeros_like(values)
                    next_values[:-1] = values[1:].clone()
                    next_values[-1] = final_value

                    # Calculate advantages
                    advantages, returns = self.buffer.compute_gae(
                        rewards=rewards.cpu().numpy(),
                        values=values.cpu().numpy(),
                        dones=dones.cpu().numpy(),
                        next_value=final_value.item(),
                        gamma=self.gamma
                    )
                    advantages = torch.FloatTensor(advantages).to(self.device)
                    returns = torch.FloatTensor(returns).to(self.device)

                    # Track advantage statistics
                    total_advantage_mean += advantages.mean().item()
                    total_advantage_std += advantages.std().item()
                    total_value_mean += values.mean().item()
                    total_value_std += values.std().item()

                # Calculate policy loss with clipping
                ratio = torch.exp(log_probs - old_log_probs)
                policy_loss_1 = ratio * advantages
                policy_loss_2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Calculate value loss
                value_loss = ((returns - values) ** 2).mean()

                # Calculate total loss
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                # Optimize
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()

            if update_count == 0:
                return {
                    'policy_loss': 0.0,
                    'value_loss': 0.0,
                    'entropy': 0.0,
                    'batch_size_used': current_batch_size,
                    'updates_performed': 0,
                    'value_mean': 0.0,
                    'value_std': 0.0,
                    'advantage_mean': 0.0,
                    'advantage_std': 0.0,
                    'learning_rate': self.actor_optimizer.param_groups[0]['lr']
                }

            return {
                'policy_loss': total_policy_loss / update_count,
                'value_loss': total_value_loss / update_count,
                'entropy': total_entropy / update_count,
                'batch_size_used': current_batch_size,
                'updates_performed': update_count,
                'value_mean': total_value_mean / update_count,
                'value_std': total_value_std / update_count,
                'advantage_mean': total_value_mean / update_count,
                'advantage_std': total_advantage_std / update_count,
                'learning_rate': self.actor_optimizer.param_groups[0]['lr']
            }

    def update_networks(self, final_state: np.ndarray) -> Dict[str, float]:
        if len(self.buffer) < self.minimum_required_samples:
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
                'skipped_reason': 'insufficient_data',
                'available_samples': len(self.buffer)
            }

        adjusted_batch_size = min(self.batch_size, len(self.buffer))
        metrics = self._perform_update(final_state, adjusted_batch_size)
        self.buffer.clear()
        return metrics

    def save(self, path: str) -> None:
        """
        Save agent networks.

        :param path: Path to save the model
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, path)

    def load(self, path: str) -> None:
        """
        Load agent networks.
        
        :param path: Path to load the model
        """
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
