from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.optim as optim

from agents.agent import Agent
from networks.ppo_networks import ActorNetwork, CriticNetwork
from buffers.ppo_buffer import PPOBuffer
from utils.config_reader import ConfigReader

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

        # Initialize networks
        self.actor = ActorNetwork(
            input_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=actor_hidden,
            activation=activation,
            device=device
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

    def act(self, state: np.ndarray, reward: Optional[float] = None, done: Optional[bool] = None, explore: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Select an action given the current state and optionally store transition in buffer.
        
        :param state: Current environment state
        :param reward: Reward from previous action (if not first step)
        :param done: Whether previous step terminated the episode
        :param explore: Whether to explore (ignored in PPO as it always samples from policy)
        :return: Tuple of (action, value, log_prob) as numpy arrays, where:
                - action is the selected action
                - value is the critic's value estimate
                - log_prob is the log probability of the selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob = self.actor.get_action_and_log_prob(state_tensor)
            value = self.critic(state_tensor)

            if reward is not None and done is not None:
                self.buffer.store(
                    state=state,
                    action=action.cpu().numpy(),
                    reward=reward,
                    value=value.cpu().numpy(),
                    log_prob=log_prob.cpu().numpy(),
                    done=done
                )

        return action.cpu().numpy(), value.cpu().numpy(), log_prob.cpu().numpy()
    
    def update(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> Dict[str, float]:
        """
        Check if episode is done and update networks.
        
        :param state: Current state (unused in PPO)
        :param action: Action taken (unused in PPO)
        :param reward: Reward received (unused in PPO)
        :param next_state: Next state (used for computing final values)
        :param done: Whether episode terminated
        :return: Dictionary of training metrics if update performed, empty dict otherwise
        """

        if done:
            metrics = self.update_networks(next_state)
            return metrics
        return {}
    
    def update_networks(self, final_state: np.ndarray) -> Dict[str, float]:
        """
        Update actor and critic networks using PPO algorithm.
        
        :param final_state: Final state of episode, used for computing final value
        :return: Dictionary of training metrics including:
                - policy_loss: Average policy loss across updates
                - value_loss: Average value function loss across updates
                - entropy: Average entropy of policy distribution
        """
        data = self.buffer.get()
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        # Perform multiple epochs of updates
        for _ in range(self.num_epochs):
            # Generate random permutation for minibatches
            indices = torch.randperm(len(self.buffer))

            # Update in minibatches
            for start in range(0, len(self.buffer), self.batch_size):
                batch_indices = indices[start:start + self.batch_size]

                # Get batch data
                states = data['states'][batch_indices]
                actions = data['actions'][batch_indices]
                old_values = data['values'][batch_indices]
                old_log_probs = data['log_probs'][batch_indices]
                rewards = data['rewards'][batch_indices]
                dones = data['dones'][batch_indices]

                # Get current policy distribution and value estimates
                distribution = self.actor.get_distribution(states)
                values = self.critic(states)
                log_probs = distribution.log_prob(actions)
                entropy = distribution.entropy().mean()

                # Calculate advantages and returns
                with torch.no_grad():
                    final_value = self.critic(torch.FloatTensor(final_state).unsqueeze(0).to(self.device))
                    next_values = torch.zeros_like(values)
                    next_values[:-1] = values[1:].clone()
                    next_values[-1] = final_value

                    # Calculate advantages
                    advantages = rewards + self.gamma * next_values * (1 - dones.float()) - values
                    returns = advantages + values

                # Calculate policy loss with clipping
                ratio = torch.exp(log_probs - old_log_probs)
                policy_loss_1 = ratio * advantages
                policy_loss_2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Calculate value loss
                value_loss = ((returns - values) ** 2).mean()

                # Calculate total loss
                loss = (policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy)

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
        
        # Clear buffer after update
        self.buffer.clear()

        # Return metrics
        num_updates = self.num_epochs * (len(self.buffer) // self.batch_size)
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }
    
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