import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from gym import Space
from gym.spaces import Discrete
from torch.optim import Adam

from src.agents.sac.model import GaussianPolicy, QNetwork, DeterministicPolicy
from src.agents.sac.replay_memory import ReplayMemory
from src.agents.sac.utils import soft_update, hard_update
from src.utils.configs.config_reader import ConfigReader


class DiscreteSAC(object):
    """
    Soft Actor-Critic (SAC) implementation with automatic temperature tuning.

    SAC is an off-policy maximum entropy deep reinforcement learning algorithm
    that provides a sample-efficient and stable way to learn policies in continuous action spaces.

    :param num_inputs: Number of input dimensions (state space size)
    :param action_space: Action space of the environment
    :param config_reader: Configuration containing hyperparameters:
        - gamma (float): Discount factor for future rewards (default: 0.99)
        - tau (float): Soft update coefficient for target networks (default: 0.005)
        - alpha (float): Temperature parameter for entropy term
        - policy_type (str): Type of policy ("Gaussian" or "Deterministic")
        - target_update_interval (int): Frequency of target network updates
        - automatic_entropy_tuning (bool): Whether to automatically tune entropy temperature
        - device (torch.device): Device to run computations on (CPU or CUDA)
    """

    def __init__(self, num_inputs: int, action_space: Discrete, config_reader: ConfigReader):
        self.grad_clip = config_reader.get_param("network.grad_clip", v_type=float)
        self.gamma = config_reader.get_param("sac.gamma", v_type=float)
        self.tau = config_reader.get_param("sac.tau", v_type=float)
        self.alpha = config_reader.get_param("sac.alpha", v_type=float)

        self.policy_type = config_reader.get_param("sac.policy", v_type=str, domain={'Gaussian', 'Deterministic'})
        self.target_update_interval = config_reader.get_param("sac.target_update_interval", v_type=int)
        self.automatic_entropy_tuning = config_reader.get_param("sac.automatic_entropy_tuning", v_type=bool)

        self.device = torch.device("cuda" if config_reader.get_param("device.cuda", v_type=bool) else "cpu")

        hidden_size = config_reader.get_param("network.hidden_size", v_type=int)
        lr = config_reader.get_param("network.lr", v_type=float)

        self.critic = QNetwork(num_inputs, action_space.n, hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        self.critic_target = QNetwork(num_inputs, action_space.n, hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        self._setup_policy(num_inputs, action_space, lr, hidden_size)

    def _setup_policy(self, num_inputs: int, action_space: Discrete, lr: float, hidden_size: int):

        if self.policy_type == "Gaussian":            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -float(np.log(action_space.n))
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=lr)

            self.policy = GaussianPolicy(num_inputs, action_space.n, hidden_size, action_space).to(
                self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.n, hidden_size, action_space).to(
                self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """
        Select an action given the current state.
        :param state: Current state observation
        :param evaluate: Whether to evaluate deterministically or sample from policy
        :return: Selected action
        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _, _ = self.policy.sample(state)

        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory: ReplayMemory, batch_size: int, updates: int) -> Tuple[
        float, float, float, float, float]:
        """
        Update the networks' parameters using batched experience from replay buffer.

        :param memory: Replay buffer containing transitions
        :param batch_size: Size of batch to sample from replay buffer
        :param updates: Current number of updates (used for target network updates)
        :return: Tuple of (Q1 loss, Q2 loss, policy loss, alpha loss, alpha value)
        """
        # Step 1: Sample experiences from replay buffer and convert to Tensor for effectiveness
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        # Step 2: Calculate the TD target for Q-function update
        with torch.no_grad():
            # Get next actions and their log probabilities from current policy
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            # Get Q-values for next state-action pairs from target networks
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            # Take minimum Q-value for each state-action to prevent overestimation
            # Subtract entropy term (alpha * log_prob) for maximum entropy objective
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            # Compute TD target: reward + gamma * (min_Q - alpha * log_prob)
            # mask_batch is 0 for terminal states, eliminating future reward term
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        # Step 3: Update Critics (Q-functions)
        # Get current Q-value estimates for state-action pairs
        # Two Q-functions to mitigate positive bias in the policy improvement step
        # Compute MSE loss between current Q-values and TD targets
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optim.step()
        # Step 4: Update Policy
        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        # Compute policy loss: maximize Q-value while maintaining entropy
        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.policy_optim.step()

        # Step 5: Update temperature parameter (alpha) if automatic tuning is enabled
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            torch.nn.utils.clip_grad_norm_([self.log_alpha], self.grad_clip)
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_checkpoint(self, ckpt_path: str | Path):
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
