import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from RL-buffer import ReplayBuffer

class TD3:

    def __init__(

            self,

            state_dim,

            action_dim=1,

            max_action=0.6,

            gamma=0.99,

            tau=0.005,

            policy_noise=0.2,

            noise_clip=0.5,

            policy_delay=2

    ):

        self.actor = Actor(

            state_dim,

            action_dim,

            max_action

        ).to(device)

        self.actor_target = copy.deepcopy(self.actor)

        self.actor_optimizer = torch.optim.Adam(

            self.actor.parameters(),

            lr=1e-4

        )

        self.critic = TwinCritic(

            state_dim,

            action_dim

        ).to(device)

        self.critic_target = copy.deepcopy(self.critic)

        self.critic_optimizer = torch.optim.Adam(

            self.critic.parameters(),

            lr=1e-3

        )

        # hyperparameters

        self.gamma = gamma

        self.tau = tau

        self.policy_noise = policy_noise

        self.noise_clip = noise_clip

        self.policy_delay = policy_delay

        self.total_it = 0

    # Select action

    def select_action(

            self,

            state

    ):

        state = torch.FloatTensor(

            state.reshape(1, -1)

        ).to(device)

        action = self.actor(state)

        return action.cpu().data.numpy().flatten()

    # Train TD3

    def train(

            self,

            replay_buffer,

            batch_size=256

    ):

        self.total_it += 1

        # sample batch

        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(device)

        action = torch.FloatTensor(action).to(device)

        reward = torch.FloatTensor(reward).to(device).unsqueeze(1)

        next_state = torch.FloatTensor(next_state).to(device)

        done = torch.FloatTensor(done).to(device).unsqueeze(1)

        # Target policy smoothing

        noise = (

            torch.randn_like(action)

            * self.policy_noise

        ).clamp(

            -self.noise_clip,

            self.noise_clip

        )

        next_action = (

            self.actor_target(next_state)

            + noise

        ).clamp(

            -self.actor.actor[-1].out_features,

            self.actor.actor[-1].out_features

        )

        # Target Q computation

        target_Q1, target_Q2 = self.critic_target(

            next_state,

            next_action

        )

        target_Q = torch.min(

            target_Q1,

            target_Q2

        )

        target_Q = reward + (

            (1 - done)

            * self.gamma

            * target_Q

        )

        # Current Q estimate

        current_Q1, current_Q2 = self.critic(

            state,

            action

        )

        critic_loss = F.mse_loss(

            current_Q1,

            target_Q

        ) + F.mse_loss(

            current_Q2,

            target_Q

        )

        # update critic

        self.critic_optimizer.zero_grad()

        critic_loss.backward()

        self.critic_optimizer.step()

        # Delayed policy update

        if self.total_it % self.policy_delay == 0:

            actor_loss = -self.critic.Q1_value(

                state,

                self.actor(state)

            ).mean()

            self.actor_optimizer.zero_grad()

            actor_loss.backward()

            self.actor_optimizer.step()

            # soft update

            for param, target_param in zip(

                    self.actor.parameters(),

                    self.actor_target.parameters()

            ):

                target_param.data.copy_(

                    self.tau * param.data +

                    (1 - self.tau)

                    * target_param.data

                )

            for param, target_param in zip(

                    self.critic.parameters(),

                    self.critic_target.parameters()

            ):

                target_param.data.copy_(

                    self.tau * param.data +

                    (1 - self.tau)

                    * target_param.data

                )

    # Save model

    def save(self, filename):

        torch.save(

            self.actor.state_dict(),

            filename + "_actor.pth"

        )

        torch.save(

            self.critic.state_dict(),

            filename + "_critic.pth"

        )

    # Load model

    def load(self, filename):

        self.actor.load_state_dict(

            torch.load(filename + "_actor.pth")

        )

        self.critic.load_state_dict(

            torch.load(filename + "_critic.pth")

        )