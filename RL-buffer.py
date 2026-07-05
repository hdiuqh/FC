import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"

)

# Replay Buffer

class ReplayBuffer:

    def __init__(

            self,

            capacity=int(1e6)

    ):

        self.capacity = capacity

        self.buffer = []

        self.position = 0

    def push(

            self,

            state,

            action,

            reward,

            next_state,

            done

    ):

        if len(self.buffer) < self.capacity:

            self.buffer.append(None)

        self.buffer[self.position] = (

            state,

            action,

            reward,

            next_state,

            done

        )

        self.position = (

            self.position + 1

        ) % self.capacity

    def sample(

            self,

            batch_size

    ):

        batch = random.sample(

            self.buffer,

            batch_size

        )

        state,

        action,

        reward,

        next_state,

        done = map(

            np.stack,

            zip(*batch)

        )

        return (

            state,

            action,

            reward,

            next_state,

            done

        )

    def __len__(self):

        return len(self.buffer)


# Actor Network

class Actor(nn.Module):

    def __init__(

            self,

            state_dim,

            action_dim,

            max_action

    ):

        super().__init__()

        self.max_action = max_action

        self.net = nn.Sequential(

            nn.Linear(

                state_dim,

                256

            ),

            nn.ReLU(),

            nn.Linear(

                256,

                256

            ),

            nn.ReLU(),

            nn.Linear(

                256,

                128

            ),

            nn.ReLU(),

            nn.Linear(

                128,

                action_dim

            ),

            nn.Tanh()

        )

    def forward(

            self,

            state

    ):

        action = self.net(state)

        action = (

            action

            * self.max_action

        )

        return action

# Critic Network

class Critic(nn.Module):

    def __init__(

            self,

            state_dim,

            action_dim

    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(

                state_dim +

                action_dim,

                256

            ),

            nn.ReLU(),

            nn.Linear(

                256,

                256

            ),

            nn.ReLU(),

            nn.Linear(

                256,

                128

            ),

            nn.ReLU(),

            nn.Linear(

                128,

                1

            )

        )

    def forward(

            self,

            state,

            action

    ):

        x = torch.cat(

            [

                state,

                action

            ],

            dim=1

        )

        q = self.net(x)

        return q

# Twin Critic

class TwinCritic(nn.Module):

    def __init__(

            self,

            state_dim,

            action_dim

    ):

        super().__init__()

        self.Q1 = Critic(

            state_dim,

            action_dim

        )

        self.Q2 = Critic(

            state_dim,

            action_dim

        )

    def forward(

            self,

            state,

            action

    ):

        q1 = self.Q1(

            state,

            action

        )

        q2 = self.Q2(

            state,

            action

        )

        return q1, q2

    def Q1_value(

            self,

            state,

            action

    ):

        return self.Q1(

            state,

            action

        )