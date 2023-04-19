import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class BaseAgent:
    """Base Agent class."""

    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        """
        Initialize the Agent.

        Args:
            action_space: The action space of the environment.
            observation_space: The observation space of the environment.
        """
        self.action_space = action_space
        self.observation_space = observation_space

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        Select an action based on the current observation.
        """
        return self.action_space.sample()

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        Learn from the given experience.
        """
        pass


class DQNetwork(nn.Module):
    """DQN"""

    def __init__(self, input_size, output_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class Agent(BaseAgent):
    """
    A deep Q-network (DQN) agent that learns to perform actions based on observations using a neural network.
    """

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: gym.spaces.Box,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        memory_size: int = 10000,
    ):
        super().__init__(action_space, observation_space)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        self.network = DQNetwork(
            self.observation_space.shape[0], self.action_space.n
        ).to(self.device)
        self.target_network = DQNetwork(
            self.observation_space.shape[0], self.action_space.n
        ).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self.last_observation = None
        self.last_action = None

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        Select an action based on the current observation using the DQN.

        Args:
            observation: The current observation from the environment.

        Returns:
            action: The chosen action.
        """
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        else:
            observation = torch.tensor([observation], dtype=torch.float32).to(
                self.device
            )
            with torch.no_grad():
                return int(torch.argmax(self.network(observation)).item())

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        Learn from the given experience using the DQN.

        Args:
            observation: The current observation from the environment.
            reward: The reward received after performing the action.
            terminated: Whether the episode has terminated.
            truncated: Whether the episode has been truncated.
        """
        if len(self.memory) < self.batch_size:
            return
        self.memory.append(
            (
                self.last_observation,
                self.last_action,
                reward,
                observation,
                terminated or truncated,
            )
        )
        self.last_observation = observation
        self.last_action = self.act(observation)

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        observations, actions, rewards, next_observations, dones = zip(*minibatch)
        observations = torch.tensor(np.stack(observations), dtype=torch.float32).to(
            self.device
        )
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = (
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        )
        next_observations = torch.tensor(
            np.stack(next_observations), dtype=torch.float32
        ).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.network(observations).gather(1, actions)
        next_q_values = (
            self.target_network(next_observations).detach().max(1)[0].unsqueeze(1)
        )
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)

        if terminated or truncated:
            self.target_network.load_state_dict(self.network.state_dict())
