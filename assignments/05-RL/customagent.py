import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, List


class QNetwork(nn.Module):
    """Deep Q-Network for approximating action-value function."""

    def __init__(self, input_size, output_size, hidden_size=32):
        """
        Initialize the Q-Network.

        Args:
            input_size: The size of the input state.
            output_size: The number of actions in the action space.
            hidden_size: The size of the hidden layers (default: 32).
        """
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x) -> torch.Tensor:
        """Forward pass to compute Q-values for the given state."""
        return self.network(x)


class ReplayBuffer:
    """Experience Replay Buffer for storing and sampling experiences."""

    def __init__(self, capacity):
        """
        Initialize the ReplayBuffer.

        Args:
            capacity: The maximum number of experiences to store in the buffer.
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        """
        Add an experience to the buffer.

        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The next state.
            done: Whether the episode has terminated.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size) -> List[Tuple]:
        """
        Sample a batch of experiences from the buffer.

        Args:
            batch_size: The number of experiences to sample.

        Returns:
            A batch of experiences.
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)


class Agent:
    """Deep Q-Learning Agent."""

    def __init__(self, action_space: gym.Space, observation_space: gym.Space):
        """
        Initialize the DQNAgent.

        Args:
            action_space: The action space of the environment.
            observation_space: The observation space of the environment.
        """
        self.action_space = action_space
        self.observation_space = observation_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.update_frequency = 8
        self.target_update_frequency = 100

        self.epsilon = self.epsilon_start
        self.steps = 0

        input_size = observation_space.shape[0]
        output_size = action_space.n

        self.q_network = QNetwork(input_size, output_size).to(self.device)
        self.target_network = QNetwork(input_size, output_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.005)

        self.replay_buffer = ReplayBuffer(5000)

        self.last_state = None
        self.last_action = None

    def act(self, observation: gym.Space) -> gym.Space:
        """
        Select an action based on the current observation using an epsilon-greedy policy.

        Args:
            observation: The current state of the environment.

        Returns:
            The selected action.
        """
        if np.random.rand() < self.epsilon:
            action = self.action_space.sample()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(observation).to(self.device)
                q_values = self.q_network(state)
                action = int(torch.argmax(q_values).item())

        self.last_state = observation
        self.last_action = action

        return action

    def learn(
        self, observation: gym.Space, reward: float, terminated: bool, truncated: bool
    ) -> None:
        """
        Learn from the given experience.

        Args:
            observation: The current state of the environment.
            reward: The reward received.
            terminated: Whether the episode has terminated.
            truncated: Whether the episode was truncated.
        """
        if self.last_state is None or self.last_action is None:
            return

        self.steps += 1
        self.replay_buffer.push(
            self.last_state, self.last_action, reward, observation, terminated
        )

        if (
            len(self.replay_buffer) < self.batch_size
            or self.steps % self.update_frequency != 0
        ):
            return

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)


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
