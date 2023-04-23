import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import List
import time


class PIDController:
    """
    A PID (Proportional-Integral-Derivative) controller class for smooth control.
    """

    def __init__(self, kp: float, ki: float, kd: float):
        """
        Initialize the PID controller with given gain parameters.
        Args:
            kp (float): Proportional gain
            ki (float): Integral gain
            kd (float): Derivative gain
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def control(self, error: float, dt: float) -> float:
        """
        Calculate control output based on given error and time step.
        Args:
            error (float): The current error value
            dt (float): The time step between successive control updates

        Returns:
            float: The control output value
        """
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output


class Agent:
    """
    Agent
    """

    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.prev_time = time.time()
        self.angle_controller = PIDController(kp=0.5, ki=0.1, kd=0.2)
        self.hover_controller = PIDController(kp=0.5, ki=0.1, kd=0.2)

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        Act
        """
        curr_time = time.time()
        time_step = curr_time - self.prev_time
        self.prev_time = curr_time

        T = 57 / 100

        angle_threshold = T * (1 - np.abs(observation[0]))
        target_angle = observation[0] * 0.6 + observation[2] * 1.0
        if target_angle > angle_threshold:
            target_angle = angle_threshold
        if target_angle < -angle_threshold:
            target_angle = -angle_threshold
        target_hover = 0.5 * np.abs(observation[0])

        angle_action = self.angle_controller.control(
            target_angle - observation[4], time_step
        )
        hover_action = self.hover_controller.control(
            target_hover - observation[1], time_step
        )

        if observation[6] or observation[7]:
            angle_action = 0
            hover_action = -(observation[3]) * 0.5

        action = 0
        if hover_action > np.abs(angle_action) and hover_action > 0.05:
            action = 2
        elif angle_action < -0.05:
            action = 3
        elif angle_action > +0.05:
            action = 1
        return action

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


class QAgent:
    """Agent"""

    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.num_actions = action_space.n
        self.num_bins = 10
        self.bins = self.create_bins()
        self.q_table = np.zeros([*map(len, self.bins), self.num_actions])
        self.alpha = 0.3
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def create_bins(self) -> List:
        """Create bins"""
        bins = []
        for low, high in zip(self.observation_space.low, self.observation_space.high):
            bins.append(np.linspace(low, high, self.num_bins - 1))
        return bins

    def discretize_state(self, state: np.ndarray) -> tuple:
        """Discretize"""
        discretized_state = []
        for value, bin_edges in zip(state, self.bins):
            index = np.digitize(value, bin_edges)
            index = np.clip(index, 0, len(bin_edges) - 1)
            discretized_state.append(index)
        return tuple(discretized_state)

    def act(self, observation: gym.spaces.Box) -> int:
        """Act"""
        discretized_state = self.discretize_state(observation)
        if np.random.rand() <= self.epsilon:
            return self.action_space.sample()
        return np.argmax(self.q_table[discretized_state])

    def learn(
        self, state: np.ndarray, reward: float, terminated: bool, truncated: bool
    ) -> None:
        """Learn"""
        action = self.act(state)
        next_state = state.copy()
        next_state[0] += state[2]
        next_state[1] += state[3]
        done = terminated or truncated
        discretized_state = self.discretize_state(state)
        discretized_next_state = self.discretize_state(next_state)
        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[discretized_next_state])
        self.q_table[discretized_state + (action,)] += self.alpha * (
            target - self.q_table[discretized_state + (action,)]
        )

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


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


class DQNAgent(BaseAgent):
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
        batch_size: int = 4,
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
