import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, obs_size, action_size, device='cpu', gamma=0.99, lr=1e-3, batch_size=64, memory_size=10000):
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.model = DQN(obs_size, action_size).to(device)
        self.target_model = DQN(obs_size, action_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.action_size = action_size
        self.obs_size = obs_size
        self.steps = 0

    def select_action(self, state, valid_action_indices, epsilon=0.1):
        if random.random() < epsilon:
            return random.choice(valid_action_indices)
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state).cpu().numpy().flatten()
        # Mask invalid actions, might be "lazy" option
        mask = np.full_like(q_values, -np.inf)
        mask[valid_action_indices] = q_values[valid_action_indices]
        return int(np.argmax(mask))

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.model(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1, keepdim=True)[0]
            target = rewards + self.gamma * next_q_values * (1 - dones)
        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.update_target()