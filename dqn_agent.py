import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

from blokus import BlokusState, pieces, orientations, N

# Hyperparameters
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 32
BUFFER_SIZE = 5000
MIN_REPLAY_SIZE = 1000
TARGET_UPDATE_FREQ = 500
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 10000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, board_size, num_pieces, action_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * board_size * board_size + num_pieces * 2 + 1, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x_img, x_hand):
        x = self.conv(x_img).view(x_img.size(0), -1)
        x = torch.cat([x, x_hand], dim=1)
        return self.fc(x)

def encode_state(state, action_list):
    board_tensor = np.zeros((3, N, N), dtype=np.float32)
    for y in range(N):
        for x in range(N):
            symbol = state.board[y][x]
            if symbol == ".":
                board_tensor[2, y, x] = 1
            elif symbol == "\033[34m□\033[0m":
                board_tensor[0, y, x] = 1
            elif symbol == "\033[31m□\033[0m":
                board_tensor[1, y, x] = 1

    hand_tensor = np.zeros(len(pieces) * 2 + 1, dtype=np.float32)
    for i, name in enumerate(pieces):
        hand_tensor[i] = name in state.hands[0]
        hand_tensor[i + len(pieces)] = name in state.hands[1]
    hand_tensor[-1] = state.P

    valid_mask = np.zeros(len(action_list), dtype=np.float32)
    legal_actions = state.get_actions()
    if legal_actions == "Pass":
        legal_actions = [("Pass",)]
    legal_set = set(legal_actions)
    for i, action in enumerate(action_list):
        if action in legal_set:
            valid_mask[i] = 1.0

    return (
        torch.tensor(board_tensor).unsqueeze(0).to(DEVICE),
        torch.tensor(hand_tensor).unsqueeze(0).to(DEVICE),
        torch.tensor(valid_mask).unsqueeze(0).to(DEVICE)
    )

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.q_net = QNetwork(N, len(pieces), len(action_space)).to(DEVICE)
        self.target_q_net = QNetwork(N, len(pieces), len(action_space)).to(DEVICE)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.steps = 0
        self.eps = EPS_START

    def select_action(self, state):
        self.steps += 1
        self.eps = max(EPS_END, EPS_START - self.steps / EPS_DECAY)

        if random.random() < self.eps:
            legal = state.get_actions()
            return random.choice(legal) if legal != "Pass" else "Pass"

        x_img, x_hand, mask = encode_state(state, self.action_space)
        q_values = self.q_net(x_img, x_hand)
        if mask.sum().item() == 0:
            print("WARNING: DQN found no legal actions. Defaulting to 'Pass'")
            return "Pass"
        masked_q = q_values.masked_fill(mask == 0, -1e9)
        best_idx = torch.argmax(masked_q, dim=1).item()
        return self.action_space[best_idx]

    def store_transition(self, transition):
        self.replay_buffer.add(transition)

    def train_step(self):
        if len(self.replay_buffer) < MIN_REPLAY_SIZE:
            return
        batch = self.replay_buffer.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        x_imgs, x_hands = [], []
        for s in states:
            img, hand, _ = encode_state(s, self.action_space)
            x_imgs.append(img)
            x_hands.append(hand)
        x_imgs = torch.cat(x_imgs)
        x_hands = torch.cat(x_hands)

        indices = [self.action_space.index(a) if a != "Pass" else 0 for a in actions]
        action_indices = torch.tensor(indices, device=DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(dones, dtype=torch.float32, device=DEVICE)

        next_qs = []
        for s in next_states:
            img, hand, mask = encode_state(s, self.action_space)
            q_vals = self.target_q_net(img, hand)
            next_qs.append(torch.max(q_vals.masked_fill(mask == 0, -1e9)).item())
        next_qs = torch.tensor(next_qs, dtype=torch.float32, device=DEVICE)

        pred_q = self.q_net(x_imgs, x_hands)[range(BATCH_SIZE), action_indices]
        target_q = rewards + (1 - dones) * GAMMA * next_qs

        loss = nn.MSELoss()(pred_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % TARGET_UPDATE_FREQ == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

def dqn_policy(agent: DQNAgent):
    def policy(state: BlokusState):
        action = agent.select_action(state)
        return action, [(None, action)]
    return policy
