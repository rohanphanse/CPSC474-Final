# gym_blokus.py

import gym
from gym import spaces
import numpy as np

from blokus import BlokusState, pieces, orientations, N

class BlokusEnv(gym.Env):
    def __init__(self):
        super(BlokusEnv, self).__init__()

        self.state = BlokusState()
        self.player = 0  # DQN agent is always player 0
        self.opponent_policy = None  # Fill this with a callable if needed

        # Define the full action space
        self.actions = [("Pass",)] + [
            (p, x, y, rot, refl)
            for p in pieces
            for (coords, rot, refl) in orientations[p]
            for x in range(N)
            for y in range(N)
        ]
        self.action_space = spaces.Discrete(len(self.actions))

        # Board: 3 channels (P1, P2, empty), plus hand vector + player indicator
        board_shape = (3, N, N)
        hand_vec_len = len(pieces) * 2 + 1
        obs_shape = (np.prod(board_shape) + hand_vec_len, )
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=obs_shape, dtype=np.float32)

    def reset(self):
        self.state = BlokusState()
        self.player = 0
        return self._get_obs()

    def _get_obs(self):
        board_tensor = np.zeros((3, N, N), dtype=np.float32)
        for y in range(N):
            for x in range(N):
                symbol = self.state.board[y][x]
                if symbol == ".":
                    board_tensor[2, y, x] = 1
                elif symbol == "\033[34m□\033[0m":
                    board_tensor[0, y, x] = 1
                elif symbol == "\033[31m□\033[0m":
                    board_tensor[1, y, x] = 1

        hand_tensor = np.zeros(len(pieces) * 2 + 1, dtype=np.float32)
        for i, name in enumerate(pieces):
            hand_tensor[i] = name in self.state.hands[0]
            hand_tensor[i + len(pieces)] = name in self.state.hands[1]
        hand_tensor[-1] = self.state.P

        return np.concatenate([board_tensor.flatten(), hand_tensor], axis=0)

    def step(self, action_idx):
        action = self.actions[action_idx]
        legal_actions = self.state.get_actions()
        if legal_actions == "Pass":
            legal_actions = [("Pass",)]

        # If chosen action isn't legal, replace with random legal one
        if action not in legal_actions:
            action = legal_actions[0]

        old_state = self.state
        self.state = self.state.successor(action)
        done = self.state.is_terminal()
        reward = 0.0

        if done:
            payoff = self.state.payoff()
            reward = payoff if self.player == 0 else -payoff

        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        self.state.display_board()

    def legal_action_mask(self):
        mask = np.zeros(len(self.actions), dtype=np.float32)
        legal_actions = self.state.get_actions()
        if legal_actions == "Pass":
            legal_actions = [("Pass",)]
        legal_set = set(legal_actions)
        for i, act in enumerate(self.actions):
            if act in legal_set:
                mask[i] = 1.0
        return mask

