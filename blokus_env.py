import numpy as np
from blokus import BlokusState, pieces, orientations, N, num_pieces

class BlokusEnv:
    def __init__(self, opponent_policy=None):
        self.state = None
        self.opponent_policy = opponent_policy
        self.action_list = self._build_action_list()
        self.action_size = len(self.action_list)
        self.observation_size = N * N + num_pieces
        self.current_player = 0  # 0: DQN, 1: Opponent

    def _build_action_list(self):
        # List all possible (piece, x, y, rot, refl) combinations
        action_list = []
        for piece_name in pieces:
            for orientation in orientations[piece_name]:
                angle = orientation[1]
                refl = orientation[2]
                for x in range(N):
                    for y in range(N):
                        action_list.append((piece_name, x, y, angle, refl))
        action_list.append("Pass")
        return action_list

    def encode_state(self, state):
        # Board: N x N, 0=empty, 1=player, -1=opponent
        board_enc = np.zeros((N, N), dtype=np.float32)
        for y in range(N):
            for x in range(N):
                if state.board[y][x] == ".":
                    board_enc[y, x] = 0
                elif state.board[y][x] == "\033[34m□\033[0m":
                    board_enc[y, x] = 1 if state.P == 0 else -1
                elif state.board[y][x] == "\033[31m□\033[0m":
                    board_enc[y, x] = -1 if state.P == 0 else 1
        # Hand: 1 if piece is in hand, 0 otherwise
        hand_enc = np.zeros(num_pieces, dtype=np.float32)
        for i, piece_name in enumerate(pieces):
            if piece_name in state.hands[state.P]:
                hand_enc[i] = 1
        return np.concatenate([board_enc.flatten(), hand_enc])

    def encode_action(self, action):
        # Returns index in action_list
        try:
            return self.action_list.index(action)
        except ValueError:
            return self.action_list.index("Pass")

    def decode_action(self, action_idx):
        return self.action_list[action_idx]

    def reset(self):
        self.state = BlokusState()
        self.current_player = 0
        return self.encode_state(self.state)

    def step(self, action_idx):
        action = self.decode_action(action_idx)
        valid_actions = self.state.get_actions()
        if valid_actions == "Pass":
            valid_actions = ["Pass"]
        pieces_before = len(self.state.hands[0])
        # If action is not valid, force Pass (shouldn't happen with masking)
        if action not in valid_actions:
            action = "Pass"
        self.state = self.state.successor(action)
        done = self.state.is_terminal()
        reward = 0
        info = {}
        pieces_after = len(self.state.hands[0])
        # Reward for placing a piece
        if pieces_after < pieces_before:
            reward += (pieces_before - pieces_after)
        # If not done, let opponent play
        if not done:
            if self.opponent_policy is not None:
                opp_action, _ = self.opponent_policy(self.state)
                self.state = self.state.successor(opp_action)
                done = self.state.is_terminal()
        # Final reward at end of game
        if done:
            my_score = self.state.get_score(0)
            opp_score = self.state.get_score(1)
            reward += -(my_score - opp_score)
        obs = self.encode_state(self.state)
        return obs, reward, done, info

    def get_valid_action_indices(self):
        valid_actions = self.state.get_actions()
        if valid_actions == "Pass":
            valid_actions = ["Pass"]
        return [self.action_list.index(a) for a in valid_actions]

    def render(self):
        self.state.display_board()