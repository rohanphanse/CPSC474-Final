import random
from blokus import *

def greedy_policy():
    def policy(state):
        actions = state.get_actions()
        if actions == "Pass":
            return "Pass", [(0, "Pass")]
        action_pairs = []
        for action in actions:
            score = calculate_move_score(state, action)
            action_pairs.append((score, action))
        top_actions = sorted(action_pairs, reverse=True)[:10]
        scores = [score for score, _ in top_actions]
        min_score = min(scores)
        adjusted_scores = [score - min_score + 1 for score in scores]
        total = sum(adjusted_scores)
        weights = [score / total for score in adjusted_scores]
        best_action = random.choices(top_actions, weights=weights, k=1)[0][1]
        return best_action, top_actions
    return policy

def calculate_move_score(cur_state, action):
    piece_name, x, y, _, _ = action
    score = 0
    piece_size = len(pieces[piece_name])
    score += piece_size * 2
    opponent = 1 - cur_state.P
    new_state = cur_state.successor(action)
    # Prioritize blocking the opponent
    blocking_score = calculate_blocking_score(new_state, action, players[opponent])
    score += blocking_score
    # Move toward the center in the early game
    if num_pieces - len(cur_state.hands[cur_state.P]) < 5:
        center_dist = (abs(x - N // 2) + abs(y - N // 2))
        score += N // 2 - center_dist
    return score

def calculate_blocking_score(state, action, opp):
    piece_name, x, y, rot, refl = action
    y = N - 1 - y
    piece = state.get_piece(piece_name, rot, refl)
    score = 0
    seen = set()
    # Check if move was directly adjacent (edge or corner) to an opponent piece 
    for dx, dy in piece:
        piece_x = x + dx
        piece_y = y - dy
        for check_dx, check_dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            check_x = piece_x + check_dx
            check_y = piece_y + check_dy
            if (not (check_x, check_y) in seen and 0 <= check_x < N and 0 <= check_y < N and state.board[check_y][check_x] == opp):
                if check_dx == 0 or check_dy == 0:
                    score += 1 # Edge
                else:
                    score += 2 # Corner
                seen.add((check_x, check_y))
    seen = set()
    # Protect against potential corner attacks
    for dx, dy in piece:
        piece_x = x + dx
        piece_y = y - dy
        for corner_dx, corner_dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
            corner_x = piece_x + corner_dx
            corner_y = piece_y + corner_dy
            if (not (corner_x, corner_y) in seen and 0 <= corner_x < N and 0 <= corner_y < N and state.board[corner_y][corner_x] == "."):
                for opp_dx, opp_dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                    opp_x = corner_x + opp_dx
                    opp_y = corner_y + opp_dy
                    if (0 <= opp_x < N and 0 <= opp_y < N and state.board[opp_y][opp_x] == opp):
                        score += 2
                        seen.add((corner_x, corner_y))
                        break
    return score