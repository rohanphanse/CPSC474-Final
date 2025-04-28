import random
import time
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from blokus import *

class MCTSNode():
    __slots__ = ("state", "parent", "action", "num_visits", "value", "children")

    def __init__(self, state, parent = None, action = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.num_visits = 0
        self.value = 0
        self.children = []

def build_mcts_tree(state, cpu_time_limit):
    start_time = time.perf_counter()
    root = MCTSNode(state)
    exploration_constant = 1.5
    while True:
        elapsed_time = time.perf_counter() - start_time
        if elapsed_time >= cpu_time_limit - 0.0001:
            break
        cur_node = root
        # Selection 
        while len(cur_node.children) > 0 and not cur_node.state.is_terminal():
            best_child = None
            best_ucb_score = None
            if cur_node.state.actor() == 0:
                for child in cur_node.children:
                    if child.num_visits == 0:
                        best_child = child
                        break
                    else:
                        ucb_score = child.value / child.num_visits + exploration_constant * math.sqrt(math.log(cur_node.num_visits) / child.num_visits)
                        if best_child is None or ucb_score > best_ucb_score:
                            best_child = child
                            best_ucb_score = ucb_score
            else:
                for child in cur_node.children:
                    if child.num_visits == 0:
                        best_child = child
                        break
                    else:
                        ucb_score = child.value / child.num_visits - exploration_constant * math.sqrt(math.log(cur_node.num_visits) / child.num_visits)
                        if best_child is None or ucb_score < best_ucb_score:
                            best_child = child
                            best_ucb_score = ucb_score
            cur_node = best_child
        # Expansion
        if not cur_node.state.is_terminal() and len(cur_node.children) == 0:
            # Create children all at once
            actions = cur_node.state.get_actions()
            if actions == "Pass":
                actions = ["Pass"]
            random.shuffle(actions)
            for action in actions:
                child_state = cur_node.state.successor(action)
                child_node = MCTSNode(child_state, cur_node, action)
                cur_node.children.append(child_node)
            # Choose random action
            cur_node = random.choice(cur_node.children)
        # Simulation
        cur_state = cur_node.state
        while not cur_state.is_terminal():
            actions = cur_state.get_actions()
            action = "Pass" if actions == "Pass" else random.choice(actions)
            cur_state = cur_state.successor(action)
        payoff =  cur_state.payoff()
        # Backpropogation
        while not cur_node is None:
            cur_node.num_visits += 1
            cur_node.value += payoff
            cur_node = cur_node.parent
    return root

def mcts_policy(cpu_time_limit):
    def policy(state):
        num_workers = multiprocessing.cpu_count()
        manager = multiprocessing.Manager()
        stop_event = manager.Event()
        start_time = time.perf_counter()
        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    build_mcts_tree,
                    state.copy(),
                    cpu_time_limit
                )
                for _ in range(num_workers)
            ]
            while True:
                elapsed = time.perf_counter() - start_time
                if elapsed >= cpu_time_limit:
                    stop_event.set()
                    break
                time.sleep(0.01)
            
            for future in as_completed(futures):
                results.append(future.result())
        merged_children = {}
        for tree in results:
            for child in tree.children:
                if child.action in merged_children:
                    merged_children[child.action]["visits"] += child.num_visits
                    merged_children[child.action]["value"] += child.value
                else:
                    merged_children[child.action] = {
                        "visits": child.num_visits,
                        "value": child.value
                    }
        top_actions = sorted(merged_children.items(), key=lambda x: x[1]["visits"], reverse=True)[:5]
        best = None
        best_action = None
        for action in merged_children:
            child = merged_children[action]
            if best is None or child["visits"] > best["visits"]:
                best = child
                best_action = action
        return best_action, top_actions
    return policy

def greedy_policy():
    def policy(state):
        actions = state.get_actions()
        if actions == "Pass":
            return "Pass", [(0, "Pass")]
        action_pairs = []
        cur_actions = state.successor("Pass").get_actions()
        for action in actions:
            new_state = state.successor(action)
            score = calculate_move_score(state, cur_actions, new_state.get_actions(), action)
            action_pairs.append((score, action))
        max_score = max(score for score, _ in action_pairs)
        best_pairs = [x for x in action_pairs if x[0] == max_score]
        top_actions = []
        if len(best_pairs) >= 5:
            top_actions = random.sample(best_pairs, k = 5)
        else:
            top_actions = sorted(action_pairs, reverse = True)[:5]
        best_action = top_actions[0][1]
        return best_action, top_actions
    return policy

def calculate_move_score(cur_state, cur_actions, new_actions, action):
    piece_name, x, y, _, _ = action
    score = 0
    piece_size = len(pieces[piece_name])
    score += piece_size * 2
    if (x == 0 or x == N - 1) and (y == 0 or y == N - 1):
        score += 10
    if len(cur_actions) < len(new_actions):
        score += 15
    if num_pieces - len(cur_state.hands[cur_state.P]) < 5:
        center_dist = (abs(x - N // 2) + abs(y - N // 2))
        score += (N // 2 - center_dist) * 0.5
    return score

if __name__ == "__main__":
    print("\033[2J\033[H", end="")
    print("# CPU cores:", multiprocessing.cpu_count())
    state = BlokusState() # Initial state
    mcts_player = mcts_policy(cpu_time_limit=1.0)
    greedy_player = greedy_policy()
    player_2 = "greedy"
    turn = 1
    while not state.is_terminal():
        action = None
        top_actions = None
        greedy = True
        if state.P == 0:
            if turn < 10:
                action, top_actions = greedy_player(state)
            else:
                action, top_actions = mcts_player(state)
                greedy = False
        elif player_2 == "greedy":
            action, top_actions = greedy_player(state)
        else:
            actions = state.get_actions()
            action = actions if actions == "Pass" else random.choice(actions)
        player = state.P
        state = state.successor(action)
        print("\033[2J\033[H", end="")
        print("Player #1 (Greedy Start + MCTS) vs. Player #2 (Greedy)")
        state.display_board(action, player)
        print(f"Turn {turn} - Player #{2 - state.P} chose: {action}")
        if top_actions:
            print("Top actions:")
            if not greedy:
                for i, top_action in enumerate(top_actions):
                    print(f"  {i + 1}.", top_action[0], " - visits:", top_action[1]["visits"])
            else:
                for i, top_action in enumerate(top_actions):
                    print(f"  {i + 1}.", top_action[1], "- score:", top_action[0])
        turn += 1
        input("Press enter to continue...")
    # print("Player #1 moves:", moves[0])
    # print("Player #2 moves:", moves[1])
    print("Player #1 remaining pieces:", state.hands[0])
    print("Player #2 remaining pieces:", state.hands[1])
    print("Player #1 score:", state.get_score(0))
    print("Player #2 score:", state.get_score(1))