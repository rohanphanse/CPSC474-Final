import random
import time
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from blokus import *
from greedy import *

class MCTSNode():
    __slots__ = ("state", "parent", "action", "num_visits", "value", "children")

    def __init__(self, state, parent = None, action = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.num_visits = 0
        self.value = 0
        self.children = []

def build_mcts_tree(state, cpu_time_limit, player_2):
    start_time = time.perf_counter()
    root = MCTSNode(state)
    exploration_constant = 1.4
    greedy_player = greedy_policy()
    check = 1 if player_2 else 0
    while True:
        elapsed_time = time.perf_counter() - start_time
        if elapsed_time >= cpu_time_limit - 0.0001:
            break
        cur_node = root
        # Selection 
        while len(cur_node.children) > 0 and not cur_node.state.is_terminal():
            best_child = None
            best_ucb_score = None
            if cur_node.state.actor() == check:
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
            action, _ = greedy_player(cur_state)
            cur_state = cur_state.successor(action)
        payoff = cur_state.payoff()
        if player_2:
            payoff = -payoff
        # Backpropogation
        while not cur_node is None:
            cur_node.num_visits += 1
            cur_node.value += payoff
            cur_node = cur_node.parent
    return root

def mcts_policy(cpu_time_limit, player_2 = False):
    def policy(state):
        num_workers = multiprocessing.cpu_count()
        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    build_mcts_tree,
                    state.copy(),
                    cpu_time_limit,
                    player_2
                )
                for _ in range(num_workers)
            ]
            for future in as_completed(futures):
                results.append(future.result())
        merged_children = {}
        total_visits = 0
        for tree in results:
            for child in tree.children:
                total_visits += child.num_visits
                if child.action in merged_children:
                    merged_children[child.action]["visits"] += child.num_visits
                    merged_children[child.action]["value"] += child.value
                else:
                    merged_children[child.action] = {
                        "visits": child.num_visits,
                        "value": child.value
                    }
        top_actions = sorted(merged_children.items(), key=lambda x: x[1]["value"] / (x[1]["visits"] + 1e-6), reverse = True)[:10]
        best_action = top_actions[0][0]
        return best_action, top_actions, total_visits
    return policy


if __name__ == "__main__":
    print("\033[2J\033[H", end="")
    print("No. of CPU cores:", multiprocessing.cpu_count())
    state = BlokusState() # Initial state
    mcts_player_1 = mcts_policy(cpu_time_limit=1.0)
    mcts_player_2 = mcts_policy(cpu_time_limit=1.0, player_2=True)
    greedy_player = greedy_policy()
    player_1 = input("Player #1 policy ('mcts', 'greedy', 'random'): ")
    player_2 = input("Player #2 policy ('mcts', 'greedy', 'random'): ")
    policy_names = { "mcts": "MCTS", "greedy": "Greedy", "random": "Random" }
    turn = 1
    while not state.is_terminal():
        action = None
        top_actions = None
        greedy = True
        total_visits = None
        player = state.P
        player_policy = player_1 if state.P == 0 else player_2
        if player_policy == "mcts":
            if player == 0:
                action, top_actions, total_visits = mcts_player_1(state)
            else:
                action, top_actions, total_visits = mcts_player_2(state)
        elif player_policy == "greedy":
            action, top_actions = greedy_player(state)
        elif player_policy == "random":
            actions = state.get_actions()
            action = actions if actions == "Pass" else random.choice(actions)
        state = state.successor(action)
        print("\033[2J\033[H", end="")
        print(f"Player \033[34m#1\033[0m ({policy_names[player_1]}) vs. Player \033[31m#2\033[0m ({policy_names[player_2]})")
        state.display_board(action, player)
        if player_policy == "mcts":
            print(f"Top actions (turn: {turn}, total visits: {total_visits}):")
            for i, top_action in enumerate(top_actions):
                prefix = f"\033[1;97;4{4 if player == 0 else 1}m" if i == 0 else ""
                print(f"  {prefix}{i + 1}.", top_action[0], " - avg. reward:", round(top_action[1]["value"] / (top_action[1]["visits"] + 1e-6), 5), "and visits:", top_action[1]["visits"], "\033[0m")
        elif player_policy == "greedy":
            print(f"Top actions (turn: {turn}):")
            for i, top_action in enumerate(top_actions):
                prefix = f"\033[1;97;4{4 if player == 0 else 1}m" if top_action[1] == action else ""
                if top_action[1] == "Pass":
                    print(f"  {prefix}{i + 1}.", top_action[1], "\033[0m")
                else:
                    print(f"  {prefix}{i + 1}.", top_action[1], "- score:", top_action[0], "\033[0m")
        elif player_policy == "random":
            print(f"Top actions (turn: {turn}):")
            print(f"  \033[1;97;4{4 if player == 0 else 1}m1. {action}\033[0m")
        turn += 1
        input("Press enter to continue...")
    print("Player #1 remaining pieces:", state.hands[0])
    print("Player #2 remaining pieces:", state.hands[1])
    print("Player #1 score:", state.get_score(0))
    print("Player #2 score:", state.get_score(1))