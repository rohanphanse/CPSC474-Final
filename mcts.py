# Updated mcts.py with DQN integration

import random
import time
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from blokus import *
from greedy import *
from dqn_agent import DQNAgent, dqn_policy, pieces, orientations, N

# ---------- MCTS Components ----------

class MCTSNode():
    __slots__ = ("state", "parent", "action", "num_visits", "value", "children")

    def __init__(self, state, parent=None, action=None):
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
    while time.perf_counter() - start_time < cpu_time_limit - 0.0001:
        cur_node = root
        while len(cur_node.children) > 0 and not cur_node.state.is_terminal():
            best_child, best_ucb_score = None, None
            if cur_node.state.actor() == check:
                for child in cur_node.children:
                    if child.num_visits == 0:
                        best_child = child
                        break
                    ucb_score = child.value / child.num_visits + exploration_constant * math.sqrt(math.log(cur_node.num_visits) / child.num_visits)
                    if best_ucb_score is None or ucb_score > best_ucb_score:
                        best_child, best_ucb_score = child, ucb_score
            else:
                for child in cur_node.children:
                    if child.num_visits == 0:
                        best_child = child
                        break
                    ucb_score = child.value / child.num_visits - exploration_constant * math.sqrt(math.log(cur_node.num_visits) / child.num_visits)
                    if best_ucb_score is None or ucb_score < best_ucb_score:
                        best_child, best_ucb_score = child, ucb_score
            cur_node = best_child
        if not cur_node.state.is_terminal() and len(cur_node.children) == 0:
            actions = cur_node.state.get_actions()
            if actions == "Pass":
                actions = ["Pass"]
            random.shuffle(actions)
            for action in actions:
                child_state = cur_node.state.successor(action)
                cur_node.children.append(MCTSNode(child_state, cur_node, action))
            cur_node = random.choice(cur_node.children)
        cur_state = cur_node.state
        while not cur_state.is_terminal():
            action, _ = greedy_player(cur_state)
            cur_state = cur_state.successor(action)
        payoff = cur_state.payoff()
        if player_2:
            payoff = -payoff
        while cur_node is not None:
            cur_node.num_visits += 1
            cur_node.value += payoff
            cur_node = cur_node.parent
    return root

def mcts_policy(cpu_time_limit, player_2=False):
    def policy(state):
        num_workers = multiprocessing.cpu_count()
        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(build_mcts_tree, state.copy(), cpu_time_limit, player_2) for _ in range(num_workers)]
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
                    merged_children[child.action] = {"visits": child.num_visits, "value": child.value}
        top_actions = sorted(merged_children.items(), key=lambda x: x[1]["value"] / (x[1]["visits"] + 1e-6), reverse=True)[:10]
        best_action = top_actions[0][0]
        return best_action, top_actions, total_visits
    return policy

# ---------- Main Loop with DQN Integration ----------

if __name__ == "__main__":
    print("\033[2J\033[H", end="")
    print("No. of CPU cores:", multiprocessing.cpu_count())
    state = BlokusState()

    # DQN setup
    action_space = [("Pass",)] + [
        (p, x, y, rot, refl)
        for p in pieces
        for (coords, rot, refl) in orientations[p]
        for x in range(N) for y in range(N)
    ]
    dqn_agent = DQNAgent(action_space)

    # Load pre-trained model
    import torch
    dqn_agent.q_net.load_state_dict(torch.load("dqn_model.pt"))
    dqn_agent.target_q_net.load_state_dict(dqn_agent.q_net.state_dict())

    # Create policies
    dqn_policy_p1 = dqn_policy(dqn_agent)
    dqn_policy_p2 = dqn_policy(dqn_agent)

    mcts_player_1 = mcts_policy(cpu_time_limit=1.0)
    mcts_player_2 = mcts_policy(cpu_time_limit=1.0, player_2=True)
    greedy_player = greedy_policy()

    player_1 = input("Player #1 policy ('mcts', 'greedy', 'random', 'dqn'): ")
    player_2 = input("Player #2 policy ('mcts', 'greedy', 'random', 'dqn'): ")
    policy_names = { "mcts": "MCTS", "greedy": "Greedy", "random": "Random", "dqn": "DQN" }

    turn = 1
    while not state.is_terminal():
        action = None
        top_actions = None
        total_visits = None
        player = state.P
        player_policy = player_1 if player == 0 else player_2
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
        elif player_policy == "dqn":
            if player == 0:
                action, top_actions = dqn_policy_p1(state)
            else:
                action, top_actions = dqn_policy_p2(state)

        state = state.successor(action)
        print("\033[2J\033[H", end="")
        print(f"Player \033[34m#1\033[0m ({policy_names[player_1]}) vs. Player \033[31m#2\033[0m ({policy_names[player_2]})")
        state.display_board(action, player)
        print(f"Turn: {turn}")
        if isinstance(top_actions, list):
            for i, ta in enumerate(top_actions[:5]):
                print(f"  {i+1}. {ta}")
        turn += 1
        input("Press enter to continue...")

    print("Player #1 remaining pieces:", state.hands[0])
    print("Player #2 remaining pieces:", state.hands[1])
    print("Player #1 score:", state.get_score(0))
    print("Player #2 score:", state.get_score(1))
