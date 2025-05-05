import random
import time
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import torch
import numpy as np

from blokus import *
from greedy import *
from blokus_env import BlokusEnv
from dqn_agent import DQNAgent

class MCTSNode():
    __slots__ = ("state", "parent", "action", "num_visits", "value", "children")

    def __init__(self, state, parent = None, action = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.num_visits = 0
        self.value = 0
        self.children = []

def build_mcts_tree(state, cpu_time_limit, player_2, dqn_model_path=None, dqn_weight=0.5):
    dqn_model = None
    if dqn_model_path:
        env = BlokusEnv()
        dqn_model = DQNAgent(env.observation_size, env.action_size)
        dqn_model.load(dqn_model_path)
    start_time = time.perf_counter()
    root = MCTSNode(state)
    exploration_constant = 1.4
    greedy_player = greedy_policy()
    check = 1 if player_2 else 0
    
    # Convert state to tensor for DQN if model is provided
    def get_q_value(state, action):
        if dqn_model is None:
            return 0
        env = BlokusEnv()  # Create temporary env for state encoding
        obs = env.encode_state(state)
        state_tensor = torch.tensor(obs, dtype=torch.float32, device=dqn_model.device).unsqueeze(0)
        with torch.no_grad():
            q_values = dqn_model.model(state_tensor).cpu().numpy().flatten()
            # Get q-value for the specific action
            action_idx = env.encode_action(action)
            return q_values[action_idx] if action_idx is not None else 0

    while True:
        elapsed_time = time.perf_counter() - start_time
        if elapsed_time >= cpu_time_limit - 0.0001:
            break
        cur_node = root
        # Selection 
        while len(cur_node.children) > 0 and not cur_node.state.is_terminal():
            best_child = None
            best_score = None
            if cur_node.state.actor() == check:
                for child in cur_node.children:
                    if child.num_visits == 0:
                        best_child = child
                        break
                    else:
                        # Combine UCB with Q-value
                        ucb_score = child.value / child.num_visits + exploration_constant * math.sqrt(math.log(cur_node.num_visits) / child.num_visits)
                        q_value = get_q_value(cur_node.state, child.action)
                        combined_score = ucb_score + dqn_weight * q_value
                        if best_child is None or combined_score > best_score:
                            best_child = child
                            best_score = combined_score
            else:
                for child in cur_node.children:
                    if child.num_visits == 0:
                        best_child = child
                        break
                    else:
                        # Combine UCB with Q-value (negative for opponent)
                        ucb_score = child.value / child.num_visits - exploration_constant * math.sqrt(math.log(cur_node.num_visits) / child.num_visits)
                        q_value = get_q_value(cur_node.state, child.action)
                        combined_score = ucb_score - dqn_weight * q_value
                        if best_child is None or combined_score < best_score:
                            best_child = child
                            best_score = combined_score
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
            # cur_actions = cur_state.get_actions()
            # action = cur_actions if cur_actions == "Pass" else random.choice(cur_actions)
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

def mcts_policy(cpu_time_limit, player_2=False, dqn_model_path=None, dqn_weight=0.5):
    def policy(state):
        num_workers = multiprocessing.cpu_count()
        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    build_mcts_tree,
                    state.copy(),
                    cpu_time_limit,
                    player_2,
                    dqn_model_path,
                    dqn_weight
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

def dqn_policy(model_path='dqn_blokus_random.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from greedy import greedy_policy
    greedy = greedy_policy()
    env = BlokusEnv(opponent_policy=greedy)
    agent = DQNAgent(env.observation_size, env.action_size, device=device)
    agent.load(model_path)
    def policy(state):
        # Set the environment's state to the current state
        env.state = state.copy()
        obs = env.encode_state(state)
        valid_action_indices = env.get_valid_action_indices()
        action_idx = agent.select_action(obs, valid_action_indices, epsilon=0.0)
        action = env.decode_action(action_idx)
        return action, []
    return policy

def parse_args():
    parser = argparse.ArgumentParser(description="Blokus Duo Evaluation")
    parser.add_argument("--player_1", choices=["mcts", "greedy", "random", "dqn", "mcts_dqn"], help="Policy for Player #1")
    parser.add_argument("--player_2", choices=["mcts", "greedy", "random", "dqn", "mcts_dqn"], help="Policy for Player #2")
    parser.add_argument("--dqn_model_path", type=str, default=None, choices=["dqn_blokus_random.pth", "dqn_blokus.pth"], help="Path to DQN model")
    parser.add_argument("--dqn_weight", type=float, default=0.5, help="DQN weight for MCTS+DQN (0.0 to 1.0)")
    parser.add_argument("--results_path", type=str, help="Path to store evaluation results")
    parser.add_argument("--num_games", type=int, help="Number of games to evaluate")
    parser.add_argument("--display", action="store_true", help="Display board?")
    args = parser.parse_args()
    return args.player_1, args.player_2, args.dqn_model_path, args.dqn_weight, args.results_path, args.num_games, args.display

def run_demo(player_1, player_2, dqn_model_path=None, dqn_weight=0.5):
    print("No. of CPU cores:", multiprocessing.cpu_count())
    state = None
    time_limit = 1.0
    mcts_player_1 = mcts_policy(cpu_time_limit=time_limit)
    mcts_player_2 = mcts_policy(cpu_time_limit=time_limit, player_2=True)
    greedy_player = greedy_policy()
    display = True
    policy_names = {
        "mcts": "MCTS",
        "greedy": "Greedy",
        "random": "Random",
        "dqn": "DQN",
        "mcts_dqn": "MCTS+DQN"
    }
    # Create appropriate policy instances
    if player_1 == 'mcts_dqn':
        mcts_player_1 = mcts_policy(cpu_time_limit=time_limit, dqn_model_path=dqn_model_path, dqn_weight=dqn_weight)
    elif player_1 == 'dqn':
        dqn_player_1 = dqn_policy(dqn_model_path)
    
    if player_2 == 'mcts_dqn':
        mcts_player_2 = mcts_policy(cpu_time_limit=time_limit, player_2=True, dqn_model_path=dqn_model_path, dqn_weight=dqn_weight)
    elif player_2 == 'dqn':
        dqn_player_2 = dqn_policy(dqn_model_path)

    state = BlokusState() # Initial state
    turn = 1
    while not state.is_terminal():
        action = None
        top_actions = None
        total_visits = None
        player = state.P
        player_policy = player_1 if state.P == 0 else player_2
        if player_policy == "mcts" or player_policy == "mcts_dqn":
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
                action, top_actions = dqn_player_1(state)
            else:
                action, top_actions = dqn_player_2(state)
        state = state.successor(action)
        if display:
            print(f"Turn: {turn} - Player \033[34m#1\033[0m ({policy_names[player_1]}) vs. Player \033[31m#2\033[0m ({policy_names[player_2]})")
            state.display_board(action, player)
            if player_policy in ["mcts", "mcts_dqn"]:
                print(f"Top actions (turn: {turn}, total visits: {total_visits}):")
                for i, top_action in enumerate(top_actions):
                    prefix = f"\033[1;97;4{4 if player == 0 else 1}m" if i == 0 else ""
                    print(f"  {prefix}{i + 1}.", top_action[0], " - avg. reward:", round(top_action[1]["value"] / (top_action[1]["visits"] + 1e-6), 5), "and visits:", top_action[1]["visits"], "\033[0m")
            elif player_policy == "greedy":
                print(f"Top actions:")
                for i, top_action in enumerate(top_actions):
                    prefix = f"\033[1;97;4{4 if player == 0 else 1}m" if top_action[1] == action else ""
                    if top_action[1] == "Pass":
                        print(f"  {prefix}{i + 1}.", top_action[1], "\033[0m")
                    else:
                        print(f"  {prefix}{i + 1}.", top_action[1], "- score:", top_action[0], "\033[0m")
            elif player_policy == "random":
                print(f"Top actions (turn: {turn}):")
                print(f"  \033[1;97;4{4 if player == 0 else 1}m1. {action}\033[0m")
            print()
        turn += 1
    print("Game over! Final results:")
    print("Player #1 remaining pieces:", state.hands[0])
    print("Player #2 remaining pieces:", state.hands[1])
    print("Player #1 score:", state.get_score(0))
    print("Player #2 score:", state.get_score(1))
    print()

if __name__ == "__main__":
    print("Demo for Blokus Duo Agents!")
    print("Two matchups will be shown: MCTS vs. Greedy and MCTS + DQN vs. MCTS")
    run_demo("mcts", "greedy")
    run_demo("mcts_dqn", "mcts")
