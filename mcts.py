import random
import time
import math

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

def mcts_policy(cpu_time_limit):
    def policy(state):
        start_time = time.perf_counter()
        root = MCTSNode(state)
        exploration_constant = 2
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
                if actions == "Pass":
                    action = "Pass"
                else:
                    sorted_actions = sorted(actions, key=lambda a: len(pieces[a[0]]) if a != "Pass" else 0, reverse=True)
                    action = sorted_actions[0] if random.random() < 0.7 else random.choice(actions)
                cur_state = cur_state.successor(action)
            payoff = cur_state.payoff()
            # Backpropogation
            while not cur_node is None:
                cur_node.num_visits += 1
                cur_node.value += payoff
                cur_node = cur_node.parent
        best = None
        for child in root.children:
            if best is None or child.num_visits > best.num_visits:
                best = child
        if best is None:
            return None
        return best.action
    return policy

if __name__ == "__main__":
    state = BlokusState() # Initial state
    mcts_player = mcts_policy(cpu_time_limit=2.0)
    # MCTS agent vs. random agent
    while not state.is_terminal():
        action = None
        if state.P == 0:
            action = mcts_player(state)
        else:
            actions = state.get_actions()
            action = actions if actions == "Pass" else random.choice(actions)
            time.sleep(1)
        state = state.successor(action)
        print("\033[2J\033[H", end="")
        state.display_board()
        print(f"Player #{2 - state.P} chose: {action}")
    # print("Player #1 moves:", moves[0])
    # print("Player #2 moves:", moves[1])
    print("Player #1 remaining pieces:", state.hands[0])
    print("Player #2 remaining pieces:", state.hands[1])
    print("Player #1 score:", state.get_score(0))
    print("Player #2 score:", state.get_score(1))