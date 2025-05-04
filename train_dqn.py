# train_dqn.py

import os
import torch
from blokus import BlokusState, pieces, orientations, N
from greedy import greedy_policy
from dqn_agent import DQNAgent, dqn_policy

# Setup
NUM_EPISODES = 2500
PRINT_EVERY = 100
SAVE_EVERY = 500
MODEL_PATH = "dqn_model.pt"

# Define action space
action_space = [("Pass",)] + [
    (p, x, y, rot, refl)
    for p in pieces
    for (coords, rot, refl) in orientations[p]
    for x in range(N) for y in range(N)
]

# Initialize DQN agent and policy
agent = DQNAgent(action_space)
policy_dqn = dqn_policy(agent)
policy_greedy = greedy_policy()

# Optionally load previous model
if os.path.exists(MODEL_PATH):
    print("Loading existing model...")
    agent.q_net.load_state_dict(torch.load(MODEL_PATH))
    agent.target_q_net.load_state_dict(agent.q_net.state_dict())

# Training loop
for episode in range(1, NUM_EPISODES + 1):
    state = BlokusState()
    transitions = {0: [], 1: []}

    while not state.is_terminal():
        player = state.P
        old_state = state.copy()

        if player == 0:
            action, _ = policy_dqn(state)
        else:
            action, _ = policy_greedy(state)

        if action == "Pass" or (isinstance(action, tuple) and action[0] == "Pass"):
            new_state = state.successor("Pass")
        else:
            new_state = state.successor(action)


        reward = 0
        done = new_state.is_terminal()
        if done:
            payoff = new_state.payoff()
            reward = payoff if player == 0 else -payoff

        if player == 0:
            agent.store_transition((old_state, action, reward, new_state, done))
            agent.train_step()

        state = new_state

    # Logging
    if episode % PRINT_EVERY == 0:
        print(f"Episode {episode}: Score -> P1 (DQN) = {state.get_score(0)}, P2 (Greedy) = {state.get_score(1)}")

    # Save model
    if episode % SAVE_EVERY == 0:
        torch.save(agent.q_net.state_dict(), MODEL_PATH)
        print(f"Model saved at episode {episode}.")

print("Training complete.")
torch.save(agent.q_net.state_dict(), MODEL_PATH)
