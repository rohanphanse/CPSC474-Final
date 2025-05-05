import numpy as np
import torch
import os
import csv
from blokus_env import BlokusEnv
from dqn_agent import DQNAgent
from greedy import greedy_policy

# roughly 80 episodes a minute on cluster...
def train_dqn(
    num_episodes=10000,
    target_update=10,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.997,
    save_path='dqn_models/dqn_blokus_dqn1.pth',
    reward_log_path='dqn_reward_logs/reward_log_dqn1.csv'
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}...")
    greedy = greedy_policy()
    env = BlokusEnv(opponent_policy=greedy)
    agent = DQNAgent(
        env.observation_size,
        env.action_size,
        device=device,
        memory_size=50000  # Larger replay buffer
    )
    # Load model if exists
    if os.path.exists(save_path):
        print(f"Loading model from {save_path}...")
        agent.load(save_path)
    epsilon = epsilon_start

    reward_history = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            valid_action_indices = env.get_valid_action_indices()
            action = agent.select_action(state, valid_action_indices, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.store(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            total_reward += reward
        reward_history.append(total_reward)
        if episode % target_update == 0:
            agent.update_target()
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        if episode % 50 == 0:
            print(f'Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}')
    agent.save(save_path)
    print(f'Trained model saved to {save_path}')

    # Save reward history to CSV for plotting
    with open(reward_log_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Episode', 'TotalReward'])
        for i, r in enumerate(reward_history, 1):
            writer.writerow([i, r])
    print(f'Reward log saved to {reward_log_path}')

if __name__ == '__main__':
    train_dqn()