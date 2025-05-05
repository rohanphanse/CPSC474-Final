import numpy as np
import torch
import os
import csv
import matplotlib.pyplot as plt
from blokus_env import BlokusEnv
from dqn_agent import DQNAgent
# from greedy import greedy_policy  # Not needed for random policy
import random

def random_policy(state):
    actions = state.get_actions()
    if actions == "Pass":
        return "Pass", [(0, "Pass")]
    action = random.choice(actions)
    return action, [(0, action)]

def train_dqn(
    num_episodes=5000,
    target_update=10,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.995,  # Faster decay
    save_path='dqn_blokus_random.pth',
    reward_log_path='reward_log_random.csv'
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device {}...".format(device))
    env = BlokusEnv(opponent_policy=random_policy)  # Use random opponent
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
    loss_history = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        while not done:
            valid_action_indices = env.get_valid_action_indices()
            action = agent.select_action(state, valid_action_indices, epsilon)
            next_state, reward, done, _ = env.step(action)
            # Reward shaping: +0.1 for valid move, -0.1 for invalid, +5 for win, -5 for loss
            if reward > 0:
                reward += 0.1  # Placed a piece
            elif reward < 0:
                reward -= 0.1  # Lost a piece or game
            if done:
                if reward > 0:
                    reward += 5  # Win
                elif reward < 0:
                    reward -= 5  # Loss
            agent.store(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            total_reward += reward
            step_count += 1
        reward_history.append(total_reward)
        if episode % target_update == 0:
            agent.update_target()
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        if episode % 50 == 0:
            print(f'Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}, Steps: {step_count}')
        # Debug: print Q-values for the first state every 500 episodes
        if episode % 500 == 0:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_values = agent.model(state_tensor).cpu().numpy().flatten()
            print(f"Sample Q-values (first 10): {q_values[:10]}")
    agent.save(save_path)
    print(f'Trained model saved to {save_path}')

    # Save reward history to CSV for plotting
    with open(reward_log_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Episode', 'TotalReward'])
        for i, r in enumerate(reward_history, 1):
            writer.writerow([i, r])
    print(f'Reward log saved to {reward_log_path}')

    # Plot moving average of rewards
    rewards = np.array(reward_history)
    window = 100
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(10,6))
    plt.plot(rewards, alpha=0.3, label='Episode Reward')
    plt.plot(np.arange(window-1, len(rewards)), moving_avg, color='red', label=f'{window}-Episode Moving Avg')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.title('DQN Training Reward per Episode (with Moving Average)')
    plt.show()

if __name__ == '__main__':
    train_dqn() 