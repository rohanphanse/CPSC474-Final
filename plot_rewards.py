import matplotlib.pyplot as plt
import csv
import pandas as pd

def plot_rewards(csv_path='reward_log.csv', save_path1='reward_plot.png', save_path2='moving_average.png'):
    episodes = []
    rewards = []
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            episodes.append(int(row[0]))
            rewards.append(float(row[1]))

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Training Reward per Episode')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Plot saved as {save_path}")
    
    df = pd.read_csv(csv_path)
    df['MovingAvg'] = df['TotalReward'].rolling(window=100).mean()
    plt.plot(df['Episode'], df['TotalReward'], alpha=0.3, label='Episode Reward')
    plt.plot(df['Episode'], df['MovingAvg'], color='red', label='100-Episode Moving Avg')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig(save_path2)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    plot_rewards('dqn_reward_logs/reward_log_dqn1.csv', 'dqn_training_plots/reward_plot_dqn1.png', 'dqn_training_plots/moving_average_dqn1.png')
    plot_rewards('dqn_reward_logs/reward_log_dqn2.csv', 'dqn_training_plots/reward_plot_dqn2.png', 'dqn_training_plots/moving_average_dqn2.png')