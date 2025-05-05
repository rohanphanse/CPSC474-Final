import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import argparse

def load_results(file_path):
    # Read the CSV file, handling multiple headers
    dfs = []
    current_df = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip().startswith('game_count'):
                if current_df:
                    dfs.append(pd.DataFrame(current_df, columns=['game_count', 'p1', 'p1_score', 'p2', 'p2_score']))
                current_df = []
            else:
                current_df.append(line.strip().split(','))
    
    if current_df:
        dfs.append(pd.DataFrame(current_df, columns=['game_count', 'p1', 'p1_score', 'p2', 'p2_score']))
    
    # Combine all dataframes
    df = pd.concat(dfs, ignore_index=True)
    
    # Convert columns to appropriate types
    df['game_count'] = pd.to_numeric(df['game_count'])
    df['p1_score'] = pd.to_numeric(df['p1_score'])
    df['p2_score'] = pd.to_numeric(df['p2_score'])
    
    return df

def calculate_metrics(df):
    # Basic win rate
    p1_wins = (df['p1_score'] < df['p2_score']).sum()
    p2_wins = (df['p2_score'] < df['p1_score']).sum()
    ties = (df['p1_score'] == df['p2_score']).sum()
    total_games = len(df)
    
    # Score margins
    score_margins = df['p2_score'] - df['p1_score']
    avg_margin = score_margins.mean()
    std_margin = score_margins.std()
    
    # Win rate
    p1_win_rate = (p1_wins + 0.5 * ties) / total_games
    p2_win_rate = (p2_wins + 0.5 * ties) / total_games
    
    # Average scores
    avg_p1_score = df['p1_score'].mean()
    avg_p2_score = df['p2_score'].mean()
    
    # Score distributions
    p1_score_std = df['p1_score'].std()
    p2_score_std = df['p2_score'].std()
    
    return {
        'total_games': total_games,
        'p1_wins': p1_wins,
        'p2_wins': p2_wins,
        'ties': ties,
        'p1_win_rate': p1_win_rate,
        'p2_win_rate': p2_win_rate,
        'avg_margin': avg_margin,
        'std_margin': std_margin,
        'avg_p1_score': avg_p1_score,
        'avg_p2_score': avg_p2_score,
        'p1_score_std': p1_score_std,
        'p2_score_std': p2_score_std
    }

def plot_results(df, metrics, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn')
    
    # 1. Score Distribution Plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df['p1_score'], label=df['p1'].iloc[0], fill=True)
    sns.kdeplot(data=df['p2_score'], label=df['p2'].iloc[0], fill=True)
    plt.title('Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'score_distribution.png'))
    plt.close()
    
    # 2. Score Margin Over Time
    plt.figure(figsize=(12, 6))
    plt.plot(df['game_count'], df['p2_score'] - df['p1_score'], 
             label='Score Margin (P2 - P1)', alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.title('Score Margin Over Time')
    plt.xlabel('Game Number')
    plt.ylabel('Score Margin')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'score_margin.png'))
    plt.close()
    
    # 3. Rolling Win Rate
    window_size = min(10, len(df))
    rolling_p1_wins = (df['p2_score'] > df['p1_score']).rolling(window=window_size).mean()
    rolling_p2_wins = (df['p1_score'] > df['p2_score']).rolling(window=window_size).mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['game_count'], rolling_p1_wins, label=f'{df["p1"].iloc[0]} Win Rate')
    plt.plot(df['game_count'], rolling_p2_wins, label=f'{df["p2"].iloc[0]} Win Rate')
    plt.title(f'Rolling Win Rate (Window Size: {window_size})')
    plt.xlabel('Game Number')
    plt.ylabel('Win Rate')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'rolling_win_rate.png'))
    plt.close()

def print_metrics(metrics, output_file=None):
    # Create the output string
    output = []
    output.append("\n=== Game Statistics ===")
    output.append(f"Total Games: {metrics['total_games']}")
    output.append(f"\nWins:")
    output.append(f"  {metrics['p1_wins']} wins for Player 1")
    output.append(f"  {metrics['p2_wins']} wins for Player 2")
    output.append(f"  {metrics['ties']} ties")
    output.append(f"\nWin Rates:")
    output.append(f"  Player 1: {metrics['p1_win_rate']:.2%}")
    output.append(f"  Player 2: {metrics['p2_win_rate']:.2%}")
    output.append(f"\nScore Statistics:")
    output.append(f"  Average Score Margin: {metrics['avg_margin']:.2f} ± {metrics['std_margin']:.2f}")
    output.append(f"  Player 1 Average Score: {metrics['avg_p1_score']:.2f} ± {metrics['p1_score_std']:.2f}")
    output.append(f"  Player 2 Average Score: {metrics['avg_p2_score']:.2f} ± {metrics['p2_score_std']:.2f}")
    
    # Print to console
    print('\n'.join(output))
    
    # Save to file if output_file is provided
    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(output))

def parse_args():
    parser = argparse.ArgumentParser(description='Parse and analyze game results')
    parser.add_argument('--source', type=str, default='dqn1_vs_greedy.txt',
                      help='Source file containing game results (default: dqn1_vs_greedy.txt)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Get the directory of the current script
    script_dir = Path(__file__).parent
    
    # Input and output paths
    input_file = script_dir / args.source
    output_dir = script_dir / 'evals' / args.source.split('/')[-1].split('.')[0]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and process results
    df = load_results(input_file)
    metrics = calculate_metrics(df)
    
    # Print metrics to console and save to file
    metrics_file = output_dir / 'metrics.txt'
    print_metrics(metrics, metrics_file)
    
    # Generate plots
    plot_results(df, metrics, output_dir)
    print(f"\nResults have been saved to: {output_dir}")

if __name__ == "__main__":
    main() 