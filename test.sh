python3 mcts.py --player_1 mcts_dqn --player2 mcts --dqn_model_path dqn_blokus.pth --results_path dqn_mcts_vs_mcts.txt --num_games 100 --display

python3 mcts.py --player_1 greedy --player2 random --results_path greedy_vs_random.txt --num_games 100