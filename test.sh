python3 mcts.py --player_1 mcts_dqn --player_2 mcts --dqn_model_path dqn_blokus.pth --results_path dqn_mcts_vs_mcts.txt --num_games 100 --display

python3 mcts.py --player_1 greedy --player_2 random --results_path greedy_vs_random.txt --num_games 10000

python3 mcts.py --player_1 mcts --player_2 greedy --results_path mcts_greedy_vs_greedy.txt --num_games 500

python3 mcts.py --player_1 mcts --player_2 greedy --results_path mcts_random_vs_greedy.txt --num_games 500

python3 mcts.py --player_1 dqn --player_2 greedy --dqn_model_path dqn_models/dqn_blokus_dqn1.pth --results_path dqn1_vs_greedy.txt --num_games 100

python3 mcts.py --player_1 dqn --player_2 greedy --dqn_model_path dqn_models/dqn_blokus_dqn2.pth --results_path dqn2_vs_greedy.txt --num_games 100

python3 mcts.py --player_1 dqn --player_2 random --dqn_model_path dqn_models/dqn_blokus_dqn1.pth --results_path dqn1_vs_random.txt --num_games 100

python3 mcts.py --player_1 dqn --player_2 random --dqn_model_path dqn_models/dqn_blokus_dqn2.pth --results_path dqn2_vs_random.txt --num_games 100