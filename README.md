# CPSC 474 Final Project: Monte Carlo Tree Search Techniques and Deep Q-Learning for Blokus Duo

By: Rohan Phanse and Areeb Gani

Link to repository: https://github.com/rohanphanse/CPSC474-Final

## Demo

To run the short demo for a few minutes (a sample of matchups), run the following command:

```bash
# Install dependencies
pip install -r requirements.txt
# Run demo
make
./BlokusDemo
```

Full reproducibility instructions are at the bottom of the README.

## Blokus Duo

Blokus Duo is a combinatorial game i.e. finite, two-player, deterministic, perfect information, turn-based

- Each player begins with 21 pieces. 
- Each turn, place a piece on empty spots of the 14 x 14 grid. The piece must touch one of your previous pieces (by a corner, not an edge!) 
- If you cannot play any more pieces, you must pass.
- Once both have passed, the winner is the player whose remaining pieces have smaller total value.

Note that Blokus has very large state space (exceeding $10^{100}$).

## Research Question

Hw much does integrating DQN-learned Q-values into MCTS improve agent performance compared to using MCTS or DQN alone? Does this improvement depend on the quality of the training regimen of DQN and the rollout policy for MCTS (e.g., greedy vs. random)?

## Agents


We first ran simulations with a random agent, a greedy agent, and an MCTS agent. We parallelize MCTS by building and merging multiple trees, use a greedy agent for stronger simulations with faster convergence, and accelerate action generation by precomputing piece orientations and focusing on anchor points.

We then trained two different DQN agents, dubbed **DQN1** (trained against a greedy agent) and **DQN2** (trained against a random agent with reward shaping). Both were trained using an adaptation of ``blokus.py`` to a gym environment, shown in ```blokus_env.py``` (this treats the opponent, either greedy or random, as a fixed part of the environment). 

Finally, we integrated a MCTS+DQN hybrid agent that used the $Q$-values learned from DQN for the MCTS selection criteria. This final version uses the MCTS with greedy rollout, paired with DQN1. 

![demo 1](https://lh3.googleusercontent.com/pw/AP1GczMRxUOSwPJtMfKsBjaWjvGyW3rc23gOtCI-kPkuTsQ-a8Zd3VmzIXnGoDYdIMvzQt7dUPaTGJASCUrfU1hy0abTT0owitt6g2TjxOTunBtXWOgyRM-6UOHW1eIFkhD8R60SFDWZ66SLx23EziSOwE2C=w1112-h774-s-no-gm)


## Evaluation Results
| Matchup | Win Rate (Player 1) |  Win Rate (Player 2) |
|------------------------------------------|---------------------|---------------------|
| Greedy vs. Random | 94% (Greedy) | 6% (Random) |
| DQN1 vs. Greedy | 15% (DQN1) | 85% (Greedy) |
| DQN1 vs. Random | 65% (DQN1) | 35% (Random) |
| DQN2 vs. Greedy | 6% (DQN2) | 94% (Greedy) |
| DQN2 vs. Random | 41% (DQN2) | 59% (Random) |
| MCTS (with random rollout) vs. Greedy | 59% (MCTS) | 41% (Greedy) |
| MCTS (with greedy rollout) vs. Greedy | 80% (MCTS) | 20% (Greedy) |
| MCTS+DQN vs. MCTS | 62% (MCTS+DQN) | 38% (MCTS) |

Detailed evaluations can be found in the `/evals` folder for each matchup. These include the number of games run (which varies depending on the agent), point margin, etc.

## Observations

- The greedy agent is a very strong baseline, easily defeating random play and both DQN agents.
- MCTS agents, especially when simulations use the greedy policy, are much stronger than random and competitive with the greedy agent.
- Combining MCTS with DQN (MCTS+DQN) yields an improvement over vanilla MCTS, even if the DQN agents on their own are not as powerful.
- Due to the large state space and squishing of $Q$-values, the DQN agents took a long time to train and exhibited great stochasticity. In the future, smarter reward shaping, modifications to architecture/hyperparameters, and different training scheme (e.g. with the masking loss function) could aid performance when trained over longer periods of time.

## Directory Structure

| File/Directory | Description |
|------------------------|-----------------------------------------------------------------------------|
| `README.md` | Project overview, instructions, and results. |
| `blokus.py` | Core Blokus Duo game logic and state representation. |
| `blokus_env.py` | Gym-style environment wrapper for Blokus, used for DQN training. |
| `greedy.py` | Implementation of the greedy agent. |
| `mcts.py` | Monte Carlo Tree Search agent and evaluation scripts. |
| `dqn_agent.py` | Deep Q-Network agent implementation. |
| `train_dqn.py` | Script to train DQN1 (against greedy agent). |
| `train_dqn_random.py` | Script to train DQN2 (against random agent with reward shaping). |
| `demo.py` | Script to run a short demo of agent matchups. |
| `parse_results.py` | Script to parse and analyze evaluation results, generate plots/statistics. |
| `agent_runs/` | Folder containing raw results of agent matchups. |
| `evals/` | Folder containing parsed results, plots, and metrics for each matchup. |
| `dqn_models/` | Saved DQN model weights. |
| `dqn_reward_logs/` | Reward logs from DQN training runs. |
| `dqn_training_plots/` | Training plots from DQN training runs. |
| `test.sh` | Shell script to run all agent matchups for full evaluation. |

## Reproducibility

Our complete results take hours to obtain, due to the branching factor of MCTS and training scheme of DQN. To reproduce our full results observed in ```mcts.py```, run

```bash
# Run full agent matchups
./test.sh
```

The **DQN1** model was trained using `python3 train_dqn.py`, and the **DQN2** model was trained using `python3 train_dqn_random.py`. Hyperparameters for the DQN architecture and training process are included in these files. All evals in the `/evals` folder were generated using `python3 parse_results.py --source [agent_runs/matchup]`. Information about the evaluation setup (i.e. number of games run) and more detailed statistics + plots of the results are included in `/evals`.
