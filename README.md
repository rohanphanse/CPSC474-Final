# CPSC 474 Final Project: Monte Carlo Tree Search Techniques and Deep Q-Learning for Blokus Duo

By: Rohan Phanse and Areeb Gani

## Demo

To run the short demo for a few minutes (a sample of matchups), run the following command:

```bash
# Run demo
python3 demo.py
```

Full reproducibility instructions are at the bottom of the README.

### MCTS Agent
Because Blokus Duo has a very large state space (exceeding $10^{100}$), we employed the following optimizations in our MCTS agent.

Optimizations:
* Parallelization: The MCTS agent builds multiple trees in parallel and then merges them into a single tree.
* Greedy Agent: The MCTS simulations use the greedy agent to choose actions, improving performance and convergence time compared to using a random agent.
* Get Actions: We observed that computing the possible actions for a given state was a major bottleneck given the large state space, so we precompute orientations and only check anchor points to speed up this process. 

![demo 1](https://lh3.googleusercontent.com/pw/AP1GczMRxUOSwPJtMfKsBjaWjvGyW3rc23gOtCI-kPkuTsQ-a8Zd3VmzIXnGoDYdIMvzQt7dUPaTGJASCUrfU1hy0abTT0owitt6g2TjxOTunBtXWOgyRM-6UOHW1eIFkhD8R60SFDWZ66SLx23EziSOwE2C=w1112-h774-s-no-gm)

### Greedy Agent

![demo 2](https://lh3.googleusercontent.com/pw/AP1GczNCldUc92F0z9rJEOUHXYZhvBmO9fKLD9BGdnaVunrfdW4wm_D_fXYLsWTsUSPYeloeLvaOLO-9J6_DOlJm-l52tsCubw7mH1PUhWjVHyGbm6wCeljPTsOcYYE6YA6B-Guku0uYjB-G1NmlVvRyQao3=w770-h774-s-no-gm)

### End of Game

![demo 3](https://lh3.googleusercontent.com/pw/AP1GczMNQY2UeXylocucgx1uUWRjSHRJyuFnJPo414SpYUgJKP1WQoksJtKyiAOk2EDpbDaXcSoc66dutGJKjcL5mN-fVAZVmwD3ITDIQbuYRBw4jOHI-IxbUd0TexZY9txs5-uqydLQSZBGwzTsAjqxQePx=w930-h652-s-no-gm)

## State Space Complexity

We can estimate the state space of Blokus Duo by considering the following things:

1. Each player starts off with 21 pieces and plays one piece per turn.
2. Each piece can be rotated, flipped, and placed at any valid spot in the 14 x 14 grid. Therefore, each piece has at most 8 orientations and can be played in at most 196 spots.

If we assume that each player has an average of 10 playable pieces per turn, which can each be played in 100 different ways on average, we get 1000 actions per turn.

At most, a game can take 42 turns. Assuming an average of 35 turns per game and that actions most often lead to unique states, we get an estimate of $1000^{35} = 10^{105}$ total states for Blokus Duo.

## Agents

We ran simulations with a random agent, a greedy agent, and an MCTS agent. We also trained two different DQN agents, dubbed **DQN1** (trained against a greedy agent) and **DQN2** (trained against a random agent with reward shaping). Finally, we integrated a MCTS+DQN hybrid agent that used the $Q$-values learned from DQN for the MCTS selection criteria.



## Evaluation Results
| Matchup | Win Rate (Player 1) |
|------------------------------------------|---------------------|
| Greedy vs. Random | 94% (Greedy) |
| DQN1 vs. Greedy | 13% (DQN1) |
| DQN1 vs. Random | 62% (DQN1) |
| DQN2 vs. Greedy | 6% (DQN2) |
| DQN2 vs. Random | 38% (DQN2) |
| MCTS (trained with random) vs. Greedy | 59% (MCTS) |
| MCTS (trained with greedy) vs. Greedy | 78% (MCTS) |
| MCTS+DQN vs. MCTS | 62% (MCTS+DQN) |

## Observations

- The greedy agent is a very strong baseline, easily defeating random play and both DQN agents.
- MCTS agents, especially when simulations use the greedy policy, are much stronger than random and competitive with the greedy agent.
- Combining MCTS with DQN (MCTS+DQN) yields an improvement over vanilla MCTS, even if the DQN agents on their own are not as powerful.
- Due to the large state space and squishing of $Q$-values, the DQN agents took a long time to train and exhibited great stochasticity. In the future, smarter reward shaping, modifications to architecture/hyperparameters, and different training scheme (e.g. with the masking loss function) could aid performance when trained over longer periods of time.

## Reproducibility

Our complete results take hours to obtain, due to the branching factor of MCTS and training scheme of DQN. To reproduce our full results observed in ```mcts.py```, run

```bash
# Run full agent matchups
./test.sh
```

The **DQN1** model was trained using `python3 train_dqn.py`, and the **DQN2** model was trained using `python3 train_dqn_random.py`. All evals in the `/evals` folder were generated using `python3 parse_results.py --source [agent_runs/matchup]`.