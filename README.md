# CPSC 474 Final Project: Monte Carlo Tree Search Techniques and Deep Q-Learning for Blokus Duo

By: Rohan Phanse and Areeb Gani

## Demo

```bash
# Run demo
pypy3 mcts.py
```

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