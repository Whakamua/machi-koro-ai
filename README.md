# MCTS Self Play for Machi Koro 2

This repo implements:
- An environment for both Machi Koro 1 and Machi Koro 2
- MCTS for stochastic environments through the use of afterstates following [Planning in Stochastic Environments with a Learned Model - Ioannis Antonoglou, Julian Schrittwieser, et al.](https://openreview.net/forum?id=X6D9bAHhBQ1)
- A self play system distributed using Ray.
- A training pipeline for a policy and value net.
- A chunked data storage system, seemlessly handling larger than memory data for gathering self play data and training the policy and value net.
- An ELO system for a pit where a new player only needs to battle existing players in the pit to accurately estimate all their ELOs

To run self play, you'll need python 3.10 or higher and install the requirements:
```
pip install -r requirements.txt
```

Then you can run:
```
python self_play_machi_koro.py
```

To update the hyperparameters, edit the variable 'hyperparameters' in self_play_machi_koro.py
The current hyperparameters should get an agent in iteration 5 with an ELO of 1100 for a short version of Machi Koro 2. 


There are agent checkpoints for a quick game to play against, the above mentioned agent_5 can be played against using:
```
python compete_agents.py
```