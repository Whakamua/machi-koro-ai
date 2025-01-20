from multielo import MultiElo
import random
import numpy as np
import matplotlib.pyplot as plt
import copy

"""
Experiment to test ELO rating system

There are 6 players with increasing strength.
This indicates their win chance. A player of strength 6 has a 6x chance of winning against a player of strength 1.

In the first experiment, all players play against each other in random pairs for 400 games.
This is done to estimate the ELO of each player if an infinite number of games were played.
This experiment is repeated 100 times to get a distribution of ELOs for each player.

In the second experiment, players players pair up, but players are added to the pool over time.
Where a new player in the pool plays 30 games, against random players that were already in the pool.
Players that were already in the pool, do not battle eachother any longer.
This experiment is repeated 100 times to get a distribution of ELOs for each player.

It turns out that bothmethods result in rougly the same ELO distribution for each player.
Which is nice, because this can be used to estimate the elo of different agents while training a
reinforcement learning agent. Everytime a new iteration is done, the ELO of the newestagent can be
estimated with all other iterations of the agent, with limited compute. (only 30 games needed).

Which makes it easy to track the training progress of the agent with limited compute.
"""

elo = MultiElo()


def play_game(players):
    # get random result order based on strength

    strength_sum = sum([players[player]["strength"] for player in players])
    weights = [players[player]["strength"]/strength_sum for player in players]

    winner = random.choices(list(players.keys()), weights=weights, k=1)[0]
    players[winner]["wins"] += 1
    result_order = [1 if player == winner else 2 for player in players.keys()]

    ratings = [players[player]["elo"] for player in players]
    ratings = elo.get_new_ratings(initial_ratings=ratings, result_order=result_order)
    for i, player in enumerate(players):
        players[player]["elo"] = ratings[i]


all_players = {
    f"player {i}": {
        "elo": 1000,
        "strength": i+1,
        "wins": 0,
        "elos_per_competition": []
    }
    for i in range(6)
}
# all_players[f"player {len(all_players)-1}"]["strength"] = 100

players = copy.deepcopy(all_players)

for _ in range(100):
    # reset elo
    for player in players:
        players[player]["elo"] = 1000

    game_ids = [i for i in range(400)]
    agent_pairs_per_game_id = {
        game_id: np.random.choice(list(players.keys()), 2, replace=False) 
        for game_id in game_ids
    }
    for i in game_ids:
        play_game(
            {
                player: players[player]
                for player in agent_pairs_per_game_id[i]
            }
        )
    for player in players.values():
        player["elos_per_competition"].append(player["elo"])

plt.figure()
for player, info in players.items():
    plt.hist(info["elos_per_competition"], bins=20, alpha=0.5, label=player)
    # average elo
    print(player, np.mean(info["elos_per_competition"]))
print("#######################")
plt.legend()
plt.title("Plot 1")

############################################################################################################
players = copy.deepcopy(all_players)

for _ in range(100):
    # reset elo
    for player in players:
        players[player]["elo"] = 1000
    
    for j in range(len(players) - 1):
        game_ids = [i for i in range(30)]
        agent_pairs_per_game_id = {
            game_id: [list(players.keys())[j+1], np.random.choice(list(players.keys())[:j+1], 1)[0]] 
            for game_id in game_ids
        }
        for i in game_ids:
            play_game(
                {
                    player: players[player]
                    for player in agent_pairs_per_game_id[i]
                }
            )

    for player in players.values():
        player["elos_per_competition"].append(player["elo"])
    

plt.figure()
for player, info in players.items():
    plt.hist(info["elos_per_competition"], bins=20, alpha=0.5, label=player)
    # average elo
    print(player, np.mean(info["elos_per_competition"]))
plt.legend()
plt.title("Plot 2")

plt.show()
