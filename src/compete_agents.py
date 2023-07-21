from env import GymMachiKoro, MachiKoro
from multielo import MultiElo
from gym.wrappers.flatten_observation import FlattenObservation
from random_agent import RandomAgent
from mcts_agent import MCTSAgent


env = MachiKoro(n_players=4)
env = GymMachiKoro(env)
# env = FlattenObservation(env)
elo = MultiElo()

player_elo = [1000 for _ in range(env.n_players)]
wins = [0 for _ in range(env.n_players)]

agents = {
    "player 0": MCTSAgent(env.observation_space, env.action_space),
    "player 1": RandomAgent(env.observation_space, env.action_space),
#     "player 2": RandomAgent(env.observation_space, env.action_space),
#     "player 3": RandomAgent(env.observation_space, env.action_space)
}

for game in range(10000):
    done = False
    obs, info = env.reset()
    count = 0
    while not done:

        action = agents[env.current_player].compute_action(obs, env.get_state())
    
        obs, reward, done, truncated, info = env.step(action)

        if done:
            ranking = [1 if player == env._env._current_player else 2 for player in env._env._player_info.keys()]
            for i, rank in enumerate(ranking):
                if rank == 1:
                    wins[i] += 1
            player_elo = elo.get_new_ratings(player_elo, result_order=ranking)
            print(f"game {game} | elo: {player_elo}, wins: {wins}")

# game 9999 | elo: [1035.11131277  995.44385084  985.17885556  984.26598084], wins: [2607, 2594, 2433, 2366]
breakpoint()