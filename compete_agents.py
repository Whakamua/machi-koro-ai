
from env_vector_state import GymMachiKoro
from multielo import MultiElo
from gym.wrappers.flatten_observation import FlattenObservation
from random_agent import RandomAgent
from mcts_agent import MCTSAgent
import cProfile
import copy
import gym
import time

def deepcopy_obs(obs_space, obs):
    obs_flt = gym.spaces.flatten(obs_space, obs)
    c1 = copy.deepcopy(obs_flt)
    c2 = copy.deepcopy(obs_flt)

def main():
    time_start = time.time()
    env = GymMachiKoro(n_players=2)
    elo = MultiElo()

    player_elo = [1000 for _ in range(env.n_players)]
    wins = [0 for _ in range(env.n_players)]

    agents = {
        # "player 0": MCTSAgent(env.observation_space, env.action_space),
        "player 0": RandomAgent(env.observation_space, env.action_space),
        "player 1": RandomAgent(env.observation_space, env.action_space),
    #     "player 2": RandomAgent(env.observation_space, env.action_space),
    #     "player 3": RandomAgent(env.observation_space, env.action_space)
    }
    cumulative_steps = 0
    n_games = 25
    for game in range(n_games):
        done = False
        obs, info = env.reset()

        count = 0
        steps = 0
        while not done:
            env.set_state(info["state"])
            action, probs = agents[env.current_player].compute_action(obs)    

            obs, reward, done, truncated, info = env.step(action)
            
            steps+=1
            cumulative_steps += 1

            if done:
                ranking = [1 if player == env.current_player else 2 for player in env.player_order]
                for i, rank in enumerate(ranking):
                    if rank == 1:
                        wins[i] += 1
                player_elo = elo.get_new_ratings(player_elo, result_order=ranking)
                print(f"game {game} | steps: {steps} | elo: {player_elo} | wins: {wins}")
    print(f"avg_steps: {cumulative_steps/n_games}")
    print(f"time taken: {time.time() - time_start}")

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    profiler.dump_stats("env.prof")