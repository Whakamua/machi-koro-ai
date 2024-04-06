
from env_machi_koro_2 import GymMachiKoro2
from multielo import MultiElo
from gym.wrappers.flatten_observation import FlattenObservation
from random_agent import RandomAgent
from manual_agent import ManualAgent
from mcts_agent import MCTSAgent, PVNet
import cProfile
import copy
import gym
import time

def deepcopy_obs(obs_space, obs):
    obs_flt = gym.spaces.flatten(obs_space, obs)
    c1 = copy.deepcopy(obs_flt)
    c2 = copy.deepcopy(obs_flt)

def main():
    N_PLAYERS = 2
    CARD_INFO_PATH = "card_info_machi_koro_2.yaml"
    MCTS_SIMULATIONS = None
    THINKING_TIME = 10
    PUCT = 2


    time_start = time.time()
    env_cls = GymMachiKoro2
    env_kwargs = {"n_players": N_PLAYERS, "card_info_path": CARD_INFO_PATH}
    env = env_cls(**env_kwargs, print_info=True)

    pvnet_cls = PVNet
    pvnet_kwargs = {
        "env_cls": env_cls,
        "env_kwargs": env_kwargs,
        "uniform_pvnet": True,
        "custom_policy_edit": True,
        "custom_value": True,
    }

    elo = MultiElo()

    player_elo = [1000 for _ in range(env.n_players)]
    wins = [0 for _ in range(env.n_players)]

    agents = {
        "player 0": ManualAgent(env),
        "player 1": MCTSAgent(env_cls(**env_kwargs), num_mcts_sims=MCTS_SIMULATIONS, c_puct=PUCT, pvnet=pvnet_cls(**pvnet_kwargs), thinking_time=THINKING_TIME, print_info=True),
    }
    
    cumulative_steps = 0
    n_games = 1
    for game in range(n_games):
        action_history = []
        done = False
        obs, info = env.reset()
        [agent.reset(obs) for agent in agents.values()]

        count = 0
        steps = 0
        while not done:
            current_player = copy.deepcopy(env.current_player)
            action_mask = env.action_mask()
            action, probs = agents[current_player].compute_action(obs)

            next_obs, reward, done, truncated, info = env.step(action)
            action_history.append(action)
            
            steps+=1
            cumulative_steps += 1

            if done:
                print(f"Done, {current_player} wins!")
                ranking = [1 if player == current_player else 2 for player in env.player_order]

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