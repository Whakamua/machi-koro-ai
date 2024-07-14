
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
import torch

def deepcopy_obs(obs_space, obs):
    obs_flt = gym.spaces.flatten(obs_space, obs)
    c1 = copy.deepcopy(obs_flt)
    c2 = copy.deepcopy(obs_flt)

def main():
    N_PLAYERS = 2
    # CARD_INFO_PATH = "card_info_machi_koro_2.yaml"
    CARD_INFO_PATH = "card_info_machi_koro_2_quick_game.yaml"
    MCTS_SIMULATIONS_P1 = 100
    MCTS_SIMULATIONS_P2 = 100
    THINKING_TIME = None
    PUCT = 2


    time_start = time.time()
    env_cls = GymMachiKoro2
    env_kwargs = {"n_players": N_PLAYERS, "card_info_path": CARD_INFO_PATH}
    env = env_cls(**env_kwargs, print_info=False)

    pvnet_cls = PVNet
    pvnetp0_kwargs = {
        "env_cls": env_cls,
        "env_kwargs": env_kwargs,
        "uniform_pvnet": True,
        "custom_policy_edit": True,
        "custom_value": True,
    }
    pvnetp1_kwargs = {
        "env_cls": env_cls,
        "env_kwargs": env_kwargs,
        "uniform_pvnet": True,
        "custom_policy_edit": True,
        "custom_value": True,
    }


    pvnet_p0 = pvnet_cls(**pvnetp0_kwargs)
    pvnet_p1 = pvnet_cls(**pvnetp1_kwargs)

    elo = MultiElo()

    player_elo = [1000 for _ in range(env.n_players)]
    wins = [0 for _ in range(env.n_players)]

    agents = {
        # "player 0": ManualAgent(env),
        "player 0": MCTSAgent(env_cls(**env_kwargs, print_info=False), num_mcts_sims=MCTS_SIMULATIONS_P1, c_puct=PUCT, pvnet=pvnet_p0, thinking_time=THINKING_TIME, print_info=False),
        "player 1": MCTSAgent(env_cls(**env_kwargs, print_info=False), num_mcts_sims=MCTS_SIMULATIONS_P2, c_puct=PUCT, pvnet=pvnet_p1, thinking_time=THINKING_TIME, print_info=False),
    }
    
    cumulative_steps = 0
    n_games = 100
    player_order = env.player_order
    for game in range(n_games):
        action_history = []
        done = False
        player_order = [player_order[-1]] + player_order[:-1]
        obs, info = env.reset(players_in_order=player_order)
        [agent.reset(obs) for agent in agents.values()]

        count = 0
        steps = 0
        while not done:
            current_player = copy.deepcopy(env.current_player)
            action_mask = env.action_mask()
            action, probs = agents[current_player].compute_action(obs)

            # breakpoint()
            next_obs, reward, done, truncated, info = env.step(action)
            action_history.append(action)
            
            steps+=1
            cumulative_steps += 1

            if done:
                print(f"Done, {current_player} wins!, player_order: {env.player_order}")
                ranking = [1 if current_player == "player 0" else 2, 1 if current_player == "player 1" else 2]
                for i, rank in enumerate(ranking):
                    if rank == 1:
                        wins[i] += 1
                player_elo = elo.get_new_ratings(player_elo, result_order=ranking)
                print(f"game {game} | steps: {steps} | elo: {player_elo} | wins: {wins}")
            obs = next_obs
    print(f"avg_steps: {cumulative_steps/n_games}")
    print(f"time taken: {time.time() - time_start}")

if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    main()
    # profiler.disable()
    # profiler.dump_stats("env.prof")