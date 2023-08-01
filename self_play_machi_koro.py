from env import GymMachiKoro, MachiKoro
from multielo import MultiElo
from random_agent import RandomAgent
from mcts_agent import MCTSAgent
from buffer import Buffer, BigBuffer
import gym
import copy
import numpy as np
import torch
import os
import ray
import pickle
import time

# buffers = {
#     player: Buffer(gamma=1, observation_space=env.observation_space, action_space=env.action_space, capacity=1000) for player in agents
# }

# can use a single buffer? Then cannot compute the values in montecarlo way, need to do sarsa. So
# that might not be ideal for MCTS, how does Alphazero do this? Does it bootstrap based on sarsa using search statistics or is the value of the search statistics used as the target?

@ray.remote
def get_trajectories_machi_koro(agents, buffer_capacity, gamma, worker, max_games: int | None = None):

    env = MachiKoro(n_players=len(agents))
    env = GymMachiKoro(env)

    buffer = Buffer(gamma=gamma, observation_space=env.observation_space, action_space=env.action_space, capacity=buffer_capacity)

    game = 0

    elo = MultiElo()

    player_elo = [1000 for _ in range(env.n_players)]
    wins = [0 for _ in range(env.n_players)]
    steps = 0
    t0 = time.time()

    while True:
        done = False
        obs, info = env.reset()
        [agent.reset(info["state"]) for agent in agents.values()]
        game += 1

        while not done:

            action, probs = agents[env.current_player].compute_action(obs, info["state"])

            # print(f"player {env.current_player_index} played {env._action_idx_to_str[action]} | coins = {env.player_info[env.current_player].coins}")
            # print(f"probs {' '.join(map(str, probs.round(2)))}")
            print(f"worker: {worker} | game: {game} | buffer_size: {buffer.size} | player {env.current_player_index} played {env._action_idx_to_str[action]} | coins = {env.player_info[env.current_player].coins}")
            next_obs, reward, done, truncated, info = env.step(action)     
            steps += 1
            # if buffer._size + 1 == buffer._capacity:
            #     done = True
            #     reward = 1
            buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
                probs=probs
                )
            
            if steps % 100 == 0:
                print(f"time for 100 steps: {time.time() - t0}")
                t0 = time.time()

            if done and max_games is not None and game == max_games:
                return buffer
            
            if buffer.isfull:
                return buffer

            obs = next_obs

            # if done:
            #     ranking = [1 if player == env.current_player else 2 for player in env.player_order]
            #     for i, rank in enumerate(ranking):
            #         if rank == 1:
            #             wins[i] += 1
            #     player_elo = elo.get_new_ratings(player_elo, result_order=ranking)
            #     print(f"game {game} | elo: {player_elo}, wins: {wins}")
            #     game += 1
            #     return buffer

# game 9999 | elo: [1035.11131277  995.44385084  985.17885556  984.26598084], wins: [2607, 2594, 2433, 2366]
# print("player 0")
# buffers["player 0"].post_process()
# [print(value, reward, done) for value, reward, done in zip(buffers["player 0"]._values, buffers["player 0"]._rewards, buffers["player 0"]._dones)]
# # print(buffers["player 0"]._rewards)

# print("player 1")
# buffers["player 1"].post_process()
# [print(value, reward, done) for value, reward, done in zip(buffers["player 1"]._values, buffers["player 1"]._rewards, buffers["player 1"]._dones)]
# print(buffer._obss[0] == buffer._obss[5])
if __name__ == "__main__":

    with open("src/checkpoints/9.pkl", "rb") as file:
        buffer = pickle.load(file)
    model = torch.load("src/checkpoints/9.pt")
    breakpoint()

    ray.init()
    env = MachiKoro(n_players=2)
    env = GymMachiKoro(env)

    agents = {
        f"player 0": MCTSAgent(env, num_mcts_sims=100, c_puct=2),
        f"player 1": MCTSAgent(env, num_mcts_sims=100, c_puct=2),
    }
    assert list(agents.keys()) == env.player_order

    for i in range(10):
        buffer_capacity = 6400
        max_games = None
        gamma = 1
        t1 = time.time()
        buffer_futures = [get_trajectories_machi_koro.remote(
            agents, buffer_capacity, gamma, worker, max_games) for worker in range(7)]
        buffers = ray.get(buffer_futures)
        buffer = BigBuffer(
            gamma=gamma,
            observation_space=env.observation_space,
            action_space=env.action_space
        )
        # buffer = get_trajectories_machi_koro(agents, buffer_capacity, gamma, 0, max_games)

        buffer.combine_buffers(buffers)
        print(f"time: {time.time() - t1}")

        updated_pvnet = agents["player 0"].train(buffer=buffer, batch_size=64)
        [agent.update_pvnet(updated_pvnet) for agent in agents.values()]
        checkpoint_dir = "src/checkpoints"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        with open(f"{checkpoint_dir}/{i}.pkl","wb") as file:
            pickle.dump(buffer, file)
        torch.save(updated_pvnet, f"{checkpoint_dir}/{i}.pt")
