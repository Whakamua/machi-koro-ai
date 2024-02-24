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
import random, os
from ray.util.actor_pool import ActorPool

def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

@ray.remote
class MachiKoroActor:
    def __init__(self, agents, buffer_capacity, env, worker_id):
        self._worker_id = worker_id
        self._agents = agents
        self._buffer_capacity = buffer_capacity
        self._env = env
        seed_all(self._worker_id)

    def update_pvnet(self, updated_pvnet):
        [agent.update_pvnet(updated_pvnet) for agent in self._agents.values()]

    def play_game(self, game_id):

        buffer = Buffer(observation_space=self._env.observation_space, action_space=self._env.action_space, capacity=self._buffer_capacity)

        game = 0

        steps = 0
        t0 = time.time()
        t00 = time.time()

        done = False
        obs, info = self._env.reset()
        [agent.reset(info["state"]) for agent in agents.values()]
        game += 1

        while not done:
            action, probs = agents[self._env.current_player].compute_action(obs, info["state"])
            # print(f"player {self._env.current_player_index} played {self._env._action_idx_to_str[action]} | coins = {self._env.player_info[self._env.current_player].coins}")
            # print(f"probs {' '.join(map(str, probs.round(2)))}")
            print(f"worker: {self._worker_id} | game: {game_id} | buffer_size: {buffer.size} | player {self._env.current_player_index} played {self._env._action_idx_to_str[action]} | coins = {self._env.player_info[self._env.current_player].coins}")
            next_obs, reward, done, truncated, info = self._env.step(action)     
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
                probs=probs,
                )
            
            if steps % 100 == 0:
                print(f"time for 100 steps: {time.time() - t0}")
                t0 = time.time()

            if done:
                return buffer, time.time() - t00, game_id
            
            if buffer.isfull:
                return buffer, time.time() - t00, game_id

            obs = next_obs

if __name__ == "__main__":

    # with open("src/checkpoints/9.pkl", "rb") as file:
    #     buffer = pickle.load(file)
    # model = torch.load("checkpoints_3_static/2.pt")
    # breakpoint()

    ray.init()
    env = MachiKoro(n_players=2)
    env = GymMachiKoro(env)

    agents = {
        f"player 0": MCTSAgent(env=env, num_mcts_sims=100, c_puct=2),
        f"player 1": MCTSAgent(env=env, num_mcts_sims=100, c_puct=2),
    }
    # model = torch.load("checkpoints_semi_small_doing_well/10.pt")
    # [agent.update_pvnet(model) for agent in agents.values()]
    assert list(agents.keys()) == env.player_order


    buffer_capacity = 1000
    actor_pool = ActorPool([MachiKoroActor.remote(agents, buffer_capacity, env, i) for i in range(7)])
    n_games_per_iteration = 150

    for i in range(100):
        t1 = time.time()

        list(actor_pool.map_unordered(
            lambda a, v: a.update_pvnet.remote(v),
            agents.values(),
        ))

        actor_generator = actor_pool.map_unordered(
            lambda a, v: a.play_game.remote(v),
            np.arange(n_games_per_iteration),
        )

        buffers = []
        steps_collected = 0
        for buffer, time_taken, game_id in actor_generator:
            print(f"game {game_id} took {time_taken} sec to complete")
            buffers.append(buffer)
            steps_collected += buffer.size
            # if steps_collected >= 100:
            #     break

        # buffer_futures = [get_trajectories_machi_koro.remote(
        #     agents, buffer_capacity, gamma, worker, max_games) for worker in range(7)]
        # buffers = ray.get(buffer_futures)
        buffer = BigBuffer(
            observation_space=env.observation_space,
            action_space=env.action_space
        )

        # buffer = get_trajectories_machi_koro(agents, buffer_capacity, gamma, 0, max_games)

        buffer.combine_buffers(buffers)
        print(f"time: {time.time() - t1}")

        checkpoint_dir = "checkpoints"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        with open(f"{checkpoint_dir}/{i}.pkl","wb") as file:
            pickle.dump(buffer, file)
        updated_pvnet = agents["player 0"].train(buffer=buffer, batch_size=64)

        [agent.update_pvnet(updated_pvnet) for agent in agents.values()]
        torch.save(updated_pvnet, f"{checkpoint_dir}/{i}.pt")
