from env import GymMachiKoro, MachiKoro
from env_vector_state import GymMachiKoro as VGymMachiKoro
from env_vector_state import MachiKoro as VMachiKoro
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
import cProfile
from mcts_agent import PVNet
from typing import Optional
from collections import OrderedDict

def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

# REMOVE
# @ray.remote
class MachiKoroActor:
    def __init__(self, env_cls, env_kwargs, num_mcts_sims, c_puct, pvnet_cls, pvnet_kwargs, buffer_capacity, n_players, worker_id):
        self._worker_id = worker_id
        self._buffer_capacity = buffer_capacity
        self._env = env_cls(**env_kwargs)

        self._agents = {
            f"player {i}": MCTSAgent(env_cls(**env_kwargs), num_mcts_sims=num_mcts_sims, c_puct=c_puct, pvnet=pvnet_cls(**pvnet_kwargs))
            for i in range(n_players)
        }
        assert list(self._agents.keys()) == self._env.player_order
        seed_all(self._worker_id)

    def update_pvnet(self, state_dict):
        print("worker", self._worker_id)
        for name, agent in self._agents.items():
            agent.update_pvnet(state_dict)

    def play_game(self, game_id, pvnet_state_dict: Optional[OrderedDict] = None):
        if pvnet_state_dict is not None:
            self.update_pvnet(pvnet_state_dict)

        buffer = Buffer(observation_space=self._env.observation_space, action_space=self._env.action_space, capacity=self._buffer_capacity)

        game = 0

        steps = 0
        t0 = time.time()
        t00 = time.time()

        done = False
        obs, info = self._env.reset()
        [agent.reset(info["state"]) for agent in self._agents.values()]
        game += 1

        while not done:
            action, probs = self._agents[self._env.current_player].compute_action(obs, info["state"])
            # print(f"player {self._env.current_player_index} played {self._env._action_idx_to_str[action]} | coins = {self._env.player_info[self._env.current_player].coins}")
            # print(f"probs {' '.join(map(str, probs.round(2)))}")
            print(f"worker: {self._worker_id} | game: {game_id} | steps: {steps} | buffer_size: {buffer.size} | player {self._env.current_player_index} played {self._env._action_idx_to_str[action]} | coins = {self._env.state_dict()['player_info'][self._env.current_player]['coins']}")
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
                info=info,
                )
            
            if steps % 100 == 0:
                print(f"time for 100 steps: {time.time() - t0}")
                t0 = time.time()

            if done:
                return buffer, time.time() - t00, game_id
            
            if buffer.isfull:
                return buffer, time.time() - t00, game_id

            obs = next_obs

def main():

    # with open("src/checkpoints/9.pkl", "rb") as file:
    #     buffer = pickle.load(file)
    # model = torch.load("checkpoints_3_static/2.pt")
    # breakpoint()

    # REMOVE uncomment
    # ray.init()
    n_players = 2
    env_cls = VGymMachiKoro
    env_kwargs = {"n_players": n_players}
    temp_env = env_cls(**env_kwargs)
    observation_space = temp_env.observation_space
    action_space = temp_env.action_space
    pvnet_cls = PVNet
    pvnet_kwargs = {
        "observation_space": observation_space,
        "action_space": action_space,
    }
    pvnet_for_training = PVNet(**pvnet_kwargs)
    # env = MachiKoro(n_players=2)
    # env = GymMachiKoro(env)
    # env = VMachiKoro(n_players=n_players)
    # env = VGymMachiKoro(env)

    # agents = {
    #     f"player 0": MCTSAgent(env=env, num_mcts_sims=10, c_puct=2),
    #     f"player 1": MCTSAgent(env=env, num_mcts_sims=10, c_puct=2),
    # }
    # model = torch.load("checkpoints_semi_small_doing_well/10.pt")
    # [agent.update_pvnet(model) for agent in agents.values()]
    # breakpoint()
    buffer_capacity = 1000

    # REMOVE uncomment
    # actor_pool = ActorPool(
    #     [
    #         MachiKoroActor.remote(
    #             env_cls=env_cls,
    #             env_kwargs=env_kwargs,
    #             num_mcts_sims=25,
    #             c_puct=2,
    #             pvnet_cls=pvnet_cls,
    #             pvnet_kwargs=pvnet_kwargs,
    #             buffer_capacity=buffer_capacity,
    #             n_players=n_players,
    #             worker_id=i
    #         ) for i in range(7)
    #     ]
    # )
    ## REMOVE
    actor = MachiKoroActor(
        env_cls=env_cls,
        env_kwargs=env_kwargs,
        num_mcts_sims=10,
        c_puct=2,
        pvnet_cls=pvnet_cls,
        pvnet_kwargs=pvnet_kwargs,
        buffer_capacity=buffer_capacity,
        n_players=n_players,
        worker_id=0
    )
    
    n_games_per_iteration = 1

    for i in range(1):
        t1 = time.time()
        pvnet_state_dict = pvnet_for_training.state_dict()
        # REMOVE uncomment
        # actor_generator = actor_pool.map_unordered(
        #     lambda a, v: a.play_game.remote(v, pvnet_state_dict),
        #     np.arange(n_games_per_iteration),
        # )
        actor_generator = [actor.play_game(i, pvnet_state_dict) for i in np.arange(n_games_per_iteration)]

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
            observation_space=observation_space,
            action_space=action_space
        )

        # buffer = get_trajectories_machi_koro(agents, buffer_capacity, gamma, 0, max_games)

        buffer.combine_buffers(buffers)
        print(f"time: {time.time() - t1}")

        checkpoint_dir = "checkpoints"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        with open(f"{checkpoint_dir}/{i}.pkl","wb") as file:
            pickle.dump(buffer, file)
        pvnet_for_training.train(buffer=buffer, batch_size=64, epochs=1)

        # [agent.update_pvnet(updated_pvnet) for agent in agents.values()]
        torch.save(pvnet_for_training, f"{checkpoint_dir}/{i}.pt")


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    profiler.dump_stats("self_play.prof")