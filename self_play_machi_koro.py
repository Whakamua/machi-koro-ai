# from env import GymMachiKoro, MachiKoro
from env_vector_state import GymMachiKoro as VGymMachiKoro
# from env_vector_state import MachiKoro as VMachiKoro
# from multielo import MultiElo
# from random_agent import RandomAgent
from mcts_agent import MCTSAgent
from buffer import Buffer, BigBuffer
# import gym
# import copy
import numpy as np
import torch
import os
import ray
import pickle
import time
import random, os
from ray.util.actor_pool import ActorPool
# import cProfile
from mcts_agent import PVNet
from typing import Optional
from collections import OrderedDict

def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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
        [agent.reset(obs) for agent in self._agents.values()]
        game += 1

        while not done:
            action, probs = self._agents[self._env.current_player].compute_action(obs)

            print(f"worker: {self._worker_id} | game: {game_id} | steps: {steps} | buffer_size: {buffer.size} | player {self._env.current_player_index} played {self._env._action_idx_to_str[action]} | coins = {self._env._env.player_coins(self._env.current_player)}")
            next_obs, reward, done, truncated, info = self._env.step(action)     
            steps += 1

            buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
                probs=probs,
                current_player_index=self._env.current_player_index,
                action_mask=self._env.action_mask,
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

    MCTS_SIMULATIONS = 25
    PUCT = 2
    PVNET_TRAIN_EPOCHS = 20
    BATCH_SIZE = 64
    GAMES_PER_ITERATION = 7*20
    CARD_INFO_PATH = "card_info_quick_game.yaml"
    N_PLAYERS = 2


    use_ray = True
    # with open("src/checkpoints/9.pkl", "rb") as file:
    #     buffer = pickle.load(file)
    # model = torch.load("checkpoints_3_static/2.pt")
    # breakpoint()
    if use_ray:
        ray.init()

    env_cls = VGymMachiKoro
    env_kwargs = {"n_players": N_PLAYERS, "card_info_path": CARD_INFO_PATH}
    temp_env = env_cls(**env_kwargs)
    observation_space = temp_env.observation_space
    action_space = temp_env.action_space
    pvnet_cls = PVNet
    pvnet_kwargs = {
        "observation_space": observation_space,
        "action_space": action_space,
        "info": {
            "observation_indices": temp_env.observation_indices,
            "observation_values": temp_env.observation_values,
            "action_idx_to_str": temp_env._action_idx_to_str,
            "action_str_to_idx": temp_env._action_str_to_idx,
            "player_order": temp_env.player_order,
            "stage_order": temp_env._env._stage_order,
            "landmarks": temp_env._env._landmark_cards_ascending_in_price,
            "landmarks_cost": [temp_env.card_info[landmark]["cost"] for landmark in temp_env._env._landmark_cards_ascending_in_price]
        }
    }
    pvnet_for_training = PVNet(**pvnet_kwargs)
    buffer_capacity = 1000

    if use_ray:
        n_workers = int(ray.available_resources()["CPU"] - 1)
        actor_pool = ActorPool(
            [
                ray.remote(MachiKoroActor).remote(
                    env_cls=env_cls,
                    env_kwargs=env_kwargs,
                    num_mcts_sims=MCTS_SIMULATIONS,
                    c_puct=PUCT,
                    pvnet_cls=pvnet_cls,
                    pvnet_kwargs=pvnet_kwargs,
                    buffer_capacity=buffer_capacity,
                    n_players=N_PLAYERS,
                    worker_id=i
                ) for i in range(n_workers)
            ]
        )
    else:
        actor = MachiKoroActor(
            env_cls=env_cls,
            env_kwargs=env_kwargs,
            num_mcts_sims=10,
            c_puct=2,
            pvnet_cls=pvnet_cls,
            pvnet_kwargs=pvnet_kwargs,
            buffer_capacity=buffer_capacity,
            n_players=N_PLAYERS,
            worker_id=0
        )

    for i in range(1000):
        t1 = time.time()
        pvnet_state_dict = pvnet_for_training.state_dict()

        if use_ray:
            actor_generator = actor_pool.map_unordered(
                lambda a, v: a.play_game.remote(v, pvnet_state_dict),
                np.arange(GAMES_PER_ITERATION),
            )
        else:
            actor_generator = [actor.play_game(i, pvnet_state_dict) for i in np.arange(GAMES_PER_ITERATION)]

        buffers = []
        steps_collected = 0
        for buffer, time_taken, game_id in actor_generator:
            print(f"game {game_id} took {time_taken} sec to complete")
            buffers.append(buffer)
            steps_collected += buffer.size

        buffer = BigBuffer(
            observation_space=observation_space,
            action_space=action_space
        )

        buffer.combine_buffers(buffers)
        print(f"time: {time.time() - t1}")

        checkpoint_dir = "checkpoints"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        with open(f"{checkpoint_dir}/{i}.pkl","wb") as file:
            pickle.dump(buffer, file)
        pvnet_for_training.train(buffer=buffer, batch_size=BATCH_SIZE, epochs=PVNET_TRAIN_EPOCHS)

        torch.save(pvnet_for_training, f"{checkpoint_dir}/{i}.pt")


if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    main()
    # profiler.disable()
    # profiler.dump_stats("self_play.prof")