from env_machi_koro_2 import GymMachiKoro2
from mcts_agent import MCTSAgent
from buffer import Buffer, BigBuffer
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
from multielo import MultiElo
import datetime
import copy
import mlflow

def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

GAME_STATE_EMPTY_DECKS = np.array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  3,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  3,  0,  -1, -1,
        -1,  -1,  -1,  -1,  -1, -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, -1, -1,  -1,
        -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
        -1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
       -1, -1, -1, -1, -1, -1, -1, -1, -1,  -1,  -1,  -1, -1,  -1,  -1,  -1,  -1,
        -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, -1, -1, -1, -1, -1, -1, -1,
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
       -1, -1, -1, -1,  -1,  0, -1,  0,  -1,  0,  -1,  0, -1,  0, -1,  0, -1,
        0, -1,  0,  -1,  0, -1,  0, -1,  0, -1,  0,  0,  0])

class MachiKoroActor:
    def __init__(
            self,
            env_cls,
            env_kwargs,
            num_mcts_sims,
            c_puct,
            pvnet_cls,
            pvnet_kwargs,
            buffer_capacity,
            n_players,
            worker_id
        ):
        self._worker_id = worker_id
        self._buffer_capacity = buffer_capacity
        self._env = env_cls(**env_kwargs)

        self._agents = {
            f"player {i}": MCTSAgent(env_cls(**env_kwargs), num_mcts_sims=num_mcts_sims, c_puct=c_puct, pvnet=pvnet_cls(**pvnet_kwargs))
            for i in range(n_players)
        }
        assert list(self._agents.keys()) == self._env.player_order
        seed_all(self._worker_id)

    def update_pvnet(self, state_dicts):
        for player, agent in self._agents.items():
            agent.update_pvnet(state_dicts[player])

    def play_game(self, game_id, pvnet_state_dicts: dict[str, OrderedDict] = None, start_state: Optional[np.ndarray] = None):
        self.update_pvnet(pvnet_state_dicts)

        buffer = Buffer(observation_space=self._env.observation_space, action_space=self._env.action_space, capacity=self._buffer_capacity)

        game = 0

        steps = 0
        t0 = time.time()
        t00 = time.time()

        done = False
        obs, info = self._env.reset(copy.deepcopy(start_state))
        [agent.reset(obs) for agent in self._agents.values()]
        game += 1

        while not done:
            action_mask = self._env.action_mask()
            action, probs = self._agents[self._env.current_player].compute_action(obs)

            # print(f"worker: {self._worker_id} | game: {game_id} | steps: {steps} | buffer_size: {buffer.size} | player {self._env.current_player_index} played {self._env._action_idx_to_str[action]} | coins = {self._env._env.player_coins(self._env.current_player)}")
            next_obs, reward, done, truncated, info = self._env.step(action)     
            steps += 1
            current_player_index = obs[self._env.observation_indices["current_player_index"]]
            current_player = f"player {current_player_index}"
            buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
                probs=probs,
                current_player_index=current_player_index,
                action_mask=action_mask,
                value_pred=self._agents[current_player].mcts.root.value_estimate,
                value_mcts=self._agents[current_player].mcts.root.value
                )
            
            if steps % 100 == 0:
                # print(f"time for 100 steps: {time.time() - t0}")
                t0 = time.time()

            if done:
                return buffer, time.time() - t00, game_id
            
            if buffer.isfull:
                return buffer, time.time() - t00, game_id

            obs = next_obs

def main():

    MCTS_SIMULATIONS = 100
    PUCT = 2
    PVNET_TRAIN_EPOCHS = 10
    BATCH_SIZE = 64
    GAMES_PER_ITERATION = 10
    TEST_GAMES_PER_ITERATION = 5
    CARD_INFO_PATH = "card_info_machi_koro_2.yaml"
    N_PLAYERS = 2
    TRAIN_VAL_SPLIT = 0.2
    LR = 0.001
    WEIGHT_DECAY = 1e-5
    BUFFER_CAPACITY = 1000

    use_ray = True
    # with open("src/checkpoints/9.pkl", "rb") as file:
    #     buffer = pickle.load(file)
    # model = torch.load("checkpoints_3_static/2.pt")
    # breakpoint()
    if use_ray:
        ray.init()

    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment(f"Self Play {int(time.time())}")

    env_cls = GymMachiKoro2
    env_kwargs = {"n_players": N_PLAYERS, "card_info_path": CARD_INFO_PATH}
    temp_env = env_cls(**env_kwargs)


    GAME_START_STATE = None
    # GAME_START_STATE = GAME_STATE_EMPTY_DECKS
    # for player_info in temp_env.observation_indices["player_info"].values():
    #     cards = player_info["cards"]
    #     GAME_START_STATE[cards["Harbor"]] = 1
    #     GAME_START_STATE[cards["Train Station"]] = 1
    #     GAME_START_STATE[cards["Shopping Mall"]] = 1
    #     GAME_START_STATE[cards["Amusement Park"]] = 1
    #     GAME_START_STATE[cards["Moon Tower"]] = 1
    #     GAME_START_STATE[cards["Airport"]] = 0
    #     GAME_START_STATE[player_info["coins"]] = 29
    # GAME_START_STATE[temp_env.observation_indices["marketplace"]["1-6"]["pos_0"]["card"]] = temp_env._env._card_name_to_num["Wheat Field"]
    # GAME_START_STATE[temp_env.observation_indices["marketplace"]["1-6"]["pos_0"]["count"]] = 2


    observation_space = temp_env.observation_space
    action_space = temp_env.action_space
    pvnet_cls = PVNet
    pvnet_kwargs = {
        "env_cls": env_cls,
        "env_kwargs": env_kwargs,
    }
    pvnet_for_training = PVNet(**pvnet_kwargs)

    if use_ray:
        n_workers = int(ray.available_resources()["CPU"] - 1)
        print(f"Found {n_workers} CPU cores available for workers")
        actor_pool = ActorPool(
            [
                ray.remote(MachiKoroActor).remote(
                    env_cls=env_cls,
                    env_kwargs=env_kwargs,
                    num_mcts_sims=MCTS_SIMULATIONS,
                    c_puct=PUCT,
                    pvnet_cls=pvnet_cls,
                    pvnet_kwargs=pvnet_kwargs,
                    buffer_capacity=BUFFER_CAPACITY,
                    n_players=N_PLAYERS,
                    worker_id=i
                ) for i in range(n_workers)
            ]
        )
    else:
        actor = MachiKoroActor(
            env_cls=env_cls,
            env_kwargs=env_kwargs,
            num_mcts_sims=MCTS_SIMULATIONS,
            c_puct=2,
            pvnet_cls=pvnet_cls,
            pvnet_kwargs=pvnet_kwargs,
            buffer_capacity=BUFFER_CAPACITY,
            n_players=N_PLAYERS,
            worker_id=0
        )
    elo = MultiElo()
    player_elo = [1000 for _ in range(N_PLAYERS)]
    wins = [0 for _ in range(N_PLAYERS)]

    checkpoint_dir = "checkpoints/" + str(datetime.datetime.now())
    os.makedirs(checkpoint_dir)


    for i in range(1000):
        t1 = time.time()
        pvnet_state_dict = pvnet_for_training.state_dict()

        training_pvnets = {
            f"player 0": {"pvnet_state_dict": pvnet_state_dict, "uniform_pvnet": False, "custom_policy_edit": False, "custom_value": False},
            f"player 1": {"pvnet_state_dict": pvnet_state_dict, "uniform_pvnet": False, "custom_policy_edit": False, "custom_value": False},
        }
        testing_pvnets = {
            f"player 0": {"pvnet_state_dict": pvnet_state_dict, "uniform_pvnet": False, "custom_policy_edit": False, "custom_value": False},
            f"player 1": {"pvnet_state_dict": pvnet_state_dict, "uniform_pvnet": True, "custom_policy_edit": True, "custom_value": True},
        }

        game_ids = np.arange(GAMES_PER_ITERATION + TEST_GAMES_PER_ITERATION)
        test_game_ids = game_ids[-TEST_GAMES_PER_ITERATION:] if TEST_GAMES_PER_ITERATION != 0 else []
        if use_ray:
            actor_generator = actor_pool.map_unordered(
                lambda a, v: a.play_game.remote(v, testing_pvnets if v in test_game_ids else training_pvnets, GAME_START_STATE),
                game_ids,
            )
        else:
            actor_generator = (actor.play_game(v, testing_pvnets if v in test_game_ids else training_pvnets, GAME_START_STATE) for v in game_ids)

        buffers = []
        test_buffers = []
        steps_collected = 0
        games_played = 0
        time_now = time.time()
        for buffer, time_taken, game_id in actor_generator:
            games_played += 1
            # print(f"game {game_id} took {time_taken} sec to complete", end='\r')
            time_taken = time.time() - time_now
            estimated_time_left = time_taken / games_played * (GAMES_PER_ITERATION + TEST_GAMES_PER_ITERATION - games_played)
            print(f"game {games_played}/{len(game_ids)}, estimated seconds left: {round(estimated_time_left)}", end="\r" if games_played < len(game_ids) else "\n")
            steps_collected += buffer.size
            if game_id in test_game_ids:
                winner = int(buffer.player_ids[-1].item())
                ranking = [1, 2] if winner == 0 else [2, 1]
                wins[winner] += 1
                player_elo = elo.get_new_ratings(player_elo, result_order=ranking)
                test_buffers.append(buffer)
            else:
                buffers.append(buffer)
            
        print(f"It: {i} | elo: {player_elo} | wins: {wins}")

        buffer = BigBuffer(
            observation_space=observation_space,
            action_space=action_space
        )
        test_buffer = BigBuffer(
            observation_space=observation_space,
            action_space=action_space
        )

        buffer.combine_buffers(buffers)
        test_buffer.combine_buffers(test_buffers)
        print(f"time: {time.time() - t1}")
        # obss, actions, rewards, next_obss, dones, player_ids, action_masks, values, value_preds, values_mcts, probs = test_buffer.get_episode(0)
        # test_buffer.compute_values()
        # [print(item) for item in zip(player_ids, value_preds, values_mcts)]
        # test_buffers[0]._nodes[-1].value_estimate
        # breakpoint()

        with open(f"{checkpoint_dir}/buffer_{i}.pkl", "wb") as file:
            pickle.dump(buffer, file)
        with open(f"{checkpoint_dir}/test_buffer_{i}.pkl", "wb") as file:
            pickle.dump(test_buffer, file)
        with mlflow.start_run() as run:
            pvnet_for_training.train(
                buffer=buffer,
                batch_size=BATCH_SIZE,
                epochs=PVNET_TRAIN_EPOCHS,
                train_val_split=TRAIN_VAL_SPLIT,
                lr=LR,
                weight_decay=WEIGHT_DECAY,
            )
        pvnet_for_training.save(f"{checkpoint_dir}/model_{i}")


if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    main()
    # profiler.disable()
    # profiler.dump_stats("self_play.prof")