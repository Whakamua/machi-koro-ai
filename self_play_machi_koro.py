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


class MCTSActor:
    def __init__(
            self,
            worker_id,
            env_cls,
            env_kwargs,
            agent_cls,
            agent_kwargs,
            env_start_state,
            buffer_capacity,
        ):
        self._worker_id = worker_id
        seed_all(self._worker_id)
        self._env = env_cls(**env_kwargs)
        self._agents = {
            i: agent_cls(env=copy.deepcopy(self._env), **agent_kwargs) for i in range(self._env.n_players)
        }
        self._env_start_state = env_start_state
        self._buffer_capacity = buffer_capacity

    def play_game(
            self,
            game_id,
            agent_state_dicts,
        ):

        t00 = time.time()
        obs, info = self._env.reset(self._env_start_state)

        for i in agent_state_dicts.keys():
            self._agents[i].set_state_dict(agent_state_dicts[i]["state_dict"])

        [agent.reset(obs) for agent in self._agents.values()]

        buffer = Buffer(
            observation_space=self._env.observation_space,
            action_space=self._env.action_space,
            capacity=self._buffer_capacity,
        )

        steps = 0
        # t0 = time.time()

        done = False

        while not done:
            action_mask = self._env.action_mask()
            current_player_index = obs[self._env.observation_indices["current_player_index"]]
            action, probs = self._agents[current_player_index].compute_action(obs)
            

            # print(f"worker: {worker_id} | game: {game_id} | steps: {steps} | buffer_size: {buffer.size} | player {env.current_player_index} played {env._action_idx_to_str[action]} | coins = {env._env.player_coins(env.current_player)}")
            next_obs, reward, done, truncated, info = self._env.step(action)     
            steps += 1
            if buffer is not None:
                buffer.add(
                    obs=obs,
                    action=action,
                    reward=reward,
                    next_obs=next_obs,
                    done=done,
                    probs=probs,
                    current_player_index=current_player_index,
                    action_mask=action_mask,
                    value_pred=self._agents[current_player_index].mcts.root.value_estimate,
                    value_mcts=self._agents[current_player_index].mcts.root.value
                )
            
            # if steps % 10 == 0:
            #     print(f"game_id {game_id} time for 10 steps: {time.time() - t0}")
            #     t0 = time.time()

            if done:
                if buffer is not None:
                    buffer.compute_values()
                    time_taken = time.time() - t00
                    winner = agent_state_dicts[info["winning_player_index"]]["name"]
                return buffer, time_taken, game_id, winner

            if buffer is not None and buffer.isfull:
                buffer.compute_values()
                time_taken = time.time() - t00
                winning_player_index = info["winning_player_index"]
                winner = agent_state_dicts[winning_player_index]["name"] if winning_player_index is not None else None
                return buffer, time_taken, game_id, winner
            obs = next_obs

class Pit:
    def __init__(
        self,
        env_cls,
        env_kwargs,
        agent_cls,
        agent_kwargs,
        game_start_state,
        buffer_capacity,
        use_ray,
        games_per_addition=30,
    ):
        self._agents = {}
        self._elo = MultiElo()
        self._use_ray = use_ray
        self._games_per_addition = games_per_addition
        self._n_new_agents = 0

        if self._use_ray:
            n_workers = int(ray.available_resources()["CPU"] - 1)
            print(f"Found {n_workers} CPU cores available for workers")
            self._actor_pool = ActorPool(
                [
                    ray.remote(MCTSActor).remote(
                        worker_id=i, env_cls=env_cls, env_kwargs=env_kwargs, agent_cls=agent_cls, agent_kwargs=agent_kwargs, env_start_state=game_start_state, buffer_capacity=buffer_capacity
                    ) for i in range(n_workers)
                ]
            )
        else:
            self.actor = MCTSActor(
                worker_id=0, env_cls=env_cls, env_kwargs=env_kwargs, agent_cls=agent_cls, agent_kwargs=agent_kwargs, env_start_state=game_start_state, buffer_capacity=buffer_capacity
            )


    def add_initial_agent(self, name, agent):
        assert self._agents == {}, "agents should be empty when adding initial agent"
        self._add_agent(agent, name)
    

    def _add_agent(self, agent, name):
        self._agents[name] = {
            "agent_number": len(self._agents),
            "agent": agent,
            "elo": 1000,
            "wins": 0,
        }


    def add_agent_and_compute_ratings(self, agent, player_name, start_state=None):
        self._add_agent(agent, player_name)
        self._n_new_agents += 1
        self._compute_ratings_with_1_new_agent(self._games_per_addition, start_state)

    def get_agent_dict(self, agent_pair):
        return {i: {"name": agent_name, "state_dict": self._agents[agent_name]["agent"].get_state_dict()} for i, agent_name in enumerate(agent_pair)}

    def _compute_ratings_with_1_new_agent(
        self,
        n_games,
        start_state=None,
    ):
        assert self._n_new_agents == 1, "only 1 new agent allowed when computing ratings"
        game_ids = np.arange(n_games)
        
        agent_pairs_per_game_id = {
            game_id: [list(self._agents.keys())[-1], np.random.choice(list(self._agents.keys())[:-1], 1)[0]]
            for game_id in game_ids
        }
        # shuffling agent pairs to avoid bias, since order is the order in which agents start in 
        # the game. This is important because the first player has a slight advantage in Machi Koro
        for agent_pair in agent_pairs_per_game_id.values():
            random.shuffle(agent_pair)

        if self._use_ray:
            actor_generator = self._actor_pool.map_unordered(
                lambda a, v: a.play_game.remote(v, self.get_agent_dict(agent_pairs_per_game_id[v]), None),
                game_ids,
            )
        else:
            actor_generator = (self.actor.play_game(v, self.get_agent_dict(agent_pairs_per_game_id[v]), None) for v in game_ids)

        for buffer, time_taken, game_id, winner in actor_generator:
            if winner is None:
                print(f"WARNING: [Pit] game {game_id} had no winner")
                continue
            else:
                self._agents[winner]["wins"] += 1

                # compute new elo ratings
                agents_playing = agent_pairs_per_game_id[game_id]
                current_ratings = [self._agents[agent]["elo"] for agent in agents_playing]

                ranking = [1 if agent == winner else 2 for agent in agents_playing]
                new_ratings = self._elo.get_new_ratings(current_ratings, result_order=ranking)

                for i, agent in enumerate(agents_playing):
                    self._agents[agent]["elo"] = new_ratings[i]
        self._n_new_agents = 0

    def print_rankings(self):
        for agent, info in self._agents.items():
            print(f"{agent} | elo: {info['elo']} | wins: {info['wins']}")

def main():

    MCTS_SIMULATIONS = 100
    PUCT = 2
    PVNET_TRAIN_EPOCHS = 30
    BATCH_SIZE = 64
    GAMES_PER_ITERATION = 1000
    # CARD_INFO_PATH = "card_info_machi_koro_2.yaml"
    CARD_INFO_PATH = "card_info_machi_koro_2_quick_game.yaml"
    TRAIN_VAL_SPLIT = 0.2
    LR = 0.001
    WEIGHT_DECAY = 1e-5
    N_PLAYERS = 2
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
    env = env_cls(**env_kwargs)

    pvnet_for_training = PVNet(env)
    agent_cls = MCTSAgent
    agent = agent_cls(
        env=env,
        num_mcts_sims=MCTS_SIMULATIONS,
        c_puct=PUCT,
        pvnet=pvnet_for_training
    )
    agent_kwargs = {
        "num_mcts_sims": MCTS_SIMULATIONS,
        "c_puct": PUCT,
        "pvnet": pvnet_for_training
    }
    print("WARNING: USING VERY SIMPLE START STATE")
    env.reset()
    state = env.state_dict()
    state["player_info"]["player 0"]["coins"] = 29
    state["player_info"]["player 1"]["coins"] = 29
    GAME_START_STATE = env.state_dict_to_array(state)

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
    # env.reset(GAME_START_STATE)

    pit = Pit(
        env_cls=env_cls,
        env_kwargs=env_kwargs,
        agent_cls=agent_cls,
        agent_kwargs=agent_kwargs,
        game_start_state=GAME_START_STATE,
        buffer_capacity=BUFFER_CAPACITY,
        use_ray=use_ray,
        games_per_addition=30,
    )

    if use_ray:
        n_workers = int(ray.available_resources()["CPU"] - 1)
        print(f"Found {n_workers} CPU cores available for workers")
        actor_pool = ActorPool(
            [
                ray.remote(MCTSActor).remote(
                    worker_id=i, env_cls=env_cls, env_kwargs=env_kwargs, agent_cls=agent_cls, agent_kwargs=agent_kwargs, env_start_state=GAME_START_STATE, buffer_capacity=BUFFER_CAPACITY
                ) for i in range(n_workers)
            ]
        )
    else:
        actor = MCTSActor(
            worker_id=0, env_cls=env_cls, env_kwargs=env_kwargs, agent_cls=agent_cls, agent_kwargs=agent_kwargs, env_start_state=GAME_START_STATE, buffer_capacity=BUFFER_CAPACITY
        )

    checkpoint_dir = "checkpoints/" + str(datetime.datetime.now())
    os.makedirs(checkpoint_dir)

    best_agent = copy.deepcopy(agent)
    best_agent_name = "init_random_agent"
    pit.add_initial_agent(best_agent_name, best_agent)
    current_agent_name = "agent_0"
    training_agents = {
            0: {"name": current_agent_name, "agent": agent},
            1: {"name": best_agent_name, "agent": best_agent},
        }
    assert len(training_agents) == N_PLAYERS, "number of agents should be equal to number of players"

    buffers = []
    newest_buffers = []
    steps_collected = 0
    iterations_since_new_best_agent = 0
    number_of_agent_updates = 0
    for i in range(1):
        print("##################################################################################")
        print(f"iteration {i} | agent_updates: {number_of_agent_updates} | buffers_collected | {len(buffers)} steps_collected: {steps_collected}")
        print("##################################################################################")
        t1 = time.time()

        game_ids = np.arange(GAMES_PER_ITERATION)
        if use_ray:
            actor_generator = actor_pool.map_unordered(
                lambda a, v: a.play_game.remote(v, {i: {"name": agent["name"], "state_dict": agent["agent"].get_state_dict()} for i, agent in training_agents.items()}),
                game_ids,
            )
        else:
            actor_generator = (actor.play_game(v, {i: {"name": agent["name"], "state_dict": agent["agent"].get_state_dict()} for i, agent in training_agents.items()}) for v in game_ids)

        steps_collected = 0
        games_played = 0
        time_now = time.time()

        wins = {agent["name"]: 0 for agent in training_agents.values()}
        for buffer, game_time_taken, game_id, winner in actor_generator:
            games_played += 1
            # print(f"game {game_id} took {time_taken} sec to complete", end="\r" if games_played < len(game_ids) else "\n")
            time_taken = time.time() - time_now
            estimated_time_left = time_taken / games_played * (GAMES_PER_ITERATION - games_played)
            # steps_collected += buffer.size
            pre_append = time.time()

            if winner is None:
                print(f"WARNING: [SelfPlay] game {game_id} had no winner")
                continue
            else:
                newest_buffers.append(buffer)
                wins[winner] += 1

            print(f"game {games_played}/{len(game_ids)}, estimated seconds left: {round(estimated_time_left)}, latest game took {game_time_taken}s to complete, appending buffer took {time.time() - pre_append}", end="\r" if games_played < len(game_ids) else "\n")

        print(f"It: {i} | wins: {wins} time: {time.time() - t1}")

        # current_agent_win_ratio = wins[current_agent_name] / (wins[current_agent_name] + wins[best_agent_name])
        # if current_agent_win_ratio > 0.55 and iterations_since_new_best_agent > 0:
        #     print(f"new best agent found in iteration {i}, win ratio: {current_agent_win_ratio}")
        #     print("computing elo ratings")
        #     pit.add_agent_and_compute_ratings(copy.deepcopy(agent), f"agent_{i}", start_state=GAME_START_STATE)
        #     pit.print_rankings()
        #     best_agent = copy.deepcopy(agent)
        #     # resetting buffers
        #     buffers = newest_buffers
        #     newest_buffers = []
        #     iterations_since_new_best_agent = 0
        #     number_of_agent_updates += 1
        # else:
        #     print(f"current agent win ratio: {current_agent_win_ratio}")

        #     buffers = buffers + newest_buffers
        #     newest_buffers = []

        # big_buffer = BigBuffer(
        #     observation_space=observation_space,
        #     action_space=action_space
        # )
        # big_buffer.combine_buffers(buffers)

        # with open(f"{checkpoint_dir}/buffer_{i}.pkl", "wb") as file:
        #     pickle.dump(big_buffer, file)

        # with mlflow.start_run() as run:
        #     pvnet_for_training.train(
        #         buffer=big_buffer,
        #         batch_size=BATCH_SIZE,
        #         epochs=PVNET_TRAIN_EPOCHS,
        #         train_val_split=TRAIN_VAL_SPLIT,
        #         lr=LR,
        #         weight_decay=WEIGHT_DECAY,
        #         reset_weights=True,
        #     )
        # pvnet_for_training.save(f"{checkpoint_dir}/model_{i}")
        # iterations_since_new_best_agent += 1



        # if new_best_agent:
        #     best_agent = copy.deepcopy(agent)
        #     # resetting buffer
        #     big_buffer = BigBuffer(
        #         observation_space=observation_space,
        #         action_space=action_space
        #     )
        # else:
        #     big_buffer.combine_buffers([big_buffer] + buffers)

        #     with open(f"{checkpoint_dir}/buffer_{i}.pkl", "wb") as file:
        #         pickle.dump(big_buffer, file)

        #     with mlflow.start_run() as run:
        #         pvnet_for_training.train(
        #             buffer=buffer,
        #             batch_size=BATCH_SIZE,
        #             epochs=PVNET_TRAIN_EPOCHS,
        #             train_val_split=TRAIN_VAL_SPLIT,
        #             lr=LR,
        #             weight_decay=WEIGHT_DECAY,
        #         )
        #     pvnet_for_training.save(f"{checkpoint_dir}/model_{i}")


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    profiler.dump_stats("self_play.prof")