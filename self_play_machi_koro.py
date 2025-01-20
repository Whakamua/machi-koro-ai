from env_machi_koro_2 import GymMachiKoro2
from mcts_agent import MCTSAgent
from buffer import Buffer
import numpy as np
import torch
import os
import ray
import pickle
import sys
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
import h5py
from mcts_agent import NotEnoughDataError
from collections import deque

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

class SelfPlayLogger:
    def __init__(self, buffer_log_path: str):
        self.iteration = -1
        self.steps_collected_per_iteration = {"val": {}, "train": {}}
        self.completed_games_per_iteration = {"val": {}, "train": {}}
        self.current_agents = None
        self.buffer_log_path = buffer_log_path
        self.number_of_agent_updates = 0

    def __enter__(self):
        self.h5f = h5py.File(self.buffer_log_path, "w")
        self.h5f.create_group("train")
        self.h5f.create_group("val")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.h5f.close()

    def start_iteration(
            self,
            game_ids: int,
            val_fraction,
            subset_rules: dict[str, float],
            agent_names: list[str],
        ):
        self.iteration += 1
        self.new_iteration = True

        # self.averge_time_left_estimate = deque(maxlen=10)
        
        assert len(subset_rules) == (self.iteration+1), "subset rules should be defined for each iteration"
        self.subset_rules = subset_rules
        self.iteration_start_time = time.time()
        self.number_of_games_for_current_iteration = len(game_ids)
        self.val_indices = np.random.choice(game_ids, int(len(game_ids) * (val_fraction)), replace=False)

        self.games_played = {}
        self.games_time_taken_deque = deque(maxlen=100)

        self.steps_collected_per_iteration["train"][f"iteration_{self.iteration}"] = 0
        self.steps_collected_per_iteration["val"][f"iteration_{self.iteration}"] = 0
        self.completed_games_per_iteration["train"][f"iteration_{self.iteration}"] = 0
        self.completed_games_per_iteration["val"][f"iteration_{self.iteration}"] = 0

        self.h5f["train"].create_group(f"iteration_{self.iteration}")
        self.h5f["val"].create_group(f"iteration_{self.iteration}")

        self.wins_current_iteration = {agent_name: 0 for agent_name in agent_names}
        self.games_with_no_winner_current_iteration = 0

    def log_game(self, buffer, game_time_taken, game_id, winner):
        self.games_played[game_id] = {"time_taken": game_time_taken, "winner": winner}

        self.games_time_taken_deque.append(game_time_taken)

        if "columns" not in self.h5f.attrs:
            self.h5f.attrs["columns"] = list(buffer.flattened_column_names_and_types_dict.keys())

        if winner is not None:
            buffer_np = buffer.export_flattened()
            self.wins_current_iteration[winner] += 1
            if game_id in self.val_indices:
                self.h5f["val"][f"iteration_{self.iteration}"].create_dataset(f"game_{game_id}", data=buffer_np)
                self.completed_games_per_iteration["val"][f"iteration_{self.iteration}"] += 1
                self.steps_collected_per_iteration["val"][f"iteration_{self.iteration}"] += buffer.size
            else:
                self.h5f["train"][f"iteration_{self.iteration}"].create_dataset(f"game_{game_id}", data=buffer_np)
                self.completed_games_per_iteration["train"][f"iteration_{self.iteration}"] += 1
                self.steps_collected_per_iteration["train"][f"iteration_{self.iteration}"] += buffer.size
        else:
            self.games_with_no_winner_current_iteration += 1

        self.log()

    def log_agent_update(self):
        self.number_of_agent_updates += 1

    @property
    def completed_games_current_iteration(self):
        return self.completed_games_per_iteration["train"][f"iteration_{self.iteration}"] + self.completed_games_per_iteration["val"][f"iteration_{self.iteration}"]

    def log(
        self,
    ):
        # Cursor handling: Clear previous output if not the first call
        if not self.new_iteration:
            # Move cursor up 13 lines and clear each line
            for _ in range(13):  # 13 is the number of lines printed
                sys.stdout.write("\033[F\033[K")  # Move up and clear line
        else:
            self.new_iteration = False
        
        avg_game_time = np.mean(self.games_time_taken_deque)
        estimated_iteration_time_left = avg_game_time * (self.number_of_games_for_current_iteration - len(self.games_played))

        train_games_for_next_training_run = int(
            sum([
                    self.completed_games_per_iteration["train"][iteration] * fraction
                    for iteration, fraction in self.subset_rules.items()
                ])
        )
        val_games_for_next_training_run = int(
            sum([
                    self.completed_games_per_iteration["val"][iteration] * fraction
                    for iteration, fraction in self.subset_rules.items()
                ])
        )

        train_steps_for_next_training_run = int(
            sum([
                    self.steps_collected_per_iteration["train"][iteration] * fraction
                    for iteration, fraction in self.subset_rules.items()
                ])
        )
        val_steps_for_next_training_run = int(
            sum([
                    self.steps_collected_per_iteration["val"][iteration] * fraction
                    for iteration, fraction in self.subset_rules.items()
                ])
        )
        # breakpoint()

        sys.stdout.write("\n")
        sys.stdout.write("##################################################################################\n")
        sys.stdout.write(f"iteration {self.iteration}\n")
        sys.stdout.write(f"agent_updates: {self.number_of_agent_updates}\n")
        sys.stdout.write(f"games_played: {len(self.games_played)}/{self.number_of_games_for_current_iteration}\n")
        sys.stdout.write(f"games_with_no_winner/completed_games: {self.games_with_no_winner_current_iteration}/{len(self.games_played)}\n")
        sys.stdout.write(f"estimated_iteration_time_left: {estimated_iteration_time_left:.0f}s\n")
        sys.stdout.write(f"last_game_time_taken: {self.games_time_taken_deque[-1]:.4f}s\n")
        sys.stdout.write(f"wins: {self.wins_current_iteration}\n")
        sys.stdout.write(f"next_training_run: {train_games_for_next_training_run} games, {train_steps_for_next_training_run} steps\n")
        sys.stdout.write(f"next_validation_run: {val_games_for_next_training_run} games, {val_steps_for_next_training_run} steps\n")
        sys.stdout.write("##################################################################################\n")
        sys.stdout.write("\n")
        sys.stdout.flush()


class MCTSActor:
    def __init__(
            self,
            worker_id
        ):
        self._worker_id = worker_id
        seed_all(self._worker_id)

    def play_game(
            self,
            game_id,
            env,
            game_start_state,
            agents,
            buffer,
        ):
        t00 = time.time()
        env = copy.deepcopy(env)
        obs, info = env.reset(copy.deepcopy(game_start_state))

        agents = copy.deepcopy(agents)
        [agent["agent"].reset(obs) for agent in agents.values()]

        buffer = copy.deepcopy(buffer)

        steps = 0
        t0 = time.time()

        done = False

        while not done:
            action_mask = env.action_mask()
            current_player_index = obs[env.observation_indices["current_player_index"]]
            action, probs = agents[current_player_index]["agent"].compute_action(obs)

            # print(f"worker: {worker_id} | game: {game_id} | steps: {steps} | buffer_size: {buffer.size} | player {env.current_player_index} played {env._action_idx_to_str[action]} | coins = {env._env.player_coins(env.current_player)}")
            next_obs, reward, done, truncated, info = env.step(action)     
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
                    value_pred=agents[current_player_index]["agent"].mcts.root.value_estimate,
                    value_mcts=agents[current_player_index]["agent"].mcts.root.value
                )

            if done:
                if buffer is not None:
                    buffer.compute_values()
                time_taken = time.time() - t00
                winner = agents[info["winning_player_index"]]["name"]
                return buffer, time_taken, game_id, winner

            if buffer is not None and buffer.isfull:
                if done:
                    buffer.compute_values()
                    winning_player_index = info["winning_player_index"]
                    winner = agents[winning_player_index]["name"] if winning_player_index is not None else None
                else:
                    winner = None
                time_taken = time.time() - t00
                # del env
                # del agents
                return buffer, time_taken, game_id, winner
            obs = next_obs


class Pit:
    def __init__(
        self,
        env,
        use_ray,
        games_per_addition=30,
    ):
        self._env = env
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
                        worker_id=i
                    ) for i in range(n_workers)
                ]
            )
        else:
            self.actor = MCTSActor(
                worker_id=0
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
        return {i: {"name": agent_name, "agent": self._agents[agent_name]["agent"]} for i, agent_name in enumerate(agent_pair)}

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
                lambda a, v: a.play_game.remote(v, self._env, start_state, self.get_agent_dict(agent_pairs_per_game_id[v]), None),
                game_ids,
            )
        else:
            actor_generator = (self.actor.play_game(v, self._env, start_state, self.get_agent_dict(agent_pairs_per_game_id[v]), None) for v in game_ids)

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
    GAMES_PER_ITERATION = 2000
    # CARD_INFO_PATH = "card_info_machi_koro_2.yaml"
    CARD_INFO_PATH = "card_info_machi_koro_2_quick_game.yaml"
    TRAIN_VAL_SPLIT = 0.2
    LR = 0.001
    WEIGHT_DECAY = 1e-5
    N_PLAYERS = 2
    BUFFER_CAPACITY = 10

    use_ray = True
    # with open("src/checkpoints/9.pkl", "rb") as file:
    #     buffer = pickle.load(file)
    # model = torch.load("checkpoints_3_static/2.pt")
    # breakpoint()
    if use_ray:
        ray.init()

    env_cls = GymMachiKoro2
    env_kwargs = {"n_players": N_PLAYERS, "card_info_path": CARD_INFO_PATH}
    env = env_cls(**env_kwargs)

    pit = Pit(
        env=env,
        use_ray=use_ray,
        games_per_addition=30,
    )

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

    current_agent = MCTSAgent(
        env=env,
        num_mcts_sims=MCTS_SIMULATIONS,
        c_puct=PUCT,
        pvnet=PVNet(env, mlflow_experiment_name=f"Self Play {int(time.time())}"),
        dirichlet_to_root_node=True,
    )
    if use_ray:
        n_workers = int(ray.available_resources()["CPU"] - 1)
        print(f"Found {n_workers} CPU cores available for workers")
        actor_pool = ActorPool(
            [
                ray.remote(MCTSActor).remote(
                    worker_id=i
                ) for i in range(n_workers)
            ]
        )
    else:
        actor = MCTSActor(
            worker_id=0
        )

    checkpoint_dir = "checkpoints/" + str(datetime.datetime.now())
    os.makedirs(checkpoint_dir)

    # the best agent is considered the randomly initialized agent at the start
    best_agent = copy.deepcopy(current_agent)
    best_agent_name = "init_random_agent"
    pit.add_initial_agent(best_agent_name, best_agent)

    # the current agent is initially random as well
    current_agent_name = "agent_0"
    training_agents = {
            0: {"name": current_agent_name, "agent": current_agent},
            1: {"name": best_agent_name, "agent": best_agent},
        }
    assert len(training_agents) == N_PLAYERS, "number of agents should be equal to number of players"

    buffer_log_path=f"{checkpoint_dir}/buffers.h5"
    subset_rules = {}
    iterations_since_new_best_agent = 0

    with SelfPlayLogger(buffer_log_path=buffer_log_path) as selfplaylogger:
        for i in range(100):
            subset_rules[f"iteration_{i}"] = 1.0
            game_ids = np.arange(GAMES_PER_ITERATION)

            selfplaylogger.start_iteration(
                game_ids=game_ids,
                val_fraction=TRAIN_VAL_SPLIT,
                subset_rules=subset_rules,
                agent_names=[agent["name"] for agent in training_agents.values()],
            )

            buffer = Buffer(
                observation_space=env.observation_space,
                action_space=env.action_space,
                capacity=BUFFER_CAPACITY,
            )

            print("WARNING: USING VERY SIMPLE START STATE")
            env.reset()
            state = env.state_dict()
            state["player_info"]["player 0"]["coins"] = 29
            state["player_info"]["player 1"]["coins"] = 29
            state["marketplace"]["landmark"]["pos_0"]["card"] = "Launch Pad"
            state["marketplace"]["landmark"]["pos_1"]["card"] = "Loan Office"
            state["marketplace"]["landmark"]["pos_2"]["card"] = "Soda Bottling Plant"
            state["marketplace"]["landmark"]["pos_3"]["card"] = "Charterhouse"
            state["marketplace"]["landmark"]["pos_4"]["card"] = "Temple"
            GAME_START_STATE = env.state_dict_to_array(state)

            if use_ray:
                actor_generator = actor_pool.map_unordered(
                    lambda a, v: a.play_game.remote(v, env, GAME_START_STATE, training_agents, buffer),
                    game_ids,
                )
            else:
                actor_generator = (actor.play_game(v, env, GAME_START_STATE, training_agents, buffer) for v in game_ids)
                
            for filled_buffer, game_time_taken, game_id, winner in actor_generator:
                selfplaylogger.log_game(filled_buffer, game_time_taken, game_id, winner)
            
            if selfplaylogger.completed_games_current_iteration == 0:
                print(f"completed games this iteration is 0, pitting is not possible, skipping pitting computation and model trainig")
                continue
            
            wins = selfplaylogger.wins_current_iteration
            current_agent_win_ratio = wins[current_agent_name] / (wins[current_agent_name] + wins[best_agent_name])
            
            if current_agent_win_ratio > 0.55 and iterations_since_new_best_agent > 0:
                print(f"new best agent found in iteration {i}, win ratio: {current_agent_win_ratio}")
                print("computing elo ratings")
                pit.add_agent_and_compute_ratings(copy.deepcopy(current_agent), f"agent_{i}", start_state=GAME_START_STATE)
                pit.print_rankings()
                best_agent = copy.deepcopy(current_agent)
                best_agent_name = copy.deepcopy(current_agent_name)
                training_agents[1] = {"name": best_agent_name, "agent": best_agent}
                iterations_since_new_best_agent = 0
                selfplaylogger.log_agent_update()
                for iteration in subset_rules.keys():
                    subset_rules[iteration] = 0.0
                subset_rules[f"iteration_{i}"] = 1.0
            else:
                print(f"current agent win ratio: {current_agent_win_ratio}")

            try:
                current_agent.mcts.pvnet.train_hdf5(
                    batch_size=BATCH_SIZE,
                    epochs=PVNET_TRAIN_EPOCHS,
                    train_val_split=TRAIN_VAL_SPLIT,
                    lr=LR,
                    weight_decay=WEIGHT_DECAY,
                    hdf5_file_path=buffer_log_path,
                    subset_rules=subset_rules,
                    reset_weights=True,
                )
                current_agent.mcts.pvnet.save(f"{checkpoint_dir}/model_{i}.pt")
                current_agent_name = f"agent_{i}"
                training_agents[0] = {"name": current_agent_name, "agent": current_agent}
            except NotEnoughDataError:
                print("Not enough data to train model, skipping training")
                current_agent.mcts.pvnet.save(f"{checkpoint_dir}/model_{i}_same_as_before_due_to_not_enough_data.pt") 
            iterations_since_new_best_agent += 1


if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    main()
    # profiler.disable()
    # profiler.dump_stats("self_play.prof")