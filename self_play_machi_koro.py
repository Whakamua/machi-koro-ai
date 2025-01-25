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
from mcts_agent import PVNet
from multielo import MultiElo
import datetime
import copy
import h5py
from mcts_agent import NotEnoughDataError
from collections import deque
import logging

def setup_logger(logging_file_path: str):
    """Sets up a centralized logger to log to both the console and a file."""
    logger = logging.getLogger("SelfPlayFileLogger")
    logger.setLevel(logging.INFO)  # Set the logging level

    # Check if the logger already has handlers to avoid duplicates
    if not logger.handlers:
        # File handler (writes to selfplay.log)
        file_handler = logging.FileHandler(logging_file_path)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(file_handler)

        # Stream handler (prints to stdout)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(console_handler)

    return logger

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
    def __init__(self, checkpoint_dir: str, env, hyperparams):
        self.iteration = -1
        self.steps_collected_per_iteration = {"val": {}, "train": {}}
        self.completed_games_per_iteration = {"val": {}, "train": {}}
        self.buffer_log_path = checkpoint_dir + "/buffers.h5"
        self.logger_state_path = checkpoint_dir + "/logger_state.pkl"
        self.number_of_agent_updates = 0
        self.env = env
        self.hyperparams = hyperparams
        self.best_agent_name = "agent_0_init_copy"
        self.current_agent_name = "agent_0"
        self.agents = {
            "agent_0_init_copy": {
                "elo": 1000,
                "wins": 0,
                "subset_rules": {},
                "pittable": True
            },
            "agent_0": {
                "elo": 1000,
                "wins": 0,
                "subset_rules": {},
                "pittable": False
            }
        }
        self.iterations_since_new_best_agent = 0

        # Get the centralized logger
        self.file_logger = logging.getLogger("SelfPlayFileLogger")


    def save_logger_state(self):
        with open(self.logger_state_path, "wb") as file:
            pickle.dump({k: v for k, v in self.__dict__.items() if k not in ["h5f"]}, file)

    def load_logger_state(self):
        with open(self.logger_state_path, "rb") as file:
            self.__dict__.update(pickle.load(file))

        # deleting iteration groups that were not completed
        if f"iteration_{self.iteration+1}" in self.h5f["train"]:
            del self.h5f["train"][f"iteration_{self.iteration+1}"]
        if f"iteration_{self.iteration+1}" in self.h5f["val"]:
            del self.h5f["val"][f"iteration_{self.iteration+1}"]

    def __enter__(self):
        #check if the buffer log file exists
        if os.path.exists(self.buffer_log_path):
            self.h5f = h5py.File(self.buffer_log_path, "r+")
        else:
            self.h5f = h5py.File(self.buffer_log_path, "w")
            self.h5f.create_group("train")
            self.h5f.create_group("val")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.h5f.close()

    def current_agent_win_ratio_current_iteration(self):
        wins_current_agent = self.wins_current_iteration[self.current_agent_name]
        total_wins = sum(self.wins_current_iteration.values())
        return wins_current_agent / total_wins if total_wins > 0 else 0

    def start_iteration(self, game_ids: int, val_fraction, agent_names_for_iteration: list[str], subset_rule_for_current_agent_this_iteration: float):
        self.iteration += 1
        self.new_iteration = True

        self.agents[self.current_agent_name]["subset_rules"][f"iteration_{self.iteration}"] = subset_rule_for_current_agent_this_iteration
        
        self.iteration_start_time = time.time()
        self.number_of_games_for_current_iteration = len(game_ids)
        self.val_indices = np.random.choice(game_ids, int(len(game_ids) * val_fraction), replace=False)

        self.games_played = {}
        self.games_time_taken_deque = deque(maxlen=100)

        self.steps_collected_per_iteration["train"][f"iteration_{self.iteration}"] = 0
        self.steps_collected_per_iteration["val"][f"iteration_{self.iteration}"] = 0
        self.completed_games_per_iteration["train"][f"iteration_{self.iteration}"] = 0
        self.completed_games_per_iteration["val"][f"iteration_{self.iteration}"] = 0

        self.h5f["train"].create_group(f"iteration_{self.iteration}")
        self.h5f["val"].create_group(f"iteration_{self.iteration}")

        self.wins_current_iteration = {agent_name: 0 for agent_name in agent_names_for_iteration}
        self.games_with_no_winner_current_iteration = 0

        self.game_lengths_current_iteration = []

    def log_game(self, buffer, game_time_taken, game_id, winner):
        self.games_played[game_id] = {"time_taken": game_time_taken, "winner": winner}

        self.games_time_taken_deque.append(game_time_taken)

        if "columns" not in self.h5f.attrs:
            self.h5f.attrs["columns"] = list(buffer.flattened_column_names_and_types_dict.keys())

        if winner is not None:
            buffer_np = buffer.export_flattened()
            self.game_lengths_current_iteration.append(buffer.size)
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

        self.log_summary(final=False)

    def log_agent_update(self, new_current_agent_name, best_agent_name):
        self.number_of_agent_updates += 1

        assert best_agent_name in self.agents, f"best agent {best_agent_name} not in agents"
        assert new_current_agent_name not in self.agents, f"new current agent {new_current_agent_name} already in agents"

        if best_agent_name == self.best_agent_name:
            self.iterations_since_new_best_agent += 1
        else:
            self.iterations_since_new_best_agent = 0

        self.agents[new_current_agent_name] = {
            "elo": 1000,
            "wins": 0,
            "subset_rules": self.agents[self.current_agent_name]["subset_rules"],
            "pittable": False
        }

        self.current_agent_name = new_current_agent_name
        self.best_agent_name = best_agent_name


    @property
    def completed_games_current_iteration(self):
        return self.completed_games_per_iteration["train"][f"iteration_{self.iteration}"] + self.completed_games_per_iteration["val"][f"iteration_{self.iteration}"]

    def log_summary(self, final=False):
        avg_game_time = np.mean(self.games_time_taken_deque) if self.games_time_taken_deque else 0
        estimated_iteration_time_left = avg_game_time * (self.number_of_games_for_current_iteration - len(self.games_played))

        train_games_for_next_training_run = int(
            sum([
                self.completed_games_per_iteration["train"][iteration] * fraction
                for iteration, fraction in self.agents[self.current_agent_name]["subset_rules"].items()
            ])
        )
        val_games_for_next_training_run = int(
            sum([
                self.completed_games_per_iteration["val"][iteration] * fraction
                for iteration, fraction in self.agents[self.current_agent_name]["subset_rules"].items()
            ])
        )

        train_steps_for_next_training_run = int(
            sum([
                self.steps_collected_per_iteration["train"][iteration] * fraction
                for iteration, fraction in self.agents[self.current_agent_name]["subset_rules"].items()
            ])
        )
        val_steps_for_next_training_run = int(
            sum([
                self.steps_collected_per_iteration["val"][iteration] * fraction
                for iteration, fraction in self.agents[self.current_agent_name]["subset_rules"].items()
            ])
        )

        summary = f"""
##################################################################################
Iteration: {self.iteration}
Agent Updates: {self.number_of_agent_updates}
Games Played: {len(self.games_played)}/{self.number_of_games_for_current_iteration}
Games With No Winner/Completed Games: {self.games_with_no_winner_current_iteration}/{len(self.games_played)}
Estimated Iteration Time Left: {estimated_iteration_time_left:.0f}s
Last Game Time Taken: {self.games_time_taken_deque[-1] if self.games_time_taken_deque else 0:.4f}s
Wins: {self.wins_current_iteration}
Next Training Run: {train_games_for_next_training_run} games, {train_steps_for_next_training_run} steps
Next Validation Run: {val_games_for_next_training_run} games, {val_steps_for_next_training_run} steps
Mean Game Length: {np.mean(self.game_lengths_current_iteration):.2f} steps
Standard Deviation Game Length: {np.std(self.game_lengths_current_iteration):.2f} steps
Minimum Game Length: {np.min(self.game_lengths_current_iteration) if self.game_lengths_current_iteration else 0} steps
Maximum Game Length: {np.max(self.game_lengths_current_iteration) if self.game_lengths_current_iteration else 0} steps
Agent Elo Ratings:
"""
        for agent, info in self.agents.items():
            if info["pittable"]:
                summary += f"{agent} | elo: {info['elo']} | wins: {info['wins']}\n"
        summary += """
##################################################################################
"""
        if not self.new_iteration:
            self.clear_stdout()
        else:
            self.new_iteration = False

        sys.stdout.write(summary + "\n")
        self.n_agents_last_summary = len([agent for agent in self.agents.values() if agent["pittable"]])
        sys.stdout.flush()

        if final:
            self.clear_stdout()
            sys.stdout.flush()
            self.file_logger.info(summary + "\n")
            self.n_agents_last_summary = len([agent for agent in self.agents.values() if agent["pittable"]])

    def clear_stdout(self):
        for _ in range(19+self.n_agents_last_summary):
            sys.stdout.write("\033[F\033[K")

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
            buffer=None,
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
                return buffer, time_taken, game_id, winner
            obs = next_obs


def pit_agents(
    existing_agents,
    new_agent,
    n_games,
    env,
    checkpoint_dir,
    start_state=None,
    use_ray=True,
):
    existing_agents[new_agent]["pittable"] = True
    game_ids = np.arange(n_games)

    agent_objects = {
        agent_name: MCTSAgent.load_from_pickle(f"{checkpoint_dir}/{agent_name}.pickle")
        for agent_name in existing_agents if existing_agents[agent_name]["pittable"]
    }

    def get_agent_dict(agent_pair, agent_objects):
        return {i: {"name": agent_name, "agent": agent_objects[agent_name]} for i, agent_name in enumerate(agent_pair)}

    agent_pairs_per_game_id = {
        game_id: [new_agent, np.random.choice([agent for agent, agent_info in existing_agents.items() if agent != new_agent and agent_info["pittable"]], 1)[0]]
        for game_id in game_ids
    }

    # shuffling agent pairs to avoid bias, since order is the order in which agents start in 
    # the game. This is important because the first player has a slight advantage in Machi Koro
    for agent_pair in agent_pairs_per_game_id.values():
        random.shuffle(agent_pair)

    if use_ray:
        actor_pool = create_ray_actor_pool()
        actor_generator = actor_pool.map_unordered(
            lambda a, v: a.play_game.remote(v, env, start_state, get_agent_dict(agent_pairs_per_game_id[v], agent_objects)),
            game_ids,
        )
    else:
        actor = MCTSActor(
            worker_id=0
        )
        actor_generator = (actor.play_game(v, env, start_state, get_agent_dict(agent_pairs_per_game_id[v], agent_objects)) for v in game_ids)

    elo = MultiElo()

    for _, _, game_id, winner in actor_generator:
        if winner is None:
            print(f"WARNING: [Pit] game {game_id} had no winner")
            continue
        else:
            existing_agents[winner]["wins"] += 1

            # compute new elo ratings
            agents_playing = agent_pairs_per_game_id[game_id]
            current_ratings = [existing_agents[agent]["elo"] for agent in agents_playing]

            ranking = [1 if agent == winner else 2 for agent in agents_playing]
            new_ratings = elo.get_new_ratings(current_ratings, result_order=ranking)

            for i, agent in enumerate(agents_playing):
                existing_agents[agent]["elo"] = new_ratings[i]
    return existing_agents

def create_ray_actor_pool():
    n_workers = int(ray.available_resources()["CPU"] - 1)
    print(f"Found {n_workers} CPU cores available for workers")
    return ActorPool(
        [
            ray.remote(MCTSActor).remote(
                worker_id=i
            ) for i in range(n_workers)
        ]
    )

def main(checkpoint_dir=None):
    
    if checkpoint_dir is None:
        starting_from_checkpoint = False
        checkpoint_dir = "checkpoints/" + str(datetime.datetime.now())
        os.makedirs(checkpoint_dir)
    else:
        starting_from_checkpoint = True

    setup_logger(checkpoint_dir + "/selfplay.log")

    logger = logging.getLogger("SelfPlayFileLogger")

    if starting_from_checkpoint:
        logger.info(f"Starting self play from checkpoint {checkpoint_dir}")
    else:
        logger.info("Starting self play...")

    hyperparams = {
        "MCTS_SIMULATIONS": 10,
        "PUCT": 2,
        "PVNET_TRAIN_EPOCHS": 30,
        "BATCH_SIZE": 64,
        "GAMES_PER_ITERATION": 25,
        # "CARD_INFO_PATH": "card_info_machi_koro_2.yaml",
        "CARD_INFO_PATH": "card_info_machi_koro_2_quick_game.yaml",
        "TRAIN_VAL_SPLIT": 0.2,
        "LR": 0.001,
        "WEIGHT_DECAY": 1e-5,
        "N_PLAYERS": 2,
        "BUFFER_CAPACITY": 1000,
        "GAME_START_STATE": None
    }

    use_ray = True
    if use_ray:
        ray.init()

    env_cls = GymMachiKoro2
    env_kwargs = {"n_players": hyperparams["N_PLAYERS"], "card_info_path": hyperparams["CARD_INFO_PATH"]}
    env = env_cls(**env_kwargs)

    if use_ray:
            actor_pool = create_ray_actor_pool()
    else:
        actor = MCTSActor(
            worker_id=0
        )

    with SelfPlayLogger(checkpoint_dir=checkpoint_dir, env=env, hyperparams=hyperparams) as selfplaylogger:
        current_agent_name = selfplaylogger.current_agent_name
        current_agent = MCTSAgent(
            env=env,
            num_mcts_sims=hyperparams["MCTS_SIMULATIONS"],
            c_puct=hyperparams["PUCT"],
            pvnet=PVNet(env, mlflow_experiment_name=f"Self Play {int(time.time())}"),
            dirichlet_to_root_node=True,
        )
        best_agent_name = selfplaylogger.best_agent_name
        best_agent = copy.deepcopy(current_agent)
        best_agent.pickle(f"{checkpoint_dir}/{best_agent_name}.pickle")

        if starting_from_checkpoint:
            selfplaylogger.load_logger_state()
            current_agent_name = selfplaylogger.current_agent_name
            current_agent = MCTSAgent.load_from_pickle(f"{checkpoint_dir}/{current_agent_name}.pickle")
            best_agent_name = selfplaylogger.best_agent_name
            best_agent = MCTSAgent.load_from_pickle(f"{checkpoint_dir}/{best_agent_name}.pickle")
            env = selfplaylogger.env
            hyperparams = selfplaylogger.hyperparams

        for i in range(100):
            training_agents = {
                0: {"name": selfplaylogger.current_agent_name, "agent": current_agent},
                1: {"name": selfplaylogger.best_agent_name, "agent": best_agent},
            }
            assert len(training_agents) == hyperparams["N_PLAYERS"], "number of agents should be equal to number of players"

            game_ids = np.arange(hyperparams["GAMES_PER_ITERATION"])

            selfplaylogger.start_iteration(
                game_ids=game_ids,
                val_fraction=hyperparams["TRAIN_VAL_SPLIT"],
                agent_names_for_iteration=[agent["name"] for agent in training_agents.values()],
                subset_rule_for_current_agent_this_iteration=1.0,
            )

            buffer = Buffer(
                observation_space=env.observation_space,
                action_space=env.action_space,
                capacity=hyperparams["BUFFER_CAPACITY"],
            )

            if use_ray:
                actor_generator = actor_pool.map_unordered(
                    lambda a, v: a.play_game.remote(v, env, hyperparams["GAME_START_STATE"], training_agents, buffer),
                    game_ids,
                )
            else:
                actor_generator = (actor.play_game(v, env, hyperparams["GAME_START_STATE"], training_agents, buffer) for v in game_ids)

            try:
                for filled_buffer, game_time_taken, game_id, winner in actor_generator:
                    selfplaylogger.log_game(filled_buffer, game_time_taken, game_id, winner)
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt: stopping self play for current iteration")
                if use_ray:
                    for actor in actor_pool._future_to_actor.values():
                        ray.kill(actor[1])
                    actor_pool = create_ray_actor_pool()
            
            if selfplaylogger.completed_games_current_iteration == 0:
                logger.info(f"completed games this iteration is 0, pitting is not possible, skipping pitting computation and model trainig")
                continue
            
            current_agent_win_ratio = selfplaylogger.current_agent_win_ratio_current_iteration()
            if current_agent_win_ratio > 0.55 and selfplaylogger.iterations_since_new_best_agent > 0:
                logger.info(f"new best agent found in iteration {selfplaylogger.iteration}, win ratio: {current_agent_win_ratio}")
                logger.info("computing elo ratings")
                selfplaylogger.agents = pit_agents(
                    existing_agents=selfplaylogger.agents,
                    new_agent=current_agent_name,
                    n_games=30,
                    env=env,
                    checkpoint_dir=checkpoint_dir,
                    start_state=hyperparams["GAME_START_STATE"],
                    use_ray=use_ray,
                )
                best_agent = copy.deepcopy(current_agent)
                best_agent_name = copy.deepcopy(current_agent_name)
                
                subset_rules = selfplaylogger.agents[current_agent_name]["subset_rules"]
                for i, iteration in enumerate(subset_rules.keys()):
                    if i < len(subset_rules) - 1:
                        subset_rules[iteration] = 0.0
                    else:
                        subset_rules[iteration] = 1.0
                selfplaylogger.agents[current_agent_name]["subset_rules"] = subset_rules
            else:
                logger.info(f"current agent win ratio: {current_agent_win_ratio}")

            try:
                current_agent.mcts.pvnet.train_hdf5(
                    batch_size=hyperparams["BATCH_SIZE"],
                    epochs=hyperparams["PVNET_TRAIN_EPOCHS"],
                    train_val_split=hyperparams["TRAIN_VAL_SPLIT"],
                    lr=hyperparams["LR"],
                    weight_decay=hyperparams["WEIGHT_DECAY"],
                    hdf5_file_path=selfplaylogger.buffer_log_path,
                    subset_rules=selfplaylogger.agents[current_agent_name]["subset_rules"],
                    reset_weights=True,
                )
                current_agent_name = f"agent_{selfplaylogger.iteration+1}"
                current_agent.pickle(f"{checkpoint_dir}/{current_agent_name}.pickle")
            except NotEnoughDataError:
                logger.info("Not enough data to train model, skipping training")

            selfplaylogger.log_agent_update(current_agent_name, best_agent_name)
            selfplaylogger.log_summary(final=True)
            selfplaylogger.save_logger_state()


if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    from fire import Fire
    Fire(main)
    # profiler.disable()
    # profiler.dump_stats("self_play.prof")