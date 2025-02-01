
from env_machi_koro_2 import GymMachiKoro2
from multielo import MultiElo
from manual_agent import ManualAgent
from mcts_agent import MCTSAgent
import copy
import time

def main():
    N_PLAYERS = 2
    # CARD_INFO_PATH = "card_info_machi_koro_2.yaml"
    # opponent_path = "not_available_yet:("

    CARD_INFO_PATH = "card_info_machi_koro_2_quick_game.yaml"
    opponent_path = "trained_agents/machi_koro_2_quick_game/agent_5_elo_1100.pickle"


    time_start = time.time()
    env_cls = GymMachiKoro2
    env_kwargs = {"n_players": N_PLAYERS, "card_info_path": CARD_INFO_PATH}
    env = env_cls(**env_kwargs, print_info=False)

    elo = MultiElo()

    player_elo = [1000 for _ in range(env.n_players)]
    wins = [0 for _ in range(env.n_players)]

    opponent = MCTSAgent.load_from_pickle(opponent_path)
    opponent.mcts._thinking_time = 15

    agents = {
        "player 0": ManualAgent(env),
        "player 1": opponent
    }
    
    cumulative_steps = 0
    n_games = 100

    for game in range(n_games):
        action_history = []
        done = False
        obs, info = env.reset()
        [agent.reset(obs) for agent in agents.values()]

        steps = 0
        while not done:
            current_player = copy.deepcopy(env.current_player)
            action, probs = agents[current_player].compute_action(obs)

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