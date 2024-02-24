from env import GymMachiKoro, MachiKoro
from multielo import MultiElo
from random_agent import RandomAgent
from mcts_agent import MCTSAgent
from buffer import Buffer
import gym
import copy
import numpy as np

# buffers = {
#     player: Buffer(gamma=1, observation_space=env.observation_space, action_space=env.action_space, capacity=1000) for player in agents
# }

# can use a single buffer? Then cannot compute the values in montecarlo way, need to do sarsa. So
# that might not be ideal for MCTS, how does Alphazero do this? Does it bootstrap based on sarsa using search statistics or is the value of the search statistics used as the target?

def get_trajectories(env, agents):
    game = 0
    steps = 0
    # buffer = Buffer(gamma=1, observation_space=env.observation_space, action_space=env.action_space, capacity=1000)

    elo = MultiElo()

    player_elo = [1000 for _ in range(env.n_players)]
    wins = [0 for _ in range(env.n_players)]

    while True:
        done = False
        obs, info = env.reset()
        [agent.reset(info["state"]) for agent in agents.values()]

        while not done:
            steps+=1

            action, probs = agents[env.current_player].compute_action(obs, info["state"])

            print(f"game: {game} | steps: {steps} | player {env.current_player_index} played {env._action_idx_to_str[action]} | coins = {env.player_info[env.current_player].coins}")
            print(f"probs {' '.join(map(str, probs.round(2)))}")
            next_obs, reward, done, truncated, info = env.step(action)    

            # buffer.add(
            #     obs=obs,
            #     action=action,
            #     reward=reward,
            #     next_obs=next_obs,
            #     done=done,
            #     probs=probs
            #     )

            if done:
                ranking = [1 if player == env.current_player else 2 for player in env.player_order]
                for i, rank in enumerate(ranking):
                    if rank == 1:
                        wins[i] += 1
                player_elo = elo.get_new_ratings(player_elo, result_order=ranking)
                print("=================")
                print("=================")
                print("=================")
                print(f"game {game} | elo: {player_elo}, wins: {wins}")
                print("=================")
                print("=================")
                print("=================")
                game += 1
                # return buffer
                # if env.current_player == "player 0":
                #     return buffer

            obs = copy.deepcopy(next_obs)
            # if buffer.isfull:
            #     return buffer

# game 9999 | elo: [1035.11131277  995.44385084  985.17885556  984.26598084], wins: [2607, 2594, 2433, 2366]
# print("player 0")
# buffers["player 0"].compute_values()
# [print(value, reward, done) for value, reward, done in zip(buffers["player 0"]._values, buffers["player 0"]._rewards, buffers["player 0"]._dones)]
# # print(buffers["player 0"]._rewards)

# print("player 1")
# buffers["player 1"].compute_values()
# [print(value, reward, done) for value, reward, done in zip(buffers["player 1"]._values, buffers["player 1"]._rewards, buffers["player 1"]._dones)]
# print(buffer._obss[0] == buffer._obss[5])
if __name__ == "__main__":
    env = MachiKoro(n_players=2)
    env = GymMachiKoro(env)

    agents = {
        # f"player 0": RandomAgent(env.observation_space, env.action_space),
        f"player 0": MCTSAgent(env, num_mcts_sims=100, c_puct=2),
        f"player 1": RandomAgent(env.observation_space, env.action_space),
    }
    agents["player 0"].update_pvnet("src/checkpoints/9.pt")
    buffer = get_trajectories(env, agents)
    buffer.compute_values()
    obss, actions, rewards, next_obss, dones, player_ids, action_masks, values, probs = buffer.sample(100)
    # print(buffer._obss[0] == buffer._obss[5])
    breakpoint()