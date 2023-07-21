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
    buffer = Buffer(gamma=1, observation_space=env.observation_space, action_space=env.action_space, capacity=1000)

    elo = MultiElo()

    player_elo = [1000 for _ in range(env.n_players)]
    wins = [0 for _ in range(env.n_players)]

    while True:
        done = False
        obs, info = env.reset()
        [agent.reset(info["state"]) for agent in agents.values()]
        # [buffer.reset() for buffer in buffers.values()]
        while not done:

            # if not np.array_equal(obs["action_mask"], env.action_mask):
            #     breakpoint()

            action = agents[env.current_player].compute_action(obs, info["state"])

            # if action > 1 and env.current_stage == "diceroll":
            #     breakpoint()

            next_obs, reward, done, truncated, info = env.step(action)        

            buffer.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)

            if done:
                ranking = [1 if player == env.current_player else 2 for player in env.player_order]
                for i, rank in enumerate(ranking):
                    if rank == 1:
                        wins[i] += 1
                player_elo = elo.get_new_ratings(player_elo, result_order=ranking)
                print(f"game {game} | elo: {player_elo}, wins: {wins}")
                game += 1

            obs = copy.deepcopy(next_obs)
            if buffer.isfull:
                return buffer

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
        # f"player {i}": RandomAgent(env.observation_space, env.action_space) for i in range(env.n_players)
        f"player 0": MCTSAgent(copy.deepcopy(env)),
        f"player 1": RandomAgent(env.observation_space, env.action_space)
    }
    buffer = get_trajectories(env, agents)
    buffer.compute_values()
    obss, actions, rewards, next_obss, dones, player_ids, action_masks, values = buffer.sample(100)
    # print(buffer._obss[0] == buffer._obss[5])
    breakpoint()