import pytest
from .env import MachiKoro, GymMachiKoro
from collections import deque
import random
import numpy as np
import pprint

@pytest.fixture
def env():
    return MachiKoro(n_players=4)

@pytest.fixture
def gymenv(env):
    return GymMachiKoro(env)

def test_env_init(env):
    for player, info in env._player_info.items():
        assert info.cards == {
            'Wheat Field': 1, 'Ranch': 0, 'Flower Orchard': 0, 'Forest': 0, 'Mackerel Boat': 0, 
            'Apple Orchard': 0, 'Tuna Boat': 0, 'General Store': 0, 'Bakery': 1, 
            'Demolition Company': 0, 'Flower Shop': 0, 'Cheese Factory': 0, 'Furniture Factory': 0,
            'Soda Bottling Plant': 0, 'Fruit and Vegetable Market': 0, 
            'Sushi Bar': 0, 'Café': 0, 'French Restaurant': 0, 'Pizza Joint': 0, 
            'Family Restaurant': 0, "Member's Only Club": 0, 'Stadium': 0, 'Publisher': 0, 
            'Tax Office': 0, 'Tech Startup': 0, 
            'Harbor': False, 'Train Station': False, 'Shopping Mall': False, 
            'Amusement Park': False, 'Moon Tower': False, 'Airport': False
        }
        assert info.coins == 3
        assert info.landmarks == 0
    
def test_city_hall(env):
    for player_info in env._player_info.values():
        player_info.coins = 0

    for player in env._player_info.keys():
        env._current_player = player
        env.dice_roll(1)
        assert env._player_info[player].coins >= 1

def test_get_one_coin_cards(env):
    for card in ["Wheat Field", "Ranch", "Flower Orchard", "Forest"]:
        for player in env._player_info.keys():
            coins = env._player_info[player].coins
            env._activate_card(player, card)
            assert coins + 1 == env._player_info[player].coins

def test_mackerel_boat(env):
    for player in env._player_info.keys():
        coins = env._player_info[player].coins
        env._activate_card(player, "Mackerel Boat")
        assert coins == env._player_info[player].coins

        env._player_info[player]._add_card("Harbor")

        env._activate_card(player, "Mackerel Boat")
        assert coins + 3 == env._player_info[player].coins

def test_apple_orchard(env):
    for player in env._player_info.keys():
        coins = env._player_info[player].coins
        env._activate_card(player, "Apple Orchard")
        assert coins + 3 == env._player_info[player].coins

def test_tuna_boat(env):
     for player in env._player_info.keys():
        coins = env._player_info[player].coins
        env._activate_card(player, "Tuna Boat")
        assert coins == env._player_info[player].coins

        env._player_info[player]._add_card("Harbor")

        for _ in range(100):
            coins = env._player_info[player].coins
            env._activate_card(player, "Tuna Boat")
            assert 2 <= env._player_info[player].coins - coins <= 12

def test_general_store(env):
    for player in env._player_info.keys():
        env._current_player = "player x"

        coins = env._player_info[player].coins
        env._activate_card(player, "General Store")
        assert coins == env._player_info[player].coins

        env._current_player = player
        
        coins = env._player_info[player].coins
        env._activate_card(player, "General Store")
        assert coins + 2 == env._player_info[player].coins

        env._player_info[player]._add_card("Harbor")
        coins = env._player_info[player].coins
        env._activate_card(player, "General Store")
        assert coins + 2 == env._player_info[player].coins

        env._player_info[player]._add_card("Moon Tower")
        coins = env._player_info[player].coins
        env._activate_card(player, "General Store")
        assert coins == env._player_info[player].coins

def test_bakery(env):
    for player in env._player_info.keys():
        env._current_player = "player x"

        coins = env._player_info[player].coins
        env._activate_card(player, "Bakery")
        assert coins == env._player_info[player].coins

        env._current_player = player
        
        coins = env._player_info[player].coins
        env._activate_card(player, "Bakery")
        assert coins + 1 == env._player_info[player].coins

def test_demolition_company(env):
    for player in env._player_info.keys():
        env._current_player = "player x"

        coins = env._player_info[player].coins
        env._activate_card(player, "Demolition Company")
        assert coins == env._player_info[player].coins

        env._current_player = player

        coins = env._player_info[player].coins
        env._activate_card(player, "Demolition Company")
        assert coins == env._player_info[player].coins

        env._player_info[player]._add_card("Harbor")
        coins = env._player_info[player].coins
        env._activate_card(player, "Demolition Company")
        assert coins + 8 == env._player_info[player].coins

        assert env._player_info[player].cards["Harbor"] == False

        env._player_info[player]._add_card("Harbor")
        env._player_info[player]._add_card("Train Station")
        coins = env._player_info[player].coins
        env._activate_card(player, "Demolition Company")
        assert coins + 8 == env._player_info[player].coins

        assert env._player_info[player].cards["Harbor"] == False
        assert env._player_info[player].cards["Train Station"] == True

        coins = env._player_info[player].coins
        env._activate_card(player, "Demolition Company")
        assert coins + 8 == env._player_info[player].coins

        assert env._player_info[player].cards["Harbor"] == False
        assert env._player_info[player].cards["Train Station"] == False

def test_flower_shop(env):
    for player in env._player_info.keys():
        env._current_player = player

        coins = env._player_info[player].coins
        env._activate_card(player, "Flower Shop")
        assert coins == env._player_info[player].coins

        env._player_info[player]._add_card("Flower Orchard")
        env._player_info[player]._add_card("Flower Orchard")
        coins = env._player_info[player].coins
        env._activate_card(player, "Flower Shop")
        assert coins + 2 == env._player_info[player].coins

        env._current_player = "player x"

        coins = env._player_info[player].coins
        env._activate_card(player, "Flower Shop")
        assert coins == env._player_info[player].coins

def test_cheese_factory(env):
    for player in env._player_info.keys():
        env._current_player = player

        coins = env._player_info[player].coins
        env._activate_card(player, "Cheese Factory")
        assert coins == env._player_info[player].coins

        env._player_info[player]._add_card("Ranch")
        env._player_info[player]._add_card("Ranch")

        coins = env._player_info[player].coins
        env._activate_card(player, "Cheese Factory")
        assert coins + 2 * 3 == env._player_info[player].coins

        env._current_player = "player x"

        coins = env._player_info[player].coins
        env._activate_card(player, "Cheese Factory")
        assert coins == env._player_info[player].coins

def test_furniture_factory(env):
    for player in env._player_info.keys():
        env._current_player = player

        coins = env._player_info[player].coins
        env._activate_card(player, "Furniture Factory")
        assert coins == env._player_info[player].coins

        env._player_info[player]._add_card("Forest")
        env._player_info[player]._add_card("Forest")

        coins = env._player_info[player].coins
        env._activate_card(player, "Furniture Factory")
        assert coins + 2 * 3 == env._player_info[player].coins

        env._current_player = "player x"

        coins = env._player_info[player].coins
        env._activate_card(player, "Furniture Factory")
        assert coins == env._player_info[player].coins

def test_soda_bottling_plant(env):

    n_players = 0
    for player in env._player_info.keys():
        n_players+=1
        env._current_player = player

        coins = env._player_info[player].coins
        env._activate_card(player, "Soda Bottling Plant")
        assert coins == env._player_info[player].coins

    for player in env._player_info.keys():
        env._player_info[player]._add_card("Sushi Bar")
        env._player_info[player]._add_card("Café")
        
    for player in env._player_info.keys():

        coins = env._player_info[player].coins
        env._activate_card(player, "Soda Bottling Plant")
        if player == env._current_player:
            assert coins + n_players*2 == env._player_info[player].coins

def test_fruit_and_vegetable_market(env):
    for player in env._player_info.keys():
        env._current_player = player

        coins = env._player_info[player].coins
        env._activate_card(player, "Fruit and Vegetable Market")
        assert coins + 2 == env._player_info[player].coins

        env._player_info[player]._add_card("Flower Orchard")
        env._player_info[player]._add_card("Apple Orchard")

        coins = env._player_info[player].coins
        env._activate_card(player, "Fruit and Vegetable Market")
        assert coins + 2 * 3 == env._player_info[player].coins

        env._current_player = "player x"

        coins = env._player_info[player].coins
        env._activate_card(player, "Fruit and Vegetable Market")
        assert coins == env._player_info[player].coins

def test_sushi_bar(env):
    player = list(env._player_info.keys())[0]
    env._current_player = player

    for _player in env._player_info.keys():

        coins_player = env._player_info[_player].coins
        coins_current_player = env._player_info[env._current_player].coins

        env._activate_card(_player, "Sushi Bar")
        if _player == env._current_player:
            assert coins_player == env._player_info[_player].coins
            assert coins_current_player == env._player_info[env._current_player].coins
        else:
            assert coins_player == env._player_info[_player].coins
            assert coins_current_player == env._player_info[env._current_player].coins


    env._player_info[env._current_player].coins = 20

    for _player in env._player_info.keys():
        env._player_info[_player]._add_card("Harbor")

        coins_player = env._player_info[_player].coins
        coins_current_player = env._player_info[env._current_player].coins

        env._activate_card(_player, "Sushi Bar")
        if _player == env._current_player:
            assert coins_player == env._player_info[_player].coins
            assert coins_current_player == env._player_info[env._current_player].coins
        else:
            assert coins_player + 3 == env._player_info[_player].coins
            assert coins_current_player - 3 == env._player_info[env._current_player].coins


def test_cafe(env):
    player = list(env._player_info.keys())[0]
    env._current_player = player

    for _player in env._player_info.keys():

        coins_player = env._player_info[_player].coins
        coins_current_player = env._player_info[env._current_player].coins

        env._activate_card(_player, "Café")
        if _player == env._current_player:
            assert coins_player == env._player_info[_player].coins
            assert coins_current_player == env._player_info[env._current_player].coins
        else:
            assert coins_player + 1 == env._player_info[_player].coins
            assert coins_current_player - 1 == env._player_info[env._current_player].coins

def test_french_restaurant(env):
    player = list(env._player_info.keys())[0]
    env._current_player = player

    for _player in env._player_info.keys():

        coins_player = env._player_info[_player].coins
        coins_current_player = env._player_info[env._current_player].coins

        env._activate_card(_player, "French Restaurant")
        if _player == env._current_player:
            assert coins_player == env._player_info[_player].coins
            assert coins_current_player == env._player_info[env._current_player].coins
        else:
            assert coins_player == env._player_info[_player].coins
            assert coins_current_player == env._player_info[env._current_player].coins


    env._player_info[env._current_player]._add_card("Harbor")
    env._player_info[env._current_player]._add_card("Airport")

    env._player_info[env._current_player].coins = 20

    for _player in env._player_info.keys():
        coins_player = env._player_info[_player].coins
        coins_current_player = env._player_info[env._current_player].coins

        env._activate_card(_player, "French Restaurant")
        if _player == env._current_player:
            assert coins_player == env._player_info[_player].coins
            assert coins_current_player == env._player_info[env._current_player].coins
        else:
            assert coins_player + 5 == env._player_info[_player].coins # french restaurant requires to transfer 5 coins but players in this test have only 3 coins.
            assert coins_current_player - 5 == env._player_info[env._current_player].coins

def test_pizza_joint(env):
    player = list(env._player_info.keys())[0]
    env._current_player = player

    for _player in env._player_info.keys():

        coins_player = env._player_info[_player].coins
        coins_current_player = env._player_info[env._current_player].coins

        env._activate_card(_player, "Pizza Joint")
        if _player == env._current_player:
            assert coins_player == env._player_info[_player].coins
            assert coins_current_player == env._player_info[env._current_player].coins
        else:
            assert coins_player + 1 == env._player_info[_player].coins
            assert coins_current_player - 1 == env._player_info[env._current_player].coins

def test_family_restaurant(env):
    player = list(env._player_info.keys())[0]
    env._current_player = player

    env._player_info[env._current_player].coins = 10

    for _player in env._player_info.keys():

        coins_player = env._player_info[_player].coins
        coins_current_player = env._player_info[env._current_player].coins

        env._activate_card(_player, "Family Restaurant")
        if _player == env._current_player:
            assert coins_player == env._player_info[_player].coins
            assert coins_current_player == env._player_info[env._current_player].coins
        else:
            assert coins_player + 2 == env._player_info[_player].coins
            assert coins_current_player - 2 == env._player_info[env._current_player].coins

def test_members_only_club(env):
    player = list(env._player_info.keys())[0]
    env._current_player = player

    for _player in env._player_info.keys():

        coins_player = env._player_info[_player].coins
        coins_current_player = env._player_info[env._current_player].coins

        env._activate_card(_player, "Member's Only Club")
        if _player == env._current_player:
            assert coins_player == env._player_info[_player].coins
            assert coins_current_player == env._player_info[env._current_player].coins
        else:
            assert coins_player == env._player_info[_player].coins
            assert coins_current_player == env._player_info[env._current_player].coins


    env._player_info[env._current_player]._add_card("Harbor")
    env._player_info[env._current_player]._add_card("Airport")
    env._player_info[env._current_player]._add_card("Train Station")

    for _player in env._player_info.keys():
        coins_player = env._player_info[_player].coins
        coins_current_player = env._player_info[env._current_player].coins

        env._activate_card(_player, "Member's Only Club")
        if _player == env._current_player:
            assert coins_player == env._player_info[_player].coins
            assert coins_current_player == env._player_info[env._current_player].coins
        else:
            assert coins_player + coins_current_player == env._player_info[_player].coins
            assert env._player_info[env._current_player].coins == 0

def test_stadium(env):
    coins = {}
    n_players = 0
    for player in env._player_info.keys():
        n_players += 1
        coins[player] = env._player_info[player].coins

    env._activate_card(env._current_player, "Stadium")

    for player in env._player_info.keys():
        if player == env._current_player:
            assert coins[player] + (n_players-1)*2 == env._player_info[player].coins
        else:
            assert coins[player] - 2 == env._player_info[player].coins

def test_publisher(env):
    coins = {}
    n_players = 0
    for player in env._player_info.keys():
        n_players += 1
        coins[player] = env._player_info[player].coins
        env._player_info[player]._add_card("Café")


    env._activate_card(env._current_player, "Publisher")

    for player in env._player_info.keys():
        if player == env._current_player:
            assert coins[player] + (n_players-1)*2 == env._player_info[player].coins
        else:
            assert coins[player] - 2 == env._player_info[player].coins # coins subtracted due to (default) bakery and Café.

def test_tax_office(env):
    env._current_player = list(env._player_info.keys())[0]
    rich_player = list(env._player_info.keys())[1]
    env._player_info[rich_player].coins = 23

    coins = {}
    for player in env._player_info.keys():
        coins[player] = env._player_info[player].coins

    env._activate_card(env._current_player, "Tax Office")
    for player in env._player_info.keys():
        if player == env._current_player:
            assert coins[player] + 11 == env._player_info[player].coins
        elif player == rich_player:
            assert coins[player] - 11 == env._player_info[player].coins
        else:
            assert coins[player] == env._player_info[player].coins

def test_tech_startup(env):
    coins = {}
    n_players = 0

    env._player_info[env._current_player]._add_card("Tech Startup")
    env.invest_in_tech_startup()
    env.invest_in_tech_startup()

    for player in env._player_info.keys():
        n_players += 1
        coins[player] = env._player_info[player].coins
    


    env._activate_card(env._current_player, "Tech Startup")
    for player in env._player_info.keys():
        if player == env._current_player:
            assert coins[player] + (n_players-1)*2 == env._player_info[player].coins
        else:
            assert coins[player] - 2 == env._player_info[player].coins

def test_shopping_mall_box_icon(env):
    for card, info in env._card_info.items():
        if card == "General Store":
            payment = 2
        elif card == "Bakery":
            payment = 1
        elif card == "Flower Shop":
            env._player_info[env._current_player]._add_card("Flower Orchard")
            payment = 1
        else:
            break
        
        coins = env._player_info[env.current_player].coins
        env._activate_card(env._current_player, card)
        assert coins + payment == env._player_info[env.current_player].coins
        env._player_info[env._current_player]._add_card("Shopping Mall")
        coins = env._player_info[env.current_player].coins
        env._activate_card(env._current_player, card)
        assert coins + payment*2 == env._player_info[env.current_player].coins

def test_shopping_mall_cup_icon(env):
    env._current_player = env._player_order[0]
    second_player = env._player_order[1]

    env._player_info[second_player]._add_card("Harbor")

    for card, info in env._card_info.items():
        for player in env._player_info.keys():
            env._player_info[player].coins = 10

        if card == "Café" or card == "Pizza Joint":
            payment = 1
        elif card == "French Restaurant":
            env._player_info[env._current_player]._add_card("Harbor")
            env._player_info[env._current_player]._add_card("Airport")
            payment = 5
        elif card == "Family Restaurant":
            payment = 2
        else:
            break
        
        coins_second_player = env._player_info[second_player].coins
        coins = env._player_info[env._current_player].coins
        env._activate_card(second_player, card)
        assert coins_second_player + payment == env._player_info[second_player].coins
        assert coins - payment == env._player_info[env._current_player].coins

        for player in env._player_info.keys():
            env._player_info[player].coins = 10

        env._player_info[env._current_player]._add_card("Shopping Mall")
        
        coins_second_player = env._player_info[second_player].coins
        coins = env._player_info[env._current_player].coins
        env._activate_card(second_player, card)
        assert coins_second_player + (payment+1) == env._player_info[second_player].coins
        assert coins - (payment+1) == env._player_info[env._current_player].coins

def test_amusement_park(env):
    # so that diceroll will be 1, 1, 1, 3
    random.seed(2)
    env._player_info[env._current_player]._add_card("Amusement Park")
    env._player_info[env._current_player]._add_card("Train Station")

    for _ in range(5):
        env._player_info[env._current_player]._add_card("Flower Orchard")


    coins = env._player_info[env._current_player].coins
    env.dice_roll(2)

    # 1 for the bakery and 5 for each flower orchard
    assert coins + 1 + 5 == env._player_info[env._current_player].coins

def test_airport(env):
    player = env._current_player
    coins = env._player_info[player].coins
    env.step("Build nothing")
    assert coins == env._player_info[player].coins

    player = env._current_player
    env._player_info[player]._add_card("Airport")
    coins = env._player_info[player].coins
    env.step("Build nothing")
    assert coins + 10 == env._player_info[player].coins

def test_earn_income_order(env):
    env._player_order = [f"player {i}" for i in range(1,4)]
    env._current_player = env._player_order[0]
    second_player = env._player_order[1]

    # Test Secondary Industry after Restaurants
    env._player_info[second_player]._add_card("Café")
    env._player_info[env._current_player].coins = 0
    env._player_info[second_player].coins = 3

    env._earn_income(3)

    assert env._player_info[env._current_player].coins == 1 # from bakery
    assert env._player_info[second_player].coins == 3 # still 3 because Café could get money from current player because it had 0 coins
    
    # Test Primary Industry after Restaurants
    env._player_info[second_player]._add_card("Sushi Bar")
    env._player_info[env._current_player].coins = 0
    env._player_info[second_player].coins = 3

    env._earn_income(1)
    assert env._player_info[env._current_player].coins == 1 # from bakery
    assert env._player_info[second_player].coins == 4 # only 1 from Wheat Field (default card) because Sushi Bar could get money from current player because it had 0 coins

    # Test Major Establishments after Secondary Industry
    env._player_info[second_player]._add_card("Mackerel Boat")
    env._player_info[second_player]._add_card("Harbor")
    env._player_info[env._current_player]._add_card("Tax Office")
    env._player_info[second_player].coins = 9
    env._player_info[env._current_player].coins = 0

    env._earn_income(8)

    assert env._player_info[second_player].coins == 6 # got 3 from Mackerel Boat, then lost half through tax office of second player
    assert env._player_info[env._current_player].coins == 6 # got 12 / 2 from second player through tax office

def test_step(env):
    env._current_player = env._player_order[0]

    for i in range(len(env._player_order)):
        env._marketplace._state["1-6"][0] = deque(["Ranch"])
        env.step("Ranch")
        if i == len(env._player_order) - 1:
            assert env._current_player == env._player_order[0]
        else:
            assert env._current_player == env._player_order[i + 1]

def test_marketplace(env):
    n_cards = 0
    for info in env._card_info.values():
        if info["type"] != "Landmarks":
            n_cards += info["n_cards"]

    
    alleys = list(env._marketplace._state.keys())

    for _ in range(2):
        n_gotten_cards = 0
        for _ in range(n_cards):
            for alley in alleys:
                for stand in env._marketplace._state[alley]:
                    if len(stand) > 0:
                        card_to_get = stand[-1]
                        gotten_card = env._marketplace.get(stand[-1])
                        assert gotten_card == card_to_get
                        n_gotten_cards +=1
        
        for alley in alleys:
            for stand in env._marketplace._state[alley]:
                assert len(stand) == 0
        
        assert n_gotten_cards == n_cards
        env.reset()
    
def test_gym_env_obs(gymenv):
    test_obs = []
    gymenv.reset()

    for player in gymenv._env._player_info.keys():
        for card, amount in gymenv._env._player_info[player].cards.items():
            if gymenv._env._card_info[card]["type"] == "Landmarks":
                onehot = np.array([1, 0])
            else:
                max_cards = gymenv._env._card_info[card]["n_cards"]
                onehot = np.zeros(max_cards + 1 + (card == "Wheat Field") + (card == "Bakery"))
                if card == "Wheat Field" or card == "Bakery":
                    onehot[1] = 1
                else:
                    onehot[0] = 1
            test_obs.append(onehot)

    
    for i in range(len(gymenv._env._marketplace._state["1-6"])):
        gymenv._env._marketplace._state["1-6"][i] = deque(["Wheat Field"])
        onehot=np.zeros(len(gymenv._establishments_to_idx["1-6"]) + 1)
        onehot[1] = 1
        test_obs.append(onehot)
        onehot = np.zeros(7) 
        onehot[1] = 1
        test_obs.append(onehot)
    
    for i in range(len(gymenv._env._marketplace._state["7+"])):
        gymenv._env._marketplace._state["7+"][i] = deque(["Mackerel Boat"])
        onehot=np.zeros(len(gymenv._establishments_to_idx["7+"]) + 1)
        onehot[1] = 1
        test_obs.append(onehot)
        onehot = np.zeros(7)
        onehot[1] = 1
        test_obs.append(onehot)
    
    for i in range(len(gymenv._env._marketplace._state["major"])):
        gymenv._env._marketplace._state["major"][i] = deque(["Stadium"])
        onehot=np.zeros(len(gymenv._establishments_to_idx["major"]) + 1)
        onehot[1] = 1
        test_obs.append(onehot)
        onehot = np.zeros(6)
        onehot[1] = 1
        test_obs.append(onehot)
    
    obs, _, _, _, _ = gymenv.step(gymenv._action_str_to_idx["Build nothing"])
    assert np.array_equal(np.hstack(test_obs), obs)

def test_winner_random_policy(gymenv):
    for i in range(1000):
        gymenv._env.dice_roll(1)
        action = gymenv.sample_action()
        obs, reward, done, truncated, info = gymenv.step(action)
        if done:
            break
    assert done
