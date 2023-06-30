import pytest
from .env import MachiKoro

@pytest.fixture
def env():
    return MachiKoro(n_players=4)

def test_env_init(env):
    for player, info in env._player_info.items():
        assert info.cards == {
            'Wheat Field': 1, 'Ranch': 0, 'Flower Orchard': 0, 'Forest': 0, 'Mackerel Boat': 0, 
            'Apple Orchard': 0, 'Tuna Boat': 0, 'General Store': 0, 'Bakery': 1, 
            'Demolition Company': 0, 'Flower Shop': 0, 'Cheese Factory': 0, 'Furniture Factory': 0,
            'Soda Bottling Plant': 0, 'Fruit and Vegetable Market': 0, 
            'Sushi Bar': 0, 'Café': 0, 'French Restaurant': 0, 'Pizza Joint': 0, 
            'Family Restaurant': 0, "Member's Only Club": 0, 'Stadium': 0, 'Publisher': 0, 
            'Tax Office': 0, 'Tech Startup': 0, 'City Hall': False, 
            'Harbor': False, 'Train Station': False, 'Shopping Mall': False, 
            'Amusement Park': False, 'Moon Tower': False, 'Airport': False
        }
        assert info.coins == 3
        assert info.landmarks == 1

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
        env._activate_card(player, "Demolition Company", "Harbor")
        assert coins + 8 == env._player_info[player].coins

        assert env._player_info[player].cards["Harbor"] == False

        env._player_info[player]._add_card("Harbor")
        env._player_info[player]._add_card("Train Station")
        coins = env._player_info[player].coins
        env._activate_card(player, "Demolition Company", info="Harbor")
        assert coins + 8 == env._player_info[player].coins

        assert env._player_info[player].cards["Harbor"] == False
        assert env._player_info[player].cards["Train Station"] == True

        coins = env._player_info[player].coins
        env._activate_card(player, "Demolition Company", info="Train Station")
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
    env._player_info[rich_player].coins == 23

    coins = {}
    for player in env._player_info.keys():
        coins[player] = env._player_info[player].coins

    env._activate_card(env._current_player, "Tax Office")
    for player in env._player_info.keys():
        if player == env._current_player:
            coins[player] + 11 == env._player_info[player].coins
        elif player == rich_player:
            coins[player] - 11 == env._player_info[player].coins
        else:
            coins[player] == env._player_info[player].coins


