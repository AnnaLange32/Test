import numpy as np

import pytest

from agents.agent_MCTS.MCTS import Node
def test_get_player_actions():
    from agents.common import string_to_board
    from agents.agent_MCTS.MCTS import get_player_actions

    player_check = 2

    board1 = string_to_board('| - - - - - - - |\n| . . X O . . . |\n| . X O X . . . |\n| . X O O . . . |\n| . O X O O . . |\n| . X O X O . . |\n| . X X O O X . |\n| - - - - - - - |\n| 0 1 2 3 4 5 6 |')
    board2 = string_to_board('| - - - - - - - |\n| X X X X X O . |\n| O X O X O X . |\n| O X X X O O X |\n| X O X O X O O |\n| O O X X O X X |\n| O X O O O X O |\n| - - - - - - - |\n| 0 1 2 3 4 5 6 |')

    board4 = np.zeros((6, 7), dtype=np.int8)
    board4[0, :] = np.array([0, 0, 0, 2, 1, 1, 0])
    board4[1, :] = np.array([0, 0, 0, 2, 1, 1, 0])
    board4[2, :] = np.array([0, 0, 0, 2, 1, 0, 0])
    board4[3, :] = np.array([0, 0, 0, 2, 2, 0, 0])

    ret1 = get_player_actions(board1, player_check)
    ret2 = get_player_actions(board2, player_check)
    ret3 = get_player_actions(board4, player_check, _last_action=3)

    assert isinstance(ret1, list)
    assert ret1 == [0,1,4,5,6]
    assert ret2 == [6]
    assert ret3 == []

def test_check_result():
    from agents.common import string_to_board
    from agents.agent_MCTS.MCTS import check_result

    player_check = 2

    board1 = string_to_board('| - - - - - - - |\n| . . X O . . . |\n| . X O X . . . |\n| . X O O . . . |\n| . O X O O . . |\n| . X O X O . . |\n| . X X O O X . |\n| - - - - - - - |\n| 0 1 2 3 4 5 6 |')
    board2 = string_to_board('| - - - - - - - |\n| X X X X X O . |\n| O X O X O X . |\n| O X X X O O X |\n| X O X O X O O |\n| O O X X O X X |\n| O X O O O X O |\n| - - - - - - - |\n| 0 1 2 3 4 5 6 |')

    board4 = np.zeros((6, 7), dtype=np.int8)
    board4[0, :] = np.array([0, 0, 0, 2, 1, 1, 0])
    board4[1, :] = np.array([0, 0, 0, 2, 1, 1, 0])
    board4[2, :] = np.array([0, 0, 0, 2, 1, 0, 0])
    board4[3, :] = np.array([0, 0, 0, 2, 2, 0, 0])

    board5 = np.zeros((6, 7), dtype=np.int8)
    board5[0, :] = np.array([0, 0, 0, 1, 2, 2, 0])
    board5[1, :] = np.array([0, 0, 0, 1, 2, 2, 0])
    board5[2, :] = np.array([0, 0, 0, 1, 2, 0, 0])
    board5[3, :] = np.array([0, 0, 0, 1, 1, 0, 0])

    ret1 = check_result(board1, player_check)
    ret2 = check_result(board2, player_check)
    ret3 = check_result(board4, player_check)
    ret4 = check_result(board5, player_check)

    assert isinstance(ret1, int)
    assert ret1 == 0
    assert ret2 == 1
    assert ret3 == 1
    assert ret4 == -0.1

def test_node_setup():
    from agents.common import initialize_game_state
    from agents.agent_MCTS.MCTS import Node
    from agents.agent_MCTS.MCTS import get_player_actions

    player_check = 2
    board = initialize_game_state()
    node = Node(board = board, player = player_check)
    actions = get_player_actions(board, player_check)

    assert node.board.all() == board.all()
    assert node.parent is None
    assert node.action is None
    assert node.childNodes == []
    assert node.wins == 0
    assert node.visits == 0
    assert node.player == player_check
    assert node.action_notExp == actions

def test_selection():
    from agents.common import initialize_game_state
    from agents.agent_MCTS.MCTS import Node

    player_check = 2
    board = initialize_game_state()
    node = Node(board=board, player=player_check)
    node.childNodes = [Node(board=board, player=player_check, action = 0), Node(board=board, player=player_check, action = 1)]
    node.childNodes[0].wins = 5
    node.childNodes[0].visits = 10
    node.childNodes[1].wins = 0
    node.childNodes[1].visits = 10
    node.visits = 20
    selected_node = node.selection()

    assert selected_node == node.childNodes[0]


def test_expansion():
    from agents.common import initialize_game_state
    from agents.agent_MCTS.MCTS import Node

    player_check = 2
    board = initialize_game_state()
    node = Node(board=board, player=player_check)
    action = 4
    expanded_node = node.expansion(action=action)

    assert expanded_node.parent == node
    assert node.childNodes == [expanded_node]
    assert expanded_node.action == action
    assert node.player == 3- expanded_node.player

def test_update():
    from agents.common import initialize_game_state
    from agents.agent_MCTS.MCTS import Node

    player_check = 2
    board = initialize_game_state()
    node = Node(board=board, player=player_check)
    updated_node = node.update(result = 1)

    assert node.wins == 1
    assert node.visits == 1

def test_monte_carlo_tree_search():
    from agents.common import initialize_game_state
    from agents.agent_MCTS.MCTS import monte_carlo_tree_search
    from agents.agent_MCTS.MCTS import Node

    player_check = 2
    board = initialize_game_state()
    ret = monte_carlo_tree_search(board=board, player=player_check, saved_state=None)

    assert isinstance(ret[0], int)
