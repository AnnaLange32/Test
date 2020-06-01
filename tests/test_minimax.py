import numpy as np

import pytest

from agents.common import PlayerAction, BoardPiece, NO_PLAYER, PLAYER1, PLAYER2, CONNECT_N

# def test_window_value():
#     from agents.agent_minimax.minimax import window_value
#
#     window1 = np.array([0,0,0,0])
#     window2 = np.array([2,2,2,0])
#     window3 = np.array([1,1,1,0])
#     window4 = np.array([1,1,1,1])
#     window5 = np.array([1,0,1,0])
#
#     ret11 = window_value(window1, PLAYER1)
#     ret12 = window_value(window1, PLAYER2)
#
#     ret21 = window_value(window2, PLAYER1)
#     ret22 = window_value(window2, PLAYER2)
#
#     ret31 = window_value(window3, PLAYER1)
#     ret32 = window_value(window3, PLAYER2)
#
#     ret41 = window_value(window4, PLAYER1)
#     ret42 = window_value(window4, PLAYER2)
#
#     ret51 = window_value(window5, PLAYER1)
#     ret52 = window_value(window5, PLAYER2)
#
#
#     assert isinstance(ret11, int)
#     assert ret11 == 0 and ret12 == 0
#     assert ret21 == -55 and ret22 == 50
#     assert ret31 == 50 and ret32 == -55
#     assert ret41 == 200 and ret42 == -250
#     assert ret51 == 10 and ret52 == -12


def test_position_value():
    #from agents.agent_minimax.minimax import window_value
    from agents.agent_minimax.minimax import position_value
    from agents.common import string_to_board

    board1 = string_to_board("| - - - - - - - |\n| . . . . . . . |\n| . . . . . . . |\n| . . . . . . . |\n| . . . O . . . |\n| . . O X . . . |\n| . . X O . . . |\n| - - - - - - - |\n|0 1 2 3 4 5 6 |")
    board2 = string_to_board("| - - - - - - - |\n| . . . . . . . |\n| . . . X . . . |\n| . . O O . . . |\n| . . X O . . . |\n| . . O X . . . |\n| . . X O . . . |\n| - - - - - - - |\n| 0 1 2 3 4 5 6 |")
    board3 = string_to_board('| - - - - - - - |\n| . . X O . . . |\n| . X O X . . . |\n| . X O O . . . |\n| . O X O O . . |\n| . X O X O . . |\n| . X X O O X . |\n| - - - - - - - |\n| 0 1 2 3 4 5 6 |')
    board4 = string_to_board("| - - - - - - - |\n| . . . . . . . |\n| . . . . . . . |\n| . . . . . . . |\n| . . . . . . . |\n| . . . . . . . |\n| . . . O . . . |\n| - - - - - - - |\n| 0 1 2 3 4 5 6 |")

    ret11 = position_value(board1, PLAYER1)
    ret12 = position_value(board1, PLAYER2)

    ret21 = position_value(board2, PLAYER1)
    ret22 = position_value(board2, PLAYER2)

    ret31 = position_value(board3, PLAYER1)
    ret32 = position_value(board3, PLAYER2)

    ret41 = position_value(board4, PLAYER1)
    ret42 = position_value(board4, PLAYER2)

    assert isinstance(ret11, int)
    assert ret11 == 33 and ret12 == -26
    assert ret21 == 74 and ret22 == -73
    assert ret31 == 223 and ret32 == -232
    assert ret41 == 10 and ret42 == -5


def test_get_player_actions():
    from agents.common import string_to_board
    from agents.agent_minimax.minimax import get_player_actions

    board1 = string_to_board('| - - - - - - - |\n| . . X O . . . |\n| . X O X . . . |\n| . X O O . . . |\n| . O X O O . . |\n| . X O X O . . |\n| . X X O O X . |\n| - - - - - - - |\n| 0 1 2 3 4 5 6 |')
    board2 = string_to_board('| - - - - - - - |\n| X X X X X O . |\n| O X O X O X . |\n| O X X X O O X |\n| X O X O X O O |\n| O O X X O X X |\n| O X O O O X O |\n| - - - - - - - |\n| 0 1 2 3 4 5 6 |')

    ret1 = get_player_actions(board1)
    ret2 = get_player_actions(board2)

    assert isinstance(ret1, list)
    assert ret1 == [0,1,4,5,6]
    assert ret2 == [6]


def test_check_terminal():
    from agents.agent_minimax.minimax import check_terminal
    from agents.common import string_to_board

    board1 = string_to_board('| - - - - - - - |\n| X X X X X O 0 |\n| O X O X O X 0 |\n| O X X X O O X |\n| X O X O X O O |\n| O O X X O X X |\n| O X O O O X O |\n| - - - - - - - |\n| 0 1 2 3 4 5 6 |')
    board2 = string_to_board('| - - - - - - - |\n| . . X O . . . |\n| . X O X . . . |\n| . X O O . . . |\n| . O X O O . . |\n| . X O X O . . |\n| . X X O O X . |\n| - - - - - - - |\n| 0 1 2 3 4 5 6 |')
    board3 = string_to_board("| - - - - - - - |\n| . . . . . . . |\n| . . . . . . . |\n| . . . X . . . |\n| . O O O O . . |\n| . X X O O X . |\n| X X O O O X X |\n| - - - - - - - |\n| 0 1 2 3 4 5 6 |")

    ret1 = check_terminal(board1)
    ret2 = check_terminal(board2)
    ret3 = check_terminal(board3)

    assert isinstance(ret1, bool)
    assert ret1 == True
    assert ret2 == False
    assert ret3 == True

# def test_minimax():
#     from agents.agent_minimax.minimax import minimax
#     from agents.common import string_to_board
#
#     board1 = string_to_board("| - - - - - - - |\n| . . . . . . . |\n| . . . . . . . |\n| . . . . . . . |\n| . . . . . . . |\n| . . . . . . . |\n| . . . O . . . |\n| - - - - - - - |\n| 0 1 2 3 4 5 6 |")
#     board2 = string_to_board("| - - - - - - - |\n| . . . . . . . |\n| . . . . . . . |\n| . . . . . . . |\n| . . . . . . . |\n| . . . . . . . |\n| . . O . . . . |\n| - - - - - - - |\n| 0 1 2 3 4 5 6 |")
#
#     ret1 = minimax(board1,False,PLAYER1, depth=5)
#     ret2 = minimax(board2,False,PLAYER1, depth=5)
#
#     assert (ret1 > ret2)  == True


def test_alphabeta():
    from agents.agent_minimax.minimax import alphabeta
    from agents.common import string_to_board

    board1 = string_to_board("| - - - - - - - |\n| . . . . . . . |\n| . . . . . . . |\n| . . . . . . . |\n| . . . . . . . |\n| . . . . . . . |\n| . . . O . . . |\n| - - - - - - - |\n| 0 1 2 3 4 5 6 |")
    board2 = string_to_board("| - - - - - - - |\n| . . . . . . . |\n| . . . . . . . |\n| . . . . . . . |\n| . . . . . . . |\n| . . . . . . . |\n| . . O . . . . |\n| - - - - - - - |\n| 0 1 2 3 4 5 6 |")
    alpha = -999
    beta = 999

    ret1 = alphabeta(board1,alpha, beta, False,PLAYER1, depth=5)
    ret2 = alphabeta(board2, alpha, beta, False,PLAYER1, depth=5)

    assert (ret1 > ret2)  == True