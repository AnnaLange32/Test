import numpy as np
from agents.common import BoardPiece, NO_PLAYER, PlayerAction

def test_initialize_game_state():
    from agents.common import initialize_game_state

    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == np.int8
    assert ret.shape == (6,7)
    assert np.all(ret ==NO_PLAYER)

def test_pretty_print_board():
    from agents.common import pretty_print_board
    from agents.common import board

    ret = pretty_print_board(board)
    ret2 = pretty_print_board(np.zeros((6,7)))

    assert isinstance(ret, str)
    assert ret2 == '. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .'
    with pytest.raises(ValueError):
        pretty_print_board('hello')
    



def test_apply_player_action():
    from agents.common import apply_player_action


    board1 = np.zeros((6, 7), dtype=np.int8)
    board1[0, :] = np.array([0, 1, 2, 1, 2, 1, 0])
    board1[1, :] = np.array([0, 0, 1, 1, 2, 1, 0])
    board1[2, :] = np.array([0, 0, 0, 2, 0, 2, 0])
    board2 = np.ones((6,7))
    player_action1 = np.array([4])
    player_check = 2
    check_board = board_old = np.copy(board1)
    check_board[2,4] = 2

    ret= apply_player_action(board1, player_action1, player_check)
    ret2 = apply_player_action(board2, player_action1, player_check)


    assert isinstance(ret, np.ndarray)
    assert ret.dtype == np.int8
    assert ret.shape == (6, 7)
    assert (ret == check_board).all()
    assert ret2 == IndexError


def test_string_to_board():
    from agents.common import string_to_board
    from agents.common import pretty_print_board

    board1 = np.zeros((6, 7),dtype=np.int8)
    board1[0, :] = np.array([0, 1, 1, 1, 1, 1, 0])
    board1[1, :] = np.array([0, 1, 1, 1, 1, 1, 0])
    board1[2, :] = np.array([0, 1, 1, 1, 1, 1, 0])
    board1[3, :] = np.array([0, 2, 2, 2, 2, 0, 0])

    board2 = np.zeros((6, 7), dtype=np.int8)
    board2[:, 4] = np.array([2, 2, 2, 2, 0, 0])
    ret = pretty_print_board(board1)
    ret1 = string_to_board(ret)
    ret2 = pretty_print_board(board2)
    ret3 = string_to_board(ret2)

    assert isinstance(ret1, np.ndarray)
    assert ret1.dtype == np.int8
    assert (ret == ret1).all()
    assert (ret2 == ret4).all()

def test_connected_four():
    from agents.common import connected_four

    player_check = 2
    player_wrong = 1
    board1 = np.zeros((6,7), dtype=np.int8)
    board1[0,:] = np.array([0, 1, 1, 1, 1, 1, 0])
    board1[1,:] = np.array([0, 1, 1, 1, 1, 1, 0])
    board1[2, :] = np.array([0, 1, 1, 1, 1, 1, 0])
    board1[3,:] = np.array([0, 2, 2, 2, 2, 0, 0])
    player_action1 = np.array([4])

    board2 = np.zeros((6,7), dtype=np.int8)
    board2[:,4] = np.array([2,2,2,2,0,0])

    board3 = np.array([[0, 0, 2, 1, 1, 1], [0, 0, 0, 2, 2,1], [0, 0, 0,0, 2, 1], [0,0,0,0,0, 2], [0,0,0,0,0, 0]])
    player_action3 = np.array([5])

    board4 = np.array([[0, 2, 2, 1, 2, 1], [0, 1, 1, 2, 2,1], [0, 1, 2,0, 0, 0], [0,2,0,0,0, 0], [0,0,0,0,0, 0]])
    player_action4 = np.array([1])

    ret = connected_four(board1, player_check, player_action1)
    ret1 = connected_four(board2, player_check, player_action1)
    ret2 = connected_four(board3, player_check, player_action3)
    ret3 = connected_four(board4, player_check, player_action4)
    ret4 = connected_four(board4, player_wrong, player_action4)
    ret5 = connected_four(board4, player_check, player_action1)

    assert isinstance(ret, bool)
    assert ret == True
    assert ret1 == True
    assert ret2 == True
    assert ret3 == True
    assert ret4 == False
    assert ret5 == False



