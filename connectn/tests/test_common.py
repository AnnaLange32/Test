import numpy as np
import pytest

def test_initialize_game_state():
    from connectn.common import initialize_game_state

    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == np.int8
    assert ret.shape == (6,7)
    assert np.all(ret ==0)

def test_pretty_print_board():
    from connectn.common import pretty_print_board
    from connectn.common import board

    ret = pretty_print_board(board)
    ret2 = pretty_print_board(np.zeros((6,7)))

    assert isinstance(ret, str)
    assert ret2 == '. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .'
    with pytest.raises(ValueError):
        pretty_print_board('hello')
    



@pytest.mark.skip(reason="do not run apply player test")
def test_apply_player_action():
    from connectn.common import apply_player_action

    ret3 = apply_player_action()


    assert isinstance(ret3, str)
    assert ret3.dtype == str
    assert ret3.shape == (6,7)

@pytest.mark.skip(reason="do not run string to board test")
def test_string_to_board():
    from connectn.common import string_to_board

    ret = string_to_board()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == int8
    assert ret.shape == (6,7)

def text_connected_four():
    from connectn.common import connected_four

    ret5 = connected_four()

    assert isinstance(ret5, str)
    assert ret5.dtype == str
    assert ret5.shape == (6,7)