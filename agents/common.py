
import numpy as np
from enum import Enum
from typing import Optional
from typing import Callable, Tuple

class SavedState:
    pass

PlayerAction = np.int8   # The column to be played
BoardPiece = np.int8    # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 has a piece "0"
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 has a piece "X"
CONNECT_N = BoardPiece(4) #number of joined pieces that wins the game


class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


def initialize_game_state() -> np.ndarray:
    '''creates the empty playing bord'''
    return np.zeros((6,7), dtype=BoardPiece)

board = initialize_game_state()
board[0,0] = 2 # lower left corner on board


def pretty_print_board(board: np.ndarray) -> str:
    '''creates a printable string from the playing board'''
    # Flip the board, so that [0,0] is lower left corner, then print the board in a pretty way
    if not isinstance(board, np.ndarray):
        raise ValueError("The input must be an array")  # raises an error in case non array was given as input
    pp_board = "| - - - - - - - |\n"  # top line
    for line in reversed(board):
        pp_board += "| "  # create variable to be printed

        for element in line:
            if element == 1:
                pp_board += "O "  # player 1 has O's as playing stones
            elif element == 2:
                pp_board += "X "  # player 2 has X's as playing stones
            else:
                pp_board += ". "  # empty positions are indicated with a dot
        pp_board += "|\n"  # end of one line
    pp_board += "| - - - - - - - |\n| 0 1 2 3 4 5 6 |"  # add final useful elements
    return pp_board


def apply_player_action(board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False) -> np.ndarray:
    '''applies the player action to the playing board'''
    if copy == True:
       board= board.copy()
    try:
        row_index = np.argwhere(board[:,action] == 0)[0]  #searches for the row idx of the first empty position in the chosen column
        board[row_index,action] = player #places a "BoardPiece" in the chosen position
    except IndexError:
        return IndexError #output when no empty row exists, this will lead to game loss

    return board


def string_to_board(pp_board):
    '''converts the printable string back to an iterable array'''
    board = np.zeros((6, 7)) #empty shell for reconstrcuted board
    pp_board = pp_board[18:] #remove first line of pp board
    pp_board = pp_board.split(" ") #turning the PrettyPrintBoard into an iterable
    row_idx = 0 #initialise row index
    column_idx = 0 #initialise column index
    for item in pp_board:
        if column_idx > 6: #reset meachanism for column index once one row has been filled
            column_idx = 0
        if item == ".": #replacing the print items by the array items
            board[row_idx, column_idx] = 0
            column_idx += 1
        elif item == "O":
            board[row_idx, column_idx] = 1
            column_idx += 1
        elif item == "X":
            board[row_idx, column_idx] = 2
            column_idx += 1
        elif item == "|\n|":
            row_idx += 1

    board = np.flipud(board)
    return board.astype(np.int8)


def connected_four(board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None)-> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    If desired, the last action taken (i.e. last column played) can be provided
    for potential speed optimisation.
    """
    #get the row index
    for counter, element in enumerate(board[:, last_action]):
        if element == 0:
            row_index = counter - 1
            break


    if np.count_nonzero(board[:, last_action]) == board.shape[0]:
        row_index = board.shape[0] - 1

    check_position = np.zeros(2)
    check_position[0] = row_index
    check_position[1] = last_action

    win_sum = 0


    #check for a win in the row of the last action

    for i in board[int(check_position[0]),:]:
        if i == player:
            win_sum += 1
        else:
            win_sum = 0
        if win_sum >= 4:
            return True
    #check for a win in the column of the last action
    win_sum = 0
    for i in board[:, int(check_position[1])]:
        if i == player:
            win_sum += 1
        else:
            win_sum = 0
        if win_sum >= 4:
            return True
    #check for a win in the normal diagonal the last action falls on
    win_sum = 0
    for i in np.diag(board, k = (int(check_position[1])-int(check_position[0]))):
        if i == player:
            win_sum += 1
        else:
            win_sum = 0
        if win_sum >= 4:
            return True
    # check for a win in the inverse diagonal the last action falls on
    win_sum = 0
    for i in np.diag(np.fliplr(board), k = (board.shape[1]-1-int(check_position[1]) - int(check_position[0]))):
        if i == player:
            win_sum += 1
        else:
            win_sum = 0
        if win_sum >= 4:
            return True

    # check if the board is full and the game is a DRAW


    return False



def check_end_state(
    board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """
    if connected_four(board, player, last_action) == True:
        return GameState.IS_WIN
    if np.count_nonzero(board) == board.shape[0] * board.shape[1]:
        return GameState.IS_DRAW
    else:
        return GameState.STILL_PLAYING




GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]



