from random import randrange, uniform

import numpy as np
from typing import Optional
from typing import Tuple

from scipy.signal.sigtools import _convolve2d

from agents.common import PlayerAction, BoardPiece, SavedState, PLAYER1, PLAYER2, NO_PLAYER, CONNECT_N
from agents.common import initialize_game_state, pretty_print_board, apply_player_action, check_end_state



def get_player_actions(board: np.ndarray) -> list:

    '''
    returns an array with the possible columns that a player could place a piece in
    '''

    player_actions = []
    for col in range(board.shape[1]):
        if np.count_nonzero(board[:,col]) < board.shape[0]:
            player_actions.append(col)
    return player_actions


def position_value(
        board: np.ndarray, player: BoardPiece, _last_action: Optional[PlayerAction] = None
) -> bool:
    """
    Returns the heuristic value to the given plaer of a complete board
    """

    board1 = board.copy()
    board2 = board.copy()

    other_player = BoardPiece(player % 2 + 1)
    board1[board1 == other_player] = 5
    board1[board1 == player] = BoardPiece(1)

    board2[board2 == player] = BoardPiece(5)
    board2[board2 == other_player] = BoardPiece(1)

    value = 0

    # scoring central positions
    center = board[:, board.shape[1] // 2]
    value += (center == player).sum() * 10
    value += (center == other_player).sum() * -5

    # checking remainin positions
    for kernel in (col_kernel, row_kernel, dia_l_kernel, dia_r_kernel):
        result = _convolve2d(board1, kernel, 1, 0, 0, BoardPiece(0))
        for i in result:
            for sum in i:
                if sum == CONNECT_N:
                    value += 200

                if sum == CONNECT_N - 1:
                    value += 50

                if sum == CONNECT_N - 2:
                    value += 10

    for kernel in (col_kernel, row_kernel, dia_l_kernel, dia_r_kernel):
        result = _convolve2d(board2, kernel, 1, 0, 0, 0)
        for i in result:
            for sum in i:
                if sum == CONNECT_N:
                    value += -250

                if sum == CONNECT_N - 1:
                    value += -55

                if sum == CONNECT_N - 2:
                    value += -12

    return int(value)


def check_terminal(
        board: np.ndarray, _last_action: Optional[PlayerAction] = None
) -> bool:
    ''' check if the board is a "terminal" board: a win or a draw'''

    board1 = board.copy()
    board2 = board.copy()

    board1[board1 == PLAYER1] = NO_PLAYER
    board1[board1 == PLAYER2] = BoardPiece(1)

    for kernel in (col_kernel, row_kernel, dia_l_kernel, dia_r_kernel):
        result = _convolve2d(board1, kernel, 1, 0, 0, BoardPiece(0))
        if np.any(result == CONNECT_N):
            return True

    board2[board2 == PLAYER2] = NO_PLAYER
    board2[board2 == PLAYER1] = BoardPiece(1)

    for kernel in (col_kernel, row_kernel, dia_l_kernel, dia_r_kernel):
        result = _convolve2d(board2, kernel, 1, 0, 0, BoardPiece(0))
        if np.any(result == CONNECT_N):
            return True

    if np.count_nonzero(board) == board.shape[0] * board.shape[1]:
        return True

    return False

# def minimax(board: np.ndarray, MaximisingPlayer: bool,  player: BoardPiece, depth = 3):
#
#     '''
#     depth limited minimax: returns the value of the given board based on the minimax algorithm
#     '''
#
#     MinPiece = 3 - player.copy()
#     MaxPiece = player.copy()
#
#     terminalboard = check_terminal(board)
#
#     if depth == 0 or terminalboard == True:
#         return position_value(board,player)
#
#     if MaximisingPlayer:
#         value = -999
#         valid_actions = get_player_actions(board)
#         for action in valid_actions:
#             child_board = board.copy()
#             child_board = apply_player_action(child_board, action, MaxPiece)
#             value = max( value, minimax(child_board, False, player, depth -1))
#         return value
#     else:
#         value = 999
#         valid_actions = get_player_actions(board)
#         for action in valid_actions:
#             child_board = board.copy()
#             child_board = apply_player_action(child_board, action, MinPiece)
#             value = min(value, minimax(child_board, True, player, depth - 1))
#         return value

def alphabeta(board: np.ndarray, alpha: np.int8, beta:np.int8, MaximisingPlayer: bool, player: BoardPiece, depth = 4):

    '''
    depth limited minimax with alpha-beta pruning:
    returns the value of the given board based on the minimax algorithm with alpha beta pruning
    '''

    MinPiece = 3 - player.copy()
    MaxPiece = player.copy()

    terminalboard = check_terminal(board)

    if depth == 0 or terminalboard == True:
        return position_value(board,player) * (depth +1)

    if MaximisingPlayer:
        value = -999
        valid_actions = get_player_actions(board)
        for action in valid_actions:
            child_board = board.copy()
            child_board = apply_player_action(child_board, action, MaxPiece)
            value = max(value, alphabeta(child_board, alpha, beta, False, player, depth - 1))
            alpha = max(alpha, value)
            if alpha >= beta:
                break #β cut-off
        return value
    else:
        value = 999
        valid_actions = get_player_actions(board)
        for action in valid_actions:
            child_board = board.copy()
            child_board = apply_player_action(child_board, action, MinPiece)
            value = min(value, alphabeta(child_board, alpha, beta, True, player, depth -1))
            beta = min(beta, value)
            if beta <= alpha:
                break #α cut-off
        return value



def generate_smart_move(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:

    '''
    chooses a column (i.e. "action") that corresponds to the smallest value the other players can force the player to receive
    '''

    best_value = -999

    valid_actions = get_player_actions(board) #get the possible moves
    best_action = 0


    for action in valid_actions:
        alpha = -999
        beta = 999

        temp_board = board.copy()
        apply_player_action(temp_board, action, player)
        #value = position_value(temp_board, player)
        #value = minimax(temp_board, False, player)
        value = alphabeta(temp_board, alpha, beta, False, player)
        if value > best_value:
            best_value = value
            best_action = action
    print(type(best_action))

    return best_action, saved_state








# old, slower functions below

# def window_value(window: np.ndarray, player: BoardPiece) -> int:
#     """
#     Returns the heuristic value of "CONNECT_N" sequential pieces of the board (here a "window") to the player,
#     in order to asses the current state of a board
#     and provide a value that indicates how beneficial the state is to the given player.
#     """
#     MaxPiece = player #as this function only gets called by the AI player (i.e. the "maximising player")
#     MinPiece = 3 - player #define the minimising player, for which we get negative scores
#     value = 0
#     if (window == MaxPiece).sum() == CONNECT_N:
#         value += 200
#
#     elif (window == MaxPiece).sum() == CONNECT_N-1 and (window == NO_PLAYER).sum() == CONNECT_N-(CONNECT_N-1):
#         value += 50
#
#     elif (window == MaxPiece).sum() == CONNECT_N-2 and (window == NO_PLAYER).sum() == CONNECT_N-(CONNECT_N-2):
#         value += 10
#
#     elif (window == MinPiece).sum() == CONNECT_N:
#         value += -250
#
#     elif (window == MinPiece).sum() == CONNECT_N-1 and (window == NO_PLAYER).sum() == CONNECT_N-(CONNECT_N-1):
#         value += -55
#
#     elif (window == MinPiece).sum() == CONNECT_N-2 and (window == NO_PLAYER).sum() == CONNECT_N-(CONNECT_N-2):
#         value += -12
#
#
#     return int(value)

#def position_value(board: np.ndarray, player: BoardPiece) -> int:
#     """
#     Returns the heuristic value to the given plaer of a complete board
#     """
#     rows, cols = board.shape
#     rows_edge = rows - CONNECT_N + 1
#     cols_edge = cols - CONNECT_N + 1
#     MinPiece = 3 - player
#     value = 0
#     #scoring central positions
#     center = board[:, board.shape[1]//2]
#     value += (center == player).sum()*10
#     value += (center == MinPiece).sum() * -5
#
#     #scoring rows
#     for i in range(rows):
#         for j in range(cols_edge):
#             window = board[i, j:j + CONNECT_N]
#             value += window_value(window,player)
#     #scoring rows
#     for i in range(rows_edge):
#         for j in range(cols):
#             window = board[i:i + CONNECT_N, j]
#             value += window_value(window,player)
#     #scoring diagonals
#     for i in range(rows_edge):
#         for j in range(cols_edge):
#             block = board[i:i + CONNECT_N, j:j + CONNECT_N]
#             window = np.diag(block)
#             value += window_value(window,player)
#             window = np.diag(block[::-1, :])
#             value += window_value(window, player)
#
#     return int(value)

# def check_terminal(board: np.ndarray) -> bool:
#
#     ''' check if the board is a "terminal" board: a win or a draw'''
#
#     rows, cols = board.shape
#     rows_edge = rows - CONNECT_N + 1
#     cols_edge = cols - CONNECT_N + 1
#
#     for i in range(rows):
#         for j in range(cols_edge):
#             if np.all(board[i, j:j + CONNECT_N] == PLAYER1) or np.all(board[i, j:j + CONNECT_N] == PLAYER2):
#                 return True
#
#     for i in range(rows_edge):
#         for j in range(cols):
#             if np.all(board[i:i + CONNECT_N, j] == PLAYER1) or  np.all(board[i:i + CONNECT_N, j] == PLAYER2) :
#                 return True
#
#     for i in range(rows_edge):
#         for j in range(cols_edge):
#             block = board[i:i + CONNECT_N, j:j + CONNECT_N]
#             if np.all(np.diag(block) == PLAYER1) or np.all(np.diag(block) == PLAYER2) :
#                 return True
#             if np.all(np.diag(block[::-1, :]) == PLAYER1) or np.all(np.diag(block[::-1, :]) == PLAYER2):
#                 return True
#
#     if np.count_nonzero(board) == board.shape[0] * board.shape[1]:
#         return True
#
#     return False




