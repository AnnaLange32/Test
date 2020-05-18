#
import numpy as np
from typing import Optional
PlayerAction = np.int8
BoardPiece = np.int8

def initialize_game_state() -> np.ndarray:
    return np.zeros((6,7), dtype=BoardPiece)

board = initialize_game_state()
board[0,0] = 2 # lower left corner on board

def pretty_print_board(board: np.ndarray) -> str:
    #Flip the board, so that [0,0] is lower left corner, then print the board in a pretty way
    if not isinstance(board, np.ndarray):
        raise ValueError("The input must be an array") #raises an error in case non array was given as input
    print("| - - - - - - - |") #prints top line
    pp_board = "" #create variable to return
    for line in reversed(board):
        printed_board = "| " #create variable to be printed

        for element in line:
            if element == 1:
                printed_board += "O " # player 1 has O's as playing stones
            elif element == 2:
                printed_board += "X " # player 2 has X's as playing stones
            else:
                printed_board += ". " #empty positions are indicated with a dot
        printed_board += " |"
        stripped_pboard = printed_board[2:len(printed_board)-1:1] #strips the printed variable of unnecessary "pretty" elements
        stripped_pboard = stripped_pboard[0:len(stripped_pboard) - 1]
        pp_board += stripped_pboard
        print(printed_board) #prints board elements "pretty"
    print("| - - - - - - - |") #add final useful printed elements
    print("| 0 1 2 3 4 5 6 |")
    pp_board = pp_board[0:len(pp_board) - 1]
    return pp_board



def apply_player_action(board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False) -> np.ndarray:
    if copy == True:
        board= board.copy()
    for counter,row in enumerate(board[:,action]):
        if row == 0:
            i = counter
            break
    board[i,action] = player
    return board


def string_to_board(pp_board):
    board = np.zeros(6*7)
    pp_board = pp_board.split(" ")
    print(pp_board)
    for counter, item in enumerate(pp_board):
        if item == ".":
            board[counter] = 0
        elif item == "O":
            board[counter] = 1
        elif item == "X":
            board[counter] = 2
    print(board)
    board = np.reshape(board, (6,7),order = "F")
    print(board)
    board = np.flipud(board)
    print(board)
    return board


def connected_four(board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None)-> bool:

    for counter, row in enumerate(board):
        if np.count_nonzero(row) == 0:
            row_index = counter - 1
            break
    print(row_index)
    check_position = np.zeros(2)
    check_position[0] = row_index
    check_position[1] = last_action

    win_sum = 0

    for i in board[int(check_position[0]),:]:
        if i == player:
            win_sum += 1
        else:
            win_sum = 0
        if win_sum >= 4:
            return True

    win_sum = 0
    for i in board[:, int(check_position[1])]:
        if i == player:
            win_sum += 1
        else:
            win_sum = 0
        if win_sum >= 4:
            return True
    win_sum = 0
    for i in np.diag(board, k = (int(check_position[1])-int(check_position[0]))):
        if i == player:
            win_sum += 1
        else:
            win_sum = 0
        if win_sum >= 4:
            return True
    win_sum = 0
    for i in np.diag(np.fliplr(board), k = (board.shape[1]-1-int(check_position[1]) - int(check_position[0]))):
        if i == player:
            win_sum += 1
        else:
            win_sum = 0
        if win_sum >= 4:
            return True

    return False






