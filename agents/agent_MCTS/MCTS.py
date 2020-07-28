import numpy as np
import time
import random
from typing import Optional
from typing import Tuple


from scipy.signal.sigtools import _convolve2d

from agents.common import PlayerAction, BoardPiece, SavedState, PLAYER1, PLAYER2, NO_PLAYER, CONNECT_N
from agents.common import  apply_player_action, connected_four

col_kernel = np.ones((CONNECT_N, 1), dtype=BoardPiece)
row_kernel = np.ones((1, CONNECT_N), dtype=BoardPiece)
dia_l_kernel = np.diag(np.ones(CONNECT_N, dtype=BoardPiece))
dia_r_kernel = np.array(np.diag(np.ones(CONNECT_N, dtype=BoardPiece))[::-1, :])



def get_player_actions(board: np.ndarray, player: BoardPiece, _last_action: Optional[PlayerAction] = None) -> list: #could move this to common

    '''
    Returns an array with the possible columns that a player could place a piece in.
    Here also returns an empty list when the game is already won.
    An empty list is therefore returned whenever all actions have been explored or
    a terminal state has been reached.
    '''

    if _last_action != None:
        if connected_four(board, player, _last_action):
            return [] #if game is won

    if np.count_nonzero(board) == board.shape[0] * board.shape[1]:
        return [] #if game is draw

    player_actions = []
    for col in range(board.shape[1]):
        if np.count_nonzero(board[:,col]) < board.shape[0]:
            player_actions.append(col)
    return player_actions #if still possible actions

def check_result(
        board: np.ndarray, player: BoardPiece,  _last_action: Optional[PlayerAction] = None) -> bool:

    ''' check if the board is a "terminal" board: a win or a draw
    and assigns a value to each option that can be used by an evaluation function'''

    board1 = board.copy()
    board2 = board.copy()

    MinPiece = 3 - player

    MaxPiece = player

    board1[board1 == MinPiece] = NO_PLAYER
    board1[board1 == MaxPiece] = BoardPiece(1)

    for kernel in (col_kernel, row_kernel, dia_l_kernel, dia_r_kernel):
        result = _convolve2d(board1, kernel, 1, 0, 0, BoardPiece(0))
        if np.any(result == CONNECT_N):
            # print(time)
            # print(MaxPiece)
            # print("won")
            return 1 #self wins

    board2[board2 == MaxPiece] = NO_PLAYER
    board2[board2 == MinPiece] = BoardPiece(1)

    for kernel in (col_kernel, row_kernel, dia_l_kernel, dia_r_kernel):
        result = _convolve2d(board2, kernel, 1, 0, 0, BoardPiece(0))
        if np.any(result == CONNECT_N):
            # print(time)
            # print(MinPiece)
            # print("lost")
            return -0.1 #opponent wins

    if np.count_nonzero(board) == board.shape[0] * board.shape[1]:
        # print(board)
        # print("draw")
        return 0 #draw

    return False


class Node:
    '''
    generates one node in the game tree and if applicable has access to info about parent and child nodes
    board: the state of the board at the current node
    parent: one node up in the game tree from the current node
    action: the action that led from the parent node to the current node
    wins: the number of wins that followed visiting this node
    visits: the number of times this node was visited
    '''

    def __init__(self, player: BoardPiece=None, action: Optional[PlayerAction]=None, parent=None, board: np.ndarray=None):
        self.board = board.copy()
        self.parent = parent
        self.action = action
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.player = player
        self.action_notExp = get_player_actions(self.board, self.player, self.action)

    def selection(self):
        '''
        Selection step of MCTS:
        Traverses current tree from root node and selects the node with the highest estimated value.
        :return: a new node (class), which is the best out of the available children
        '''

        ucb = lambda child: child.wins / child.visits + np.sqrt(np.log(self.visits) / child.visits)
        return sorted(self.childNodes, key=ucb)[-1]  # child with largest UCB value

    def expansion(self, action: np.int8):
        '''
        Expansion step of MCTS:
        Once a node has been selected it finds the leaf node and adds a further child node unless it is a terminal node.
        :param action: action to be applied to board
        :param board: current palying board
        :return: a new child node
        '''

        # return child when action is taken
        # remove action from current node

        board = apply_player_action(self.board.copy(), action, 3- self.player)
        child = Node(action=action, player=3-self.player, parent=self, board=board)
        self.action_notExp.remove(action)
        self.childNodes.append(child)
        return child

    def update(self, result:float):

        """
        updates the number of state visits and wins that are associated to choosing this state
        :param self: class of current node
        :param result: 1 if win, -0.5 if draw, 0 if lost
        :return: updates elements wins and visits in class, visits consitently increase by 1
        """

        self.wins += result
        self.visits += 1


# main function for the Monte Carlo Tree Search
def monte_carlo_tree_search(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState], timeout: np.int8 =10
) -> Tuple[PlayerAction, Optional[SavedState]]:

    '''
    4 step tree search algorithm:
     1. Selection
     2. Expansion
     3. Simulation
     4. Bakpropagation

    :param board:
    :param player:
    :param saved_state:
    :param timeout:
    :return:
    '''

    MinPiece = 3 - player
    MaxPiece = player

    root= Node(board=board, player=MinPiece)

    #check immediate win

    for action in root.action_notExp:
        state = board.copy()
        apply_player_action(state, action, MaxPiece)
        if connected_four(state, MaxPiece, action) == True:
            return action,  saved_state

    start = time.clock()
    while True:

        node = root
        state = board.copy()

        # selection
        # keep going down the tree based on best UCT values until terminal (no more children) or unexpanded node (no more moves to expand)
        while node.action_notExp == [] and node.childNodes != []:
            node = node.selection()
            apply_player_action(state, node.action, MaxPiece)


        # expansion
        if node.action_notExp != []:
            action = random.choice(node.action_notExp)
            node = node.expansion(action)

        # simulation
        state = node.board.copy()

        player_roll = node.player

        result = 0

        while get_player_actions(state, 3- player_roll, action) and result != 1 and result != -0.1: #check here if win or loss already occured

            player_roll  = 3 - player_roll
            action = random.choice(get_player_actions(state, player_roll, action))

            apply_player_action(state,action, player_roll)

            result = check_result(state, MaxPiece, action) #check if the agent won or lost i.e. the player looking for max wins

        # backpropagation
        while node is not None:
            node.update(result)
            node = node.parent

        duration = time.clock() - start
        if duration > timeout: break

    choose_fnct = lambda child: child.wins / child.visits
    chosen_child = sorted(root.childNodes, key=choose_fnct)[::-1]  #change order from highest to largest

    return chosen_child[0].action, saved_state #choose largest element
