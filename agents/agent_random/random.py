from random import randrange

import numpy as np
from typing import Optional
from typing import Tuple

from agents.common import PlayerAction, BoardPiece, SavedState

def generate_move_random(
    board: np.ndarray, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:

    '''Choose a valid, non-full column randomly and return it as `action`'''

    while True:
        action = randrange(0, board.shape[0])

        if  np.count_nonzero(board[:, action]) != board.shape[0]:
            return action, saved_state


