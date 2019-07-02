from enum import Enum


class GameState(Enum):
    UNKNOWN = -2
    UNINITIALIZED = -1
    GAME_OVER = 0
    RUNNING = 1
