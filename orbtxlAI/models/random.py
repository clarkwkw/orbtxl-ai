from ..structs import GameAction
import random


class RandomModel:
    def __init__(self):
        pass

    def get_action(self, screenshot):
        return GameAction(0.7+random.random())

    def on_game_ended(self, game_record):
        pass
