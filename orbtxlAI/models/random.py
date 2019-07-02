from ..GameAction import GameAction
import random


class RandomModel:
    def __init__(self):
        pass

    def get_action(self, screenshot):
        return GameAction(0.7+random.random())
