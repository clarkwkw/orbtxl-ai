from orbtxlAI.controllers import MacOSController
from orbtxlAI import Gym
from orbtxlAI.models import RandomModel

controller = MacOSController()
controller.activate_game()
print("gameover:", controller.is_on_gameover_page())
print("front page:", controller.is_on_front_page())
model = RandomModel()
gym = Gym(controller, model)
gym.start_training_session(3)
