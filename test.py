from orbtxlAI.controllers import MacOSController
from orbtxlAI import Gym
from orbtxlAI.models import PolicyGradientModel

controller = MacOSController()
model = PolicyGradientModel(
    actions=[0, 0.8, 1.2],
    sample_shape=[600, 800]
)
gym = Gym(controller, model)
gym.start_session(3, is_train=True)
