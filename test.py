from orbtxlAI.controllers import MacOSController
from orbtxlAI import Gym
from orbtxlAI.models import PolicyGradientModel
from orbtxlAI.structs import GameAction
import cv2
import numpy as np


def preprocess_screenshot(screenshot):
    opencvImage = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(opencvImage, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    return gray


controller = MacOSController()
model = PolicyGradientModel(
    actions=[
        GameAction(
            press_time=0,
            wait_time_after_press=0.5
        ),
        GameAction(
            press_time=0.7,
            wait_time_after_press=0
        ),
        GameAction(
            press_time=1,
            wait_time_after_press=0
        )
    ],
    sample_shape=[480, 640]
)
gym = Gym(controller, model, preprocess_screenshot)
gym.start_session(1000, is_train=True)
