import time
from .GameState import GameState


class Gym:
    def __init__(self, controller, model):
        self.__controller = controller
        self.__model = model

    def start_training_session(self, n_games):
        self.__start_session(n_games, True)

    def start_demo_session(self, n_games):
        pass

    def __attempt_to_get_valid_state(self, max_attempts):
        state, screenshot = None, None
        for i in range(max_attempts):
            state, screenshot = self.__controller.get_game_state()
            print(state)
            if state != GameState.UNKNOWN:
                return state, screenshot
            if i != max_attempts - 1:
                time.sleep(0.5)
        return state, screenshot

    def __start_session(self, n_games, is_train):
        self.__controller.activate_game()
        state, _ = self.__controller.get_game_state()
        if state != GameState.UNINITIALIZED:
            raise Exception(
                "Can only start a session "
                "when it is on the front page, "
                "(state = {state})".format(
                    state=state
                )
            )
        for i in range(n_games):
            state, _ = self.__attempt_to_get_valid_state(5)
            if state == GameState.UNINITIALIZED:
                self.__controller.click_play_button()
            elif state == GameState.GAME_OVER:
                self.__controller.click_retry_button()
            else:
                raise Exception(
                    "Unexpected state at the start of game: {state}".format(
                        state=str(state)
                    )
                )
            state, screenshot = self.__attempt_to_get_valid_state(3)
            while state == GameState.RUNNING:
                action = self.__model.get_action(screenshot)
                if action.press_time > 0:
                    self.__controller.press(action.press_time)

                state, screenshot = self.__attempt_to_get_valid_state(3)

            if state != GameState.GAME_OVER:
                raise Exception(
                    "Unexpected state at the end of game: {state}".format(
                        state=str(state)
                    )
                )
            score = self.__controller.recognize_score()
            print("#{i}: {score}".format(i=i+1, score=score))
