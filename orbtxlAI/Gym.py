import time
from .structs import GameRecord, GameState, TimePoint


def empty_preprocessor(state):
    return state


class Gym:
    def __init__(
        self,
        controller,
        model,
        screenshot_preprocessor=empty_preprocessor
    ):
        self.__controller = controller
        self.__model = model
        self.__preprocessor = screenshot_preprocessor

    def __attempt_to_get_valid_state(self, max_attempts):
        state, screenshot = None, None
        for i in range(max_attempts):
            state, screenshot = self.__controller.get_game_state()
            if state != GameState.UNKNOWN:
                break
            if i != max_attempts - 1:
                time.sleep(0.5)
        return state, screenshot

    def __execute_game_action(self, action):
        if action.press_time > 0:
            self.__controller.press(action.press_time)
        if action.wait_time_after_press > 0:
            time.sleep(action.wait_time_after_press)

    def start_session(self, n_games, is_train):
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
            time_points = []
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
            self.__controller.pause_game()
            while state == GameState.RUNNING:
                preprocessed_screenshot = self.__preprocessor(screenshot)
                action = self.__model.get_action(preprocessed_screenshot)
                self.__controller.resume_game()
                self.__execute_game_action(action)
                time_points.append(TimePoint(preprocessed_screenshot, action))

                state, screenshot = self.__attempt_to_get_valid_state(3)
                self.__controller.pause_game()

            if state != GameState.GAME_OVER:
                raise Exception(
                    "Unexpected state at the end of game: {state}".format(
                        state=str(state)
                    )
                )
            score = self.__controller.recognize_score() - 5
            print("#{i}: {score}".format(i=i+1, score=score))
            self.__model.on_game_ended(GameRecord(time_points, score))
