from subprocess import Popen, PIPE
import time
import numpy as np
import pyautogui
from PIL import Image
import PIL.ImageOps
import PIL.ImageGrab
import pytesseract
import cv2
from ..structs import GameState

PROCESS_NAME = "orbt xl"
BUNDLE_ID = "unity.Nickervision Studios.orbt xl"
MENU_BAR_HEIGHT = 22
SHADOW_WIDTH = 1
DEFAULT_WINDOW_SIZE_X = 825
DEFAULT_WINDOW_SIZE_Y = 547 - MENU_BAR_HEIGHT
DEFAULT_WINDOW_SIZE_4x3_X = 400
DEFAULT_WINDOW_SIZE_4x3_Y = 322 - MENU_BAR_HEIGHT
SCORE_BOX_HEIGHT = 40


def resize_image(image, ratio_x, ratio_y):
    return image.resize((
        int(image.size[0]*ratio_x),
        int(image.size[1]*ratio_y)
    ))


def invert_image_color(image):
    if image.mode == 'RGBA':
        r, g, b, a = image.split()
        rgb_image = Image.merge('RGB', (r, g, b))
        inverted_image = PIL.ImageOps.invert(rgb_image)
        r2, g2, b2 = inverted_image.split()
        return Image.merge('RGBA', (r2, g2, b2, a))
    else:
        return PIL.ImageOps.invert(image)


class MacOSController:
    def __init__(self):
        self.play_button_img = None
        self.retry_button_img = None
        self.window_size_x, self.window_size_y = None, None

    def activate_game(self):
        script = '''
        tell application id "{bundle_id}" to activate
        delay 1
        tell application "System Events" to set position of window 1 of application process "{process_name}" to {{0, 0}}
        '''.format(bundle_id=BUNDLE_ID, process_name=PROCESS_NAME) # noqa
        p = Popen(
            ['osascript', '-'],
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
            universal_newlines=True
        )
        _, stderr = p.communicate(script)
        if len(stderr.strip()) != 0:
            raise Exception(stderr)
        p.wait()

        self.__retrieve_window_size()

    def __retrieve_window_size(self):
        script = '''
        tell application "System Events" to tell application process "orbt xl"
            get size of window 1
        end tell
        '''
        p = Popen(
            ['osascript', '-'],
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
            universal_newlines=True
        )
        stdout, stderr = p.communicate(script)
        if len(stderr.strip()) != 0:
            raise Exception(stderr)
        dimensions = stdout.strip().split(',')
        try:
            self.window_size_x = int(dimensions[0])
            self.window_size_y = int(dimensions[1]) - MENU_BAR_HEIGHT
        except (IndexError, TypeError):
            raise Exception("Fail to read window size")

        if self.window_size_x / self.window_size_y != 4/3:
            self.play_button_img = Image.open(
                "orbtxlAI/button_images/playButton.png"
            )
            self.retry_button_img = Image.open(
                "orbtxlAI/button_images/retryButton.png"
            )
            default_size_x = DEFAULT_WINDOW_SIZE_X
            default_size_y = DEFAULT_WINDOW_SIZE_Y
        else:
            self.play_button_img = Image.open(
                "orbtxlAI/button_images/playButton_4x3.png"
            )
            self.retry_button_img = Image.open(
                "orbtxlAI/button_images/retryButton_4x3.png"
            )
            default_size_x = DEFAULT_WINDOW_SIZE_4x3_X
            default_size_y = DEFAULT_WINDOW_SIZE_4x3_Y

        ratio_x = self.window_size_x/default_size_x
        ratio_y = self.window_size_y/default_size_y
        self.play_button_img = resize_image(
            self.play_button_img,
            ratio_x,
            ratio_y
        )
        self.retry_button_img = resize_image(
            self.retry_button_img,
            ratio_x,
            ratio_y
        )

        p.wait()

    def __click_pixel(self, x, y):
        pyautogui.mouseDown(x, y)
        pyautogui.mouseUp(x, y)

    def __get_matching_button_coor(self, image):
        try:
            button = pyautogui.locateCenterOnScreen(
                image, grayscale=True, confidence=0.6
            )
            return (button.x//2, button.y//2)
        except TypeError:
            return None

    def __click_matching_button(self, image):
        button = self.__get_matching_button_coor(image)
        if button is not None:
            self.__click_pixel(button[0], button[1])
        return button

    def click_play_button(self):
        self.__click_matching_button(self.play_button_img)

    def click_retry_button(self):
        self.__click_matching_button(self.retry_button_img)

    def press(self, interval=0):
        pyautogui.keyDown("down")
        if interval > 0:
            time.sleep(interval)
        pyautogui.keyUp("down")

    def get_game_state(self):
        screenshot = self.capture_screenshot()
        for i in range(self.window_size_x//2):
            status_bar = screenshot.getpixel((i, 0))
            if status_bar[0:3] == (229, 229, 229):
                return GameState.RUNNING, screenshot

        if self.is_on_front_page():
            return GameState.UNINITIALIZED, screenshot

        if self.is_on_gameover_page():
            return GameState.GAME_OVER, screenshot

        return GameState.UNKNOWN, screenshot

    def is_on_gameover_page(self):
        return self.__get_matching_button_coor(self.retry_button_img)\
            is not None

    def is_on_front_page(self):
        return self.__get_matching_button_coor(self.play_button_img)\
            is not None

    def capture_screenshot(self):
        return PIL.ImageGrab.grab(bbox=(
            0,
            MENU_BAR_HEIGHT*2*2 + SHADOW_WIDTH*2,
            self.window_size_x*2,
            self.window_size_y*2 +
            MENU_BAR_HEIGHT*2*2 + SHADOW_WIDTH*2
        ))

    def pause_game(self):
        pyautogui.press("esc")

    def resume_game(self):
        pyautogui.press("esc")

    def recognize_score(self):
        score_image = PIL.ImageGrab.grab(bbox=(
            0,
            MENU_BAR_HEIGHT*2*2 + SHADOW_WIDTH*2,
            self.window_size_x*2,
            MENU_BAR_HEIGHT*2*2 + SHADOW_WIDTH*2 +
            int(SCORE_BOX_HEIGHT*self.window_size_y/DEFAULT_WINDOW_SIZE_4x3_Y)
        ))
        score_image = invert_image_color(score_image)

        opencvImage = cv2.cvtColor(np.array(score_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(opencvImage, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        score_string_df = pytesseract.image_to_data(
            gray,
            lang="digits",
            config="--psm 7",
            output_type='data.frame'
        )

        try:
            score = score_string_df.loc[score_string_df['conf'] > 90, 'text']\
                    .iloc[0]
            return score
        except IndexError:
            return 0
