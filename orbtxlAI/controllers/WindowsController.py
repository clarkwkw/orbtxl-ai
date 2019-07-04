import win32gui
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
LEFT_PADDING = 10
RIGHT_PADDING = 12
TOP_PADDING = 45
BOTTOM_PADDING = 11

SCORE_BOX_HEIGHT = 40
DEFAULT_WINDOW_SIZE_X = 1650
DEFAULT_WINDOW_SIZE_Y = 1050
DEFAULT_WINDOW_SIZE_4x3_X = 800
DEFAULT_WINDOW_SIZE_4x3_Y = 600


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


class WindowsController:
    def __init__(self):
        self.window_handle = win32gui.FindWindow(None, 'orbt xl')
        self.play_button_img = None
        self.retry_button_img = None
        x1, y1, x2, y2 = win32gui.GetWindowRect(self.window_handle)
        self.window_size_x = x2-x1-LEFT_PADDING-RIGHT_PADDING
        self.window_size_y = y2-y1-TOP_PADDING-BOTTOM_PADDING
        self.__resize_button_image()

    def activate_game(self):
        win32gui.MoveWindow(
            self.window_handle,
            0,
            0,
            self.window_size_x+LEFT_PADDING+RIGHT_PADDING,
            self.window_size_y+TOP_PADDING+BOTTOM_PADDING,
            True
        )
        win32gui.SetForegroundWindow(self.window_handle)

    def __resize_button_image(self):
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

    def __click_pixel(self, x, y):
        pyautogui.mouseDown(x, y)
        pyautogui.mouseUp(x, y)

    def __get_matching_button_coor(self, image):
        try:
            button = pyautogui.locateCenterOnScreen(
                image, grayscale=True, confidence=0.6
            )
            return (button.x, button.y)
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
        status_bar = screenshot.getpixel((self.window_size_x//2, 0))
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
            LEFT_PADDING,
            TOP_PADDING,
            LEFT_PADDING+self.window_size_x,
            TOP_PADDING+self.window_size_y
        ))

    def pause_game(self):
        pyautogui.press("esc")

    def resume_game(self):
        pyautogui.press("esc")

    def recognize_score(self):
        score_image = PIL.ImageGrab.grab(bbox=(
            LEFT_PADDING,
            TOP_PADDING,
            LEFT_PADDING+self.window_size_x,
            TOP_PADDING +
            int(SCORE_BOX_HEIGHT*self.window_size_y/DEFAULT_WINDOW_SIZE_4x3_Y)
        ))
        score_image = invert_image_color(score_image)

        opencvImage = cv2.cvtColor(np.array(score_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(opencvImage, cv2.COLOR_BGR2GRAY)
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
