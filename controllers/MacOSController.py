from subprocess import Popen, PIPE
import pyautogui
from PIL import Image
import time

PROCESS_NAME = "orbt xl"
BUNDLE_ID = "unity.Nickervision Studios.orbt xl"
MENU_BAR_HEIGHT = 22
DEFAULT_WINDOW_SIZE_X = 825
DEFAULT_WINDOW_SIZE_Y = 547 - MENU_BAR_HEIGHT
DEFAULT_WINDOW_SIZE_4x3_X = 400
DEFAULT_WINDOW_SIZE_4x3_Y = 322 - MENU_BAR_HEIGHT


def resize_image(image, ratio_x, ratio_y):
	return image.resize((
		int(image.size[0]*ratio_x),
		int(image.size[1]*ratio_y)
	))


class MacOSController:
	def __init__(self):
		self.play_button_img = Image.open("button_images/playButton.png")
		self.retry_button_img = Image.open("button_images/retryButton.png")
		self.window_size_x, self.window_size_y = None, None

	def activate_game(self):
		script = '''
		tell application id "{bundle_id}" to activate
		delay 1
		tell application "System Events" to set position of window 1 of application process "{process_name}" to {{0, 0}}
		'''.format(bundle_id=BUNDLE_ID, process_name=PROCESS_NAME)
		p = Popen(['osascript', '-'], stdin=PIPE, stdout=PIPE, stderr=PIPE, universal_newlines=True)
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
		p = Popen(['osascript', '-'], stdin=PIPE, stdout=PIPE, stderr=PIPE, universal_newlines=True)
		stdout, stderr = p.communicate(script)
		if len(stderr.strip()) != 0:
			raise Exception(stderr)
		dimensions = stdout.strip().split(',')
		try:
			self.window_size_x, self.window_size_y = int(dimensions[0]), int(dimensions[1]) - MENU_BAR_HEIGHT
		except (IndexError, TypeError) as e:
			raise Exception("Fail to read window size")

		if self.window_size_x / self.window_size_y != 4/3:	
			self.play_button_img = Image.open("button_images/playButton.png")
			self.retry_button_img = Image.open("button_images/retryButton.png")
			default_size_x, default_size_y = DEFAULT_WINDOW_SIZE_X, DEFAULT_WINDOW_SIZE_Y
		else:
			self.play_button_img = Image.open("button_images/playButton_4x3.png")
			self.retry_button_img = Image.open("button_images/retryButton_4x3.png")
			default_size_x, default_size_y = DEFAULT_WINDOW_SIZE_4x3_X, DEFAULT_WINDOW_SIZE_4x3_Y

		ratio_x = self.window_size_x/default_size_x
		ratio_y = self.window_size_y/default_size_y
		self.play_button_img = resize_image(self.play_button_img, ratio_x, ratio_y)
		self.retry_button_img = resize_image(self.retry_button_img, ratio_x, ratio_y)

		p.wait()

	def __click_pixel(self, x, y):
		pyautogui.mouseDown(x, y)
		pyautogui.mouseUp(x, y)

	def __get_matching_button_coor(self, image):
		try:
			button = pyautogui.locateCenterOnScreen(image, grayscale=True, confidence=0.8)
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

	def is_on_gameover_page(self):
		return self.__get_matching_button_coor(self.retry_button_img) is not None

	def is_on_front_page(self):
		return self.__get_matching_button_coor(self.play_button_img) is not None