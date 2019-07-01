from subprocess import Popen, PIPE
import pyautogui
from PIL import Image
import time

PROCESS_NAME = "orbt xl"
BUNDLE_ID = "unity.Nickervision Studios.orbt xl"
DEFAULT_WINDOW_SIZE_X = 825
DEFAULT_WINDOW_SIZE_Y = 547

'''
TODO: button recognition in other resolutions
'''

class MacController:
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
			self.window_size_x, self.window_size_y = int(dimensions[0]), int(dimensions[1])
		except (IndexError, TypeError) as e:
			raise Exception("Fail to read window size")
		p.wait()

	def __click(self, x, y):
		pyautogui.mouseDown(x, y)
		pyautogui.mouseUp(x, y)

	def __click_matching_button(self, image):
		try:
			button = pyautogui.locateCenterOnScreen(image, grayscale=True, confidence=0.8)
			x, y = button.x//2, button.y//2
			self.__click(x, y)
			return (x, y)
		except TypeError:
			return None

	def click_play_button(self):
		if self.__click_matching_button(self.play_button_img) is None:
			self.__click(413, 413)

	def click_retry_button(self):
		if self.__click_matching_button(self.retry_button_img) is None:
			self.__click(412, 384)

	def press(self, interval=0):
		pyautogui.keyDown("down")
		if interval > 0:
			time.sleep(interval)
		pyautogui.keyUp("down")
