class GameAction:
    def __init__(self, press_time, wait_time_after_press):
        self.press_time = press_time
        self.wait_time_after_press = wait_time_after_press

    def serialize(self):
        return {
            'press_time': self.press_time,
            'wait_time_after_press': self.wait_time_after_press
        }

    @classmethod
    def deserialize(cls, d):
        return cls(
            press_time=d['press_time'],
            wait_time_after_press=d['wait_time_after_press']
        )
