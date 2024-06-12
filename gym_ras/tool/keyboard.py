from pynput import keyboard
import time


class Keyboard():
    def __init__(self, update_hz=3):
        self._update_time = 1.0 / update_hz
        pass

    def get_char(self):
        # The event listener will be running in this block
        start = time.time()
        with keyboard.Events() as events:
            for event in events:
                if isinstance(event, keyboard.Events.Press) and not isinstance(event, keyboard.Events.Release):
                    break
                if (time.time() - start) > self._update_time:
                    return None
        return event.key.char if hasattr(event.key, "char") else event.key
