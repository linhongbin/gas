from gym_ras.env.wrapper.base import BaseWrapper
import sys


class ActionOracle(BaseWrapper):
    KEYBOARD_MAP = {
        "w": 0,
        "s": 1,
        "a": 2,
        "d": 3,
        "k": 4,
        "i": 5,
        "j": 6,
        "l": 7,
        "n": 8,
    }

    def __init__(self, env,
                 device='keyboard',
                 **kwargs):
        super().__init__(env)
        self._device_type = device
        if device == "ds4":
            from gym_ras.tool.ds_util import DS_Controller
            self._device = DS_Controller()
        elif device in ['keyboard', 'script']:
            from gym_ras.tool.keyboard import Keyboard
            self._device = Keyboard()
        else:
            raise NotImplementedError

    def get_oracle_action(self):
        if self._device_type in ['keyboard', 'script']:
            while True:
                ch = self._device.get_char()
                # print(ch)
                if ch == 'q':
                    sys.exit(0)
                elif ch in self.KEYBOARD_MAP and self._device_type == "keyboard":
                    return self.KEYBOARD_MAP[ch]
                elif self._device_type == "script":
                    return self.env.get_oracle_action()
                else:
                    continue
