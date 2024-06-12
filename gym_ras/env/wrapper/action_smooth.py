from gym_ras.env.wrapper.base import BaseWrapper
import numpy as np
from gym_ras.tool.common import ema


class ActionSmooth(BaseWrapper):
    def __init__(self, env,
                 window=3,
                 smooth_type='ema',
                 skip=True,
                 **kwargs):
        super().__init__(env)
        self._reset_var()
        self._window = window
        self._skip = skip
        self._smooth_type = smooth_type

    def reset(self,):
        obs = self.env.reset()
        if not self._skip:
            self._reset_var()
        return obs

    def step(self, action):
        if not self._skip:
            self._action_list.append(action[:3].copy())
            self._action_list = self._action_list[max(
                len(self._action_list)-self._window, 0):]
            if self._smooth_type == 'ema':
                _action = ema(self._action_list, self._window)[-1]
            else:
                raise NotImplementedError
            action[:3] = _action
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def _reset_var(self):
        self._action_list = []
