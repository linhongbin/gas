from gym_ras.env.wrapper.base import BaseWrapper
import numpy as np
import gym


class ActionNoise(BaseWrapper):
    def __init__(self, env,
                 noise_scale=0.2,
                 **kwargs):
        super().__init__(env)
        self._is_discrete = hasattr(env.action_space, 'n')
        self._noise_scale = noise_scale

    def _add_noise(self, action):
        if not self._is_discrete:
            _low = self.env.action_space.low
            _high = self.env.action_space.high
            _range = (_high - _low)/2 * self._noise_scale
            _action_noise = self._action_noise_rng.uniform(-_range,  +_range)
            _action = action+_action_noise * \
                self.unwrapped._dr_scale[self.unwrapped.mode]
            _action = np.clip(_action, _low, _high)
        else:
            raise NotImplementedError
        return _action

    def step(self, action):
        action = self._add_noise(action)
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def _init_rng(self, seed):
        action_noise_seed = np.uint32(seed)
        print("action_noise_seed:", action_noise_seed)
        self._action_noise_rng = np.random.RandomState(action_noise_seed)
        if self.is_wrapper:
            self.env._init_rng(self.env.seed)
