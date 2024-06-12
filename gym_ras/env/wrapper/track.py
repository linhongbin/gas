
from gym_ras.env.wrapper.base import BaseWrapper
import numpy as np
import gym
import cv2
from copy import deepcopy


class Track(BaseWrapper):
    def __init__(self, env,
                 is_skip,
                 **kwargs):
        super().__init__(env)
        self._track_model = None

    def render(self,):
        img = self.env.render()
        if self._track_model is not None:
            img['mask'] = self._track_model.predict()
        return img
