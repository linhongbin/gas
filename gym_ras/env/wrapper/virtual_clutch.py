from gym_ras.env.wrapper.base import BaseWrapper


class VirtualClutch(BaseWrapper):
    """ Virtual Clutch """

    def __init__(self,
                 env,
                 start=6,
                 **kwargs):
        super().__init__(env)
        self._start = start

    def _reset_vars(self):
        self.env._reset_vars()
        self._clutch = False  # False for open, True for closed

    def step(self, action):
        self._clutch = self.unwrapped.timestep >= self._start  # time-dependent clutch
        self.unwrapped.skip = not self._clutch
        return self.env.step(action)

    @property
    def clutch_state(self):
        return self._clutch
