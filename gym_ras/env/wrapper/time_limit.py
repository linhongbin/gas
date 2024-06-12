from gym_ras.env.wrapper.base import BaseWrapper


class TimeLimit(BaseWrapper):
    """ timelimit in a rollout """

    def __init__(self, env,
                 max_timestep=100,
                 **kwargs,
                 ):
        super().__init__(env)
        self._max_timestep = max_timestep

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        is_exceed = self.unwrapped.timestep >= self._max_timestep
        if is_exceed:
            info["fsm"] = "done_fail"
        return obs, reward, done, info
