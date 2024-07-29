from gym_ras.env.wrapper.base import BaseWrapper
import gym
import numpy as np


class FSM(BaseWrapper):
    """ finite state machine 
    """

    def __init__(self, env,
                 states=[
                     "prog_norm",  # normal in progress
                     "prog_abnorm_1",  # abnormal 1 : hit workspace limit
                     "prog_abnorm_2",  # abnormal 2 : object sliding
                     "prog_abnorm_3",  # abnormal 3 : gripper toggle
                     "done_fail",  # done, failure case
                     "done_success",  # done success case
                 ],
                 ensure_norm_reward = False,
                 dsa_out_zoom_anamaly = True,
                 **kwargs
                 ):
        super().__init__(env)
        self._ensure_norm_reward = ensure_norm_reward
        self._states = states
        self._dsa_out_zoom_anamaly = dsa_out_zoom_anamaly

    def reset(self):
        obs = self.env.reset()
        fsm_state = "prog_norm"
        obs["fsm_state"] = self._states.index(fsm_state)
        return obs

    @property
    def fsm_states(self):
        return self._states

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if self._dsa_out_zoom_anamaly:
            if self.env.is_out_dsa_zoom: 
                if not info["fsm"] in [
                    "done_success",
                    "done_fail",
                    "prog_abnorm_1",
                    "prog_abnorm_2",
                ]:
                    info["fsm"] = "prog_abnorm_3"

        fsm_state = info["fsm"]

        # print(self.env.reward_cdict)
        reward = self.env.reward_dict[fsm_state]
        if self._ensure_norm_reward and fsm_state != "prog_norm":
            reward += self.unwrapped.reward_dict["prog_norm"]
        info['is_success'] = fsm_state == "done_success"
        if fsm_state in ["done_success", "done_fail"]:
            done = True
        else:
            done = False
        obs["fsm_state"] = self._states.index(fsm_state)
        # print("############",obs["fsm_state"])
        return obs, reward, done, info

    @property
    def observation_space(self):
        obs = {k: v for k, v in self.env.observation_space.items()}
        obs['fsm_state'] = gym.spaces.Box(low=0,
                                          high=len(self._states)-1, shape=(1,), dtype=np.float32)
        return gym.spaces.Dict(obs)
