import gym
from gym_ras.env.embodied.base_env import BaseEnv
import numpy as np
from gym_ras.tool.common import *


class dVRKEnv(BaseEnv):
    def __init__(self,
                 task,
                 dr_scale_train=1,
                 dr_scale_eval=1,
                 **kwargs,
                 ):
        if task in "grasp_any":
            from gym_ras.env.embodied.dvrk.grasp_any import GraspAny
            client = GraspAny(**kwargs)
        else:
            raise Exception("Not support")
        self._dr_scale = {"train": dr_scale_train, "eval": dr_scale_eval, }
        self._mode = "train"
        super().__init__(client)

    def reset(self):
        self._init_vars()
        _ = self.client.reset()
        obs, reward, done, info = self.client.step(
            np.array([0.0, 0.0, 0.0, 0.0, 1.0]))
        obs["robot_prio"], obs["gripper_state"] = self._get_prio_obs()
        self.step_func_prv = (obs, reward, done, info)
        return obs

    def step(self, action):
        self.timestep += 1
        if self.timestep % 20 == 0:
            print("Step:", self.timestep)
        if (not self.skip) or (self.step_func_prv is None):
            _prio, _ = self._get_prio_obs()
            _is_out, action_clip = self._check_new_action(_prio, action[:3])
            _action = action.copy()
            _action[:3] = action_clip
            obs, reward, done, info = self.client.step(_action)
            obs["robot_prio"], obs["gripper_state"] = self._get_prio_obs()
            gripper_toggle = np.abs(
                obs["gripper_state"]-self.step_func_prv[0]["gripper_state"]) > 0.1
            _str = self._get_fsm_prog_abnorm(gripper_toggle, _is_out)
            if _str != "":
                info["fsm"] = _str

            self.step_func_prv = obs, reward, done, info

        return self.step_func_prv

    def _get_fsm_prog_abnorm(self, gripper_toggle, is_out):
        if is_out:
            return "prog_abnorm_1"
        if gripper_toggle:
            return "prog_abnorm_3"
        return ""

    def render(self, **kwargs):  # ['human', 'rgb_array', 'mask_array']
        return self.client.render()

    def get_oracle_action(self, **kwargs):
        return self.client.get_oracle_action()

    def __getattr__(self, name):
        """__getattr__ is only invoked if the attribute wasn't found the usual ways."""
        if name[0] == "_":
            raise Exception("cannot find {}".format(name))
        else:
            return getattr(self.client, name)

    @property
    def reward_dict(self):
        return self.client.reward_dict

    def _check_new_action(self, state, action):
        pos = action * self.action2pos
        new_state = state + pos
        ws = self.workspace_limit
        _low = ws[:, 0]
        _high = ws[:, 1]
        # print("low exceed", new_state < _low, new_state - _low)
        # print("high exceed", new_state > _high)
        is_out_ws = np.any(new_state < _low) or (np.any(new_state > _high))

        eps = np.ones(_low.shape) * 0.0001  # numerical stable
        new_state_clip = np.clip(new_state, _low+eps, _high-eps)
        action_clip = (new_state_clip - state) / self.action2pos
        return is_out_ws, action_clip

    def _get_prio_obs(self):
        # norm by workspace
        obs = self.client.get_prio_obs()
        return obs["robot_prio"],  obs["gripper_state"]

    @property
    def action2pos(self,):
        return self.client.max_step_pos

    @property
    def mode(self,):
        return self._mode

    def to_eval(self):
        self._mode = "eval"

    def to_train(self):
        self._mode = "train"
