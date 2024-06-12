from abc import ABC, abstractmethod


class BaseEnv(ABC):
    def __init__(self, client):
        self.client = client
        self._init_vars()
        self._seed = 0

    def _init_vars(self):
        self.timestep = 0
        self.skip = False
        self.step_func_prv = None

    @abstractmethod
    def reset(self):
        self.timestep = 0
        obs = self.client.reset()
        return obs

    @abstractmethod
    def step(self, action):
        self.timestep += 1
        obs, reward, done, info = self.client.step(action)
        self.step_func_prv = obs, reward, done, info
        return obs, reward, done, info

    @abstractmethod
    def render(self, mode="human"):  # ['human', 'rgb_array', 'mask_array']
        return self.client.render(mode=mode)

    @abstractmethod
    def get_oracle_action(self, obs):
        return self.client.get_oracle_action(obs)

    @property
    @abstractmethod
    def reward_dict(self, ):
        return self.client._reward_dict

    @property
    def action_space(self):
        return self.client.action_space

    @property
    def observation_space(self):
        return self.client.observation_space

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self.client.seed = seed

    def _init_rng(self):
        pass

    @property
    def unwrapped(self):
        return self

    def __del__(self):
        # cv2.destroyAllWindows()
        del self.client

    @property
    def is_wrapper(self):
        return False
