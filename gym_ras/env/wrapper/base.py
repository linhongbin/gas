class BaseWrapper():
    def __init__(self,
                 env,
                 **kwargs):
        self.env = env

    @property
    def unwrapped(self):
        """Returns the base environment of the wrapper."""
        return self.env.unwrapped

    def __getattr__(self, name):
        """__getattr__ is only invoked if the attribute wasn't found the usual ways."""
        if name[0] == "_":
            raise Exception("cannot find {}".format(name))
        else:
            return getattr(self.env, name)

    def _reset_vars(self):
        pass

    def reset(self):
        obs = self.env.reset()
        self._reset_vars()
        return obs

    @property
    def seed(self):
        return self.env.seed + 10  # use incremental seed

    @property
    def is_wrapper(self):
        return hasattr(self.env, "env")

    @seed.setter
    def seed(self, seed):
        self.unwrapped.seed = seed
        self._init_rng(self.seed)

    def _init_rng(self, seed):
        if self.is_wrapper:
            self.env._init_rng(self.env.seed)

    def get_wrap_obj(self, class_name):
        if class_name == self.__class__.__name__:
            return self
        else:
            return self.env.get_wrap_obj(class_name)
