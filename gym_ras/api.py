import gym_ras
from gym_ras.env.wrapper import *
from gym_ras.tool.config import Config, load_yaml
from pathlib import Path

from gym_ras.tool.config import Config


def make_env(env_config=None, tags=[], seed=0):
    assert isinstance(tags, list)
    if env_config is None:
        yaml_dir = Path(__file__).absolute().parent / "config" / "gym_ras.yaml"
        yaml_dict = load_yaml(yaml_dir)
        yaml_config = yaml_dict["default"].copy()
        config = Config(yaml_config)
    else:
        config = env_config
    # print(train_config)
    for tag in tags:
        config = config.update(yaml_dict[tag])
    if config.embodied_name == "SurrolEnv":
        from gym_ras.env.embodied.surrol import SurrolEnv
        _call = SurrolEnv
    elif config.embodied_name == "dVRKEnv":
        from gym_ras.env.embodied.dvrk import dVRKEnv
        _call = dVRKEnv
    else:
        raise NotImplementedError
    # print(config)
    embodied_args = getattr(config.embodied, config.embodied_name)
    _kwargs = {}
    _kwargs["task"] = config.task_name
    task_name = config.task_name
    _kwargs.update(getattr(embodied_args, task_name).flat)
    for k, v in embodied_args.flat.items():
        if k.find(task_name) < 0:
            _kwargs.update({k: v})
    env = _call(**_kwargs)

    for wrapper in config.wrapper.pipeline:
        if hasattr(gym_ras.env.wrapper, wrapper):
            _call = getattr(gym_ras.env.wrapper, wrapper)
            _kwargs = getattr(config.wrapper, wrapper).flat
            env = _call(env, **_kwargs)
    env.seed = seed
    config = config.update({"seed": seed})
    return env, config
