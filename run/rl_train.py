from gym_ras.api import make_env
from gym_ras.env.wrapper import Visualizer
from gym_ras.tool.config import Config, load_yaml
import argparse
from pathlib import Path
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--baseline', type=str, default="dreamerv2")
parser.add_argument('--baseline-tag', type=str,
                    nargs='+', default=['gas_surrol'])
parser.add_argument('--env-tag', type=str, nargs='+', default=['gas'])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--logdir', type=str, default="./log")
parser.add_argument('--reload-dir', type=str, default="")
parser.add_argument('--reload-envtag', type=str, nargs='+', default=[])
parser.add_argument('--online-eval', action='store_true')
parser.add_argument('--online-eps', type=int, default=10)
parser.add_argument('--visualize', action="store_true")
parser.add_argument('--vis-tag', type=str, nargs='+', default=[])
parser.add_argument('--save-prefix', type=str, default="")
args = parser.parse_args()

if args.baseline in ["dreamerv2"]:
    import tensorflow as tf  # need to first "import tf" and "import torch" after, otherwise will cause error, see https://discuss.pytorch.org/t/use-tensorflow-and-pytorch-in-same-code-caused-error/34537

if args.online_eval:
    assert args.reload_dir != ""


if args.reload_dir == "":
    env, env_config = make_env(tags=args.env_tag, seed=args.seed)
    method_config_dir = Path(".")
    method_config_dir = method_config_dir / 'gym_ras' / \
        'config' / str(args.baseline + ".yaml")
    if not method_config_dir.is_file():
        raise NotImplementedError("baseline not implement")
    yaml_dict = load_yaml(method_config_dir)
    yaml_config = yaml_dict["default"].copy()
    baseline_config = Config(yaml_config)
    # print(train_config)
    for tag in args.baseline_tag:
        baseline_config = baseline_config.update(yaml_dict[tag])

    _env_name = "ras"
    _baseline_name = baseline_config.baseline_name
    if len(args.baseline_tag) != 0:
        _baseline_name += "-" + "-".join(args.baseline_tag)

    if len(args.env_tag) != 0:
        _env_name += "-" + "-".join(args.env_tag)

    logdir = str(Path(args.logdir) / str(datetime.now().strftime("%Y_%m_%d-%H_%M_%S") +
                 '@'+_env_name + '@' + _baseline_name + '@seed' + str(args.seed)))
    logdir = Path(logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    if args.baseline == "DreamerfD":
        baseline_config = baseline_config.update({
            'bc_dir': str(logdir + '/train_episodes/oracle'),
            'logdir': str(logdir), })
    else:
        baseline_config = baseline_config.update({
            'logdir': str(logdir), })

    baseline_config.save(str(logdir / "baseline_config.yaml"))
    env_config.save(str(logdir / "env_config.yaml"))


else:
    reload_dir = Path(args.reload_dir)
    yaml_dict = load_yaml(str(reload_dir / "baseline_config.yaml"))
    baseline_config = Config(yaml_dict)

    yaml_dict = load_yaml(str(reload_dir / "env_config.yaml"))
    env_config = Config(yaml_dict)
    if len(args.reload_envtag) == 0:
        # auto increase seed number
        env, env_config = make_env(
            env_config=env_config, seed=env_config.seed+100)
    else:
        env, env_config = make_env(
            tags=args.reload_envtag, seed=env_config.seed+100)
    logdir = reload_dir
    baseline_config = baseline_config.update({"logdir": str(logdir)})
    baseline_config.save(str(logdir / "baseline_config.yaml"))
    env_config.save(str(logdir / "env_config.yaml"))

if args.visualize and args.online_eval:
    env = Visualizer(env, update_hz=30, vis_tag=args.vis_tag, keyboard=True)

if baseline_config.baseline_name == "DreamerfD":
    if args.online_eval:
        # from gym_ras.rl import train_DreamerfD
        # train_DreamerfD.train(env, config, is_pure_train=False, is_pure_datagen=False,)
        raise NotImplementedError
    else:
        from gym_ras.rl import train_DreamerfD
        train_DreamerfD.train(
            env, config, is_pure_train=False, is_pure_datagen=False)
elif baseline_config.baseline_name == "dreamerv2":
    if args.online_eval:
        from gym_ras.rl import eval_dreamerv2
        import csv
        env.to_eval()
        eval_stat = eval_dreamerv2.eval_agnt(
            env, baseline_config, eval_eps_num=args.online_eps, is_visualize=args.visualize, save_prefix=args.save_prefix)

    else:
        from gym_ras.rl import train_dreamerv2
        train_dreamerv2.train(env, baseline_config, )

elif baseline_config.baseline_name == "ppo":
    if args.online_eval:
        from gym_ras.rl import train_ppo
        env.to_eval()

        eval_stat = train_ppo.eval_ppo(
            env, baseline_config, n_eval_episodes=args.online_eps, save_prefix=args.save_prefix)
        # import yaml
        # _dir = Path("./data") / "exp_result"
        # _dir.mkdir(parents=True, exist_ok=True)
        # _file = _dir / (args.save_prefix + "@"+ str(datetime.now().strftime("%Y_%m_%d-%H_%M_%S")) +".yml")
        # with open(str(_file), 'w') as yaml_file:
        #     yaml.dump(eval_stat, yaml_file, default_flow_style=False)
    else:
        from gym_ras.rl import train_ppo
        train_ppo.train(env, baseline_config,
                        is_reload=not (args.reload_dir == ""))
