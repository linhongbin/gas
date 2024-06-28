from dreamerv2 import common
from dreamerv2 import agent
import ruamel.yaml as yaml
import numpy as np
import collections
import functools
import logging
import os
import pathlib
import re
import sys
import warnings
from tqdm import tqdm
try:
    import rich.traceback
    rich.traceback.install()
except ImportError:
    pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))


def eval_agnt(base_env, config, eval_eps_num, is_visualize=False, max_eps_length=300, save_prefix=""):
    env = common.GymWrapper(base_env)
    env = common.ResizeImage(env)
    if hasattr(env.act_space['action'], 'n'):
        env = common.OneHotAction(env)
    else:
        env = common.NormalizeAction(env)
    env = common.TimeLimit(env, config.time_limit)
    # configs = yaml.safe_load((
    #     pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
    # parsed, remaining = common.Flags(configs=['defaults']).parse(known_only=True)
    # config = common.Config(configs['defaults'])
    # for name in parsed.configs:
    #   config = config.update(configs[name])
    # config = common.Flags(config).parse(remaining)

    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    # config.save(logdir / 'config.yaml')
    print(config, '\n')
    print('Logdir', logdir)

    import tensorflow as tf
    tf.config.experimental_run_functions_eagerly(not config.jit)
    message = 'No GPU found. To actually train on CPU remove this assert.'
    assert tf.config.experimental.list_physical_devices('GPU'), message
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    assert config.precision in (16, 32), config.precision
    if config.precision == 16:
        from tensorflow.keras import mixed_precision as prec
        prec.set_global_policy(prec.Policy('mixed_float16'))

    eval_replay = common.Replay(logdir / 'online', **config.replay)
    step = common.Counter(eval_replay.stats['total_steps'])
    outputs = [
        common.TerminalOutput(),
    ]
#   logger = common.Logger(step, outputs, multiplier=config.action_repeat)
    metrics = collections.defaultdict(list)

    should_train = common.Every(config.train_every)
    should_log = common.Every(config.log_every)
    should_video_train = common.Every(config.eval_every)
    should_video_eval = common.Every(config.eval_every)
    should_expl = common.Until(config.expl_until // config.action_repeat)

#   def per_episode(ep, mode):
#     length = len(ep['reward']) - 1
#     score = float(ep['reward'].astype(np.float64).sum())
#     print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
#     logger.scalar(f'{mode}_return', score)
#     logger.scalar(f'{mode}_length', length)
#     for key, value in ep.items():
#       if re.match(config.log_keys_sum, key):
#         logger.scalar(f'sum_{mode}_{key}', ep[key].sum())
#       if re.match(config.log_keys_mean, key):
#         logger.scalar(f'mean_{mode}_{key}', ep[key].mean())
#       if re.match(config.log_keys_max, key):
#         logger.scalar(f'max_{mode}_{key}', ep[key].max(0).mean())
#     should = {'train': should_video_train, 'eval': should_video_eval}[mode]
#     if should(step):
#       for key in config.log_keys_video:
#         logger.video(f'{mode}_policy_{key}', ep[key])
#     replay = dict(train=train_replay, eval=eval_replay)[mode]
#     logger.add(replay.stats, prefix=mode)
#     logger.write()

    act_space = env.act_space
    obs_space = env.obs_space
    eval_driver = common.Driver([env])
#   eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval'))
    eval_driver.on_episode(eval_replay.add_episode)

    init_eval_stat = {'success_eps': 0,
                      'success_rate': 0, 'total_eps': 0, "score": []}
    eval_stat = init_eval_stat.copy()

    def eval_sucess_stat(ep):
        eval_stat['total_eps'] += 1
        # print(ep['fsm_state'][-1]==2.0)
        print(np.sum(ep['fsm_state'] == 5.0))
        print(ep['fsm_state'])
        if np.sum(ep['fsm_state'] == 5.0) > 0:
            eval_stat['success_eps'] += 1
            print("+++sucess episode !")
        eps_length = ep['fsm_state'].shape[0] - 1
        score = (max_eps_length - eps_length) / max_eps_length
        eval_stat['score'].append(score)
        # print(f"sucess/progress/total: ({e
        # print(f"sucess/progress/total: ({eval_stat['sucess_eps_count']}/ {eval_stat['eps_cnt']} ")

    def render_stuff(ep, **kwargs):
        img = base_env.render()
        img_break = base_env.cv_show(imgs=img)
        if img_break:
            sys.exit(0)
    eval_driver.on_episode(eval_sucess_stat)
    if is_visualize:
        eval_driver.on_step(render_stuff)

    while True:
        prefill_total = max(0, config.dataset.length -
                            eval_replay.stats['total_steps'])
        print("")
        print("======================")
        print(
            f"Prefill {prefill_total} steps for evaluation, please wait.....")
        print(f"DreamerV2 need to first prefill dataset and create networks using such dataset under tensorflow framework.")
        if prefill_total:
            prefill_agent = common.RandomAgent(act_space)
            eval_driver(prefill_agent, episodes=1)
            eval_driver.reset()
        else:
            print("waiting, robot is prefilling....")
            break


    print('Create agent.')
    eval_dataset = iter(eval_replay.dataset(**config.dataset))
    agnt = agent.Agent(config, obs_space, act_space, step)
    train_agent = common.CarryOverState(agnt.train)
    train_agent(next(eval_dataset))
    assert (logdir / 'variables.pkl').exists()
    agnt.load(logdir / 'variables.pkl')

    eval_policy = lambda *args: agnt.policy(*args, mode='eval')

    # logger.write()
    print('Start evaluation.')
    # logger.add(agnt.report(next(eval_dataset)), prefix='eval')
    eval_stat = init_eval_stat.copy()
    import yaml as yl
    from pathlib import Path
    from datetime import datetime
    _dir = Path("./data") / "exp_result"
    _dir.mkdir(parents=True, exist_ok=True)
    _file = _dir / (save_prefix + "@" +
                    str(datetime.now().strftime("%Y_%m_%d-%H_%M_%S")) + ".yml")
    for i in tqdm(range(eval_eps_num)):
        eval_driver.reset()
        eval_driver(eval_policy, episodes=1)
        eval_stat['success_rate'] = eval_stat['success_eps'] / \
            eval_stat['total_eps']
        # logger.add(eval_stat, prefix='eval')
        print("============================")
        print(f"eval success rate: {eval_stat['success_rate']}")
        with open(str(_file), 'w') as yaml_file:
            yl.dump(eval_stat, yaml_file, default_flow_style=False)
    eval_stat["score_mean"] = np.mean(np.array(eval_stat["score"])).item(0)
    eval_stat["score_std"] = np.std(np.array(eval_stat["score"])).item(0)
    print(eval_stat)
    with open(str(_file), 'w') as yaml_file:
        yl.dump(eval_stat, yaml_file, default_flow_style=False)
    return eval_stat

    # print('Start training.')
    # train_driver.reset()
    # train_driver(train_policy, steps=config.eval_every)

    # agnt.save(logdir / 'variables.pkl')
