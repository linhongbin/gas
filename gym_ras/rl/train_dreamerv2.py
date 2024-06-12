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


def train(origin_env, config, success_id=5.0, max_eps_length=300):
    env = common.GymWrapper(origin_env)
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
    print(f"******env seed: {env.seed}")
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

    train_replay = common.Replay(logdir / 'train_episodes', **config.replay)
    eval_replay = common.Replay(logdir / 'eval_episodes', **dict(
        capacity=1e4,
        minlen=config.dataset.length,
        maxlen=config.dataset.length))
    step = common.Counter(train_replay.stats['total_steps'])
    outputs = [
        common.TerminalOutput(),
        common.JSONLOutput(logdir),
        common.TensorBoardOutput(logdir),
    ]
    logger = common.Logger(step, outputs, multiplier=config.action_repeat)
    metrics = collections.defaultdict(list)

    should_train = common.Every(config.train_every)
    should_log = common.Every(config.log_every)
    should_video_train = common.Every(config.video_every)
    should_video_eval = common.Every(config.video_every)
    should_expl = common.Until(config.expl_until // config.action_repeat)

    # def make_env(mode):
    #   suite, task = config.task.split('_', 1)
    #   if suite == 'dmc':
    #     env = common.DMC(
    #         task, config.action_repeat, config.render_size, config.dmc_camera)
    #     env = common.NormalizeAction(env)
    #   elif suite == 'atari':
    #     env = common.Atari(
    #         task, config.action_repeat, config.render_size,
    #         config.atari_grayscale)
    #     env = common.OneHotAction(env)
    #   elif suite == 'crafter':
    #     assert config.action_repeat == 1
    #     outdir = logdir / 'crafter' if mode == 'train' else None
    #     reward = bool(['noreward', 'reward'].index(task)) or mode == 'eval'
    #     env = common.Crafter(outdir, reward)
    #     env = common.OneHotAction(env)
    #   else:
    #     raise NotImplementedError(suite)
    #   env = common.TimeLimit(env, config.time_limit)
    #   return env

    def per_episode(ep, mode):
        length = len(ep['reward']) - 1
        score = float(ep['reward'].astype(np.float64).sum())
        print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
        logger.scalar(f'{mode}_return', score)
        logger.scalar(f'{mode}_length', length)
        for key, value in ep.items():
            if re.match(config.log_keys_sum, key):
                logger.scalar(f'sum_{mode}_{key}', ep[key].sum())
            if re.match(config.log_keys_mean, key):
                logger.scalar(f'mean_{mode}_{key}', ep[key].mean())
            if re.match(config.log_keys_max, key):
                logger.scalar(f'max_{mode}_{key}', ep[key].max(0).mean())
        should = {'train': should_video_train, 'eval': should_video_eval}[mode]
        if should(step):
            for key in config.log_keys_video:
                logger.video(f'{mode}_policy_{key}', ep[key])
        replay = dict(train=train_replay, eval=eval_replay)[mode]
        logger.add(replay.stats, prefix=mode)
        logger.write()

    # print('Create envs.')
    # num_eval_envs = min(config.envs, config.eval_eps)
    # if config.envs_parallel == 'none':
    #   train_envs = [make_env('train') for _ in range(config.envs)]
    #   eval_envs = [make_env('eval') for _ in range(num_eval_envs)]
    # else:
    #   make_async_env = lambda mode: common.Async(
    #       functools.partial(make_env, mode), config.envs_parallel)
    #   train_envs = [make_async_env('train') for _ in range(config.envs)]
    #   eval_envs = [make_async_env('eval') for _ in range(eval_envs)]
    act_space = env.act_space
    obs_space = env.obs_space
    train_driver = common.Driver([env])
    train_driver.on_episode(lambda ep: per_episode(ep, mode='train'))
    train_driver.on_step(lambda tran, worker: step.increment())
    train_driver.on_step(train_replay.add_step)
    train_driver.on_reset(train_replay.add_step)
    eval_driver = common.Driver([env])
    eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval'))
    eval_driver.on_episode(eval_replay.add_episode)

    init_eval_stat = {'success_eps': 0,
                      'success_rate': 0, 'total_eps': 0, "score": []}
    eval_stat = init_eval_stat.copy()

    def eval_sucess_stat(ep):
        eval_stat['total_eps'] += 1
        # print(ep['fsm_state'][-1]==2.0)
        if ep['fsm_state'][-1] == success_id:
            eval_stat['success_eps'] += 1
        eps_length = ep['fsm_state'].shape[0] - 1
        score = (max_eps_length - eps_length) / max_eps_length
        eval_stat['score'].append(score)
        # print(f"sucess/progress/total: ({eval_stat['sucess_eps_count']}/ {eval_stat['eps_cnt']} ")
    eval_driver.on_episode(eval_sucess_stat)

    prefill_config = config.prefill.flat
    prefill_total = sum([v for k, v in prefill_config.items()])
    prefill_total = max(0, prefill_total - train_replay.stats['total_steps'])
    if prefill_total:
        print(f'Prefill dataset ({prefill_total} steps).')
        prefill_remain = prefill_total
        for k, v in prefill_config.items():
            if k == "random":
                prefill_agent = common.RandomAgent(act_space)
            elif k == "oracle":
                prefill_agent = common.OracleAgent(act_space, env=env)
            else:
                raise NotImplementedError
            to_prefill = min(prefill_remain, v)
            if to_prefill <= 0:
                break
            train_driver(prefill_agent, steps=to_prefill, episodes=1)
            train_driver.reset()
            prefill_remain -= to_prefill

    prefill_agent = common.RandomAgent(act_space)
    eval_driver(prefill_agent, episodes=1)
    eval_driver.reset()

    print('Create agent.')
    train_dataset = iter(train_replay.dataset(**config.dataset))
    report_dataset = iter(train_replay.dataset(**config.dataset))
    eval_dataset = iter(eval_replay.dataset(**config.dataset))
    agnt = agent.Agent(config, obs_space, act_space, step)
    train_agent = common.CarryOverState(agnt.train)
    train_agent(next(train_dataset))
    if (logdir / 'variables.pkl').exists():
        agnt.load(logdir / 'variables.pkl')
    else:
        print('Pretrain agent.')
        for _ in range(config.pretrain):
            train_agent(next(train_dataset))
    train_policy = lambda *args: agnt.policy(
        *args, mode='explore' if should_expl(step) else 'train')
    eval_policy = lambda *args: agnt.policy(*args, mode='eval')

    def train_step(tran, worker):
        if should_train(step):
            for _ in range(config.train_steps):
                mets = train_agent(next(train_dataset))
                [metrics[key].append(value) for key, value in mets.items()]
        if should_log(step):
            for name, values in metrics.items():
                logger.scalar(name, np.array(values, np.float64).mean())
                metrics[name].clear()
            logger.add(agnt.report(next(report_dataset)), prefix='train')
            logger.write(fps=True)
    train_driver.on_step(train_step)

    while step < config.steps:
        logger.write()
        print('Start evaluation.')
        origin_env.to_eval()
        logger.add(agnt.report(next(eval_dataset)), prefix='eval')
        eval_driver.reset()
        eval_stat = init_eval_stat.copy()
        eval_driver(eval_policy, episodes=config.eval_eps)
        eval_stat['success_rate'] = eval_stat['success_eps'] / \
            eval_stat['total_eps']
        eval_stat['score'] = np.mean(np.array(eval_stat['score']))
        logger.add(eval_stat, prefix='eval')
        print("============================")
        print(f"eval success rate: {eval_stat['success_rate']}")

        print('Start training.')
        origin_env.to_train()
        train_driver.reset()
        train_driver(train_policy, steps=config.eval_every)

        agnt.save(logdir / 'variables.pkl')
    env.close()
