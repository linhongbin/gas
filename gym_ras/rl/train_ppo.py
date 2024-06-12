import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import stable_baselines3
import torch
from r3m import load_r3m
from gym_ras.api import make_env
import os
from pathlib import Path
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from pathlib import Path


class MyCheckpointCallback(CheckpointCallback):
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            print("")
            print(
                f"=============timestep {self.n_calls} =============================")
            print("")
            path = os.path.join(self.save_path, f"{self.name_prefix}")
            self.model.save(path)
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")
        return True


class Env:

    metadata = {}

    def __init__(self, env):
        # self.env = embodied.envs.load_single_env('ur5_real', length=100)
        self.env = env
        self.num_actions = self.env.action_space.n
        self.score = 0
        self.length = 0
        self.r3m = load_r3m("resnet50")
        self.r3m.cuda()
        self.r3m.eval()

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self.env.observation_space.items():
            if key.startswith('log_'):
                continue
            if key.startswith('is_'):
                continue
            if key in ('reward', 'depth'):
                continue
            if key == 'image':
                spaces[key] = gym.spaces.Box(
                    -np.inf, np.inf, (2048,), np.float)
                continue

            spaces[key] = gym.spaces.Box(
                value.low, value.high, value.shape, value.dtype)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return gym.spaces.Discrete(self.num_actions)

    def reset(self):
        obs = self.env.reset()
        with torch.no_grad():
            image = torch.tensor(obs['image'].copy()).cuda()
            obs['image'] = self.r3m(image.permute(2, 0, 1)[None])[
                0].cpu().numpy()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        with torch.no_grad():
            image = torch.tensor(obs['image'].copy()).cuda()
            obs['image'] = self.r3m(image.permute(2, 0, 1)[None])[
                0].cpu().numpy()
        return obs, reward, done, info

    def _stack_obs(self, obs):
        obs['image']


def eval_ppo(env, config, n_eval_episodes, save_prefix):
    _dir = config.logdir
    env = Env(env)
    model = PPO(
        stable_baselines3.common.policies.MultiInputActorCriticPolicy,
        env, verbose=10,
        tensorboard_log=_dir
    )
    from pathlib import Path
    model.load(str(Path(config.logdir) / "best_model.zip"),
               print_system_info=True)
    vec_env = DummyVecEnv([lambda: env])
    eval_stat = {'success_eps': 0, 'success_rate': 0,
                 'total_eps': 0, "score": []}
    max_eps_length = 300

    import yaml as yl
    from pathlib import Path
    from datetime import datetime
    _dir = Path("./data") / "exp_result"
    _dir.mkdir(parents=True, exist_ok=True)
    _file = _dir / (save_prefix + "@" +
                    str(datetime.now().strftime("%Y_%m_%d-%H_%M_%S")) + ".yml")

    for j in range(n_eval_episodes):
        obs = vec_env.reset()
        done = False
        eps_length = 0
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = vec_env.step(action)
            eps_length += 1
            done = dones[0]
            print(eps_length)
        if info[0]['fsm'] == "done_success":
            eval_stat['success_eps'] += 1
            print("+++sucess episode !")

        eval_stat['total_eps'] += 1
        score = (max_eps_length - eps_length) / max_eps_length
        eval_stat['score'].append(score)
        eval_stat['success_rate'] = eval_stat['success_eps'] / \
            eval_stat['total_eps']
        print(eval_stat)
        with open(str(_file), 'w') as yaml_file:
            yl.dump(eval_stat, yaml_file, default_flow_style=False)

    eval_stat["score_mean"] = np.mean(np.array(eval_stat["score"])).item(0)
    eval_stat["score_std"] = np.std(np.array(eval_stat["score"])).item(0)
    return eval_stat


def train(env, config, is_reload=False, only_eval=False):
    _dir = config.logdir
    env = Env(env)
    model = PPO(
        stable_baselines3.common.policies.MultiInputActorCriticPolicy,
        env, verbose=1,
        tensorboard_log=_dir
    )
    if is_reload:
        model.load(str(Path(config.logdir) / "best_model.zip"),
                   print_system_info=True)

    eval_callback = EvalCallback(env, best_model_save_path=_dir,
                                 log_path=_dir if not only_eval else str(
                                     Path(config.logdir) / "online_results"),
                                 n_eval_episodes=config.eval_eps,
                                 eval_freq=config.eval_freq,
                                 deterministic=True,
                                 render=False)
    cbs = [eval_callback]
    if not only_eval:
        checkpoint_cb = MyCheckpointCallback(
            save_freq=config.eval_eps, save_path=_dir, name_prefix="checkpoint", verbose=1,)
        cbs.append(checkpoint_cb)


# if __name__  == "__main__":
#     env, env_config = make_env(tags=["dvrk_cam_setting","grasp_any"], seed=0)
#     _dir = "./log_ppo/"
#     env = Env(env)
#     model = stable_baselines3.PPO(
#         stable_baselines3.common.policies.MultiInputActorCriticPolicy,
#         env, verbose=1,
#         tensorboard_log=_dir
#         )

    eval_callback = EvalCallback(env, best_model_save_path=_dir,
                                 log_path=_dir,
                                 #  eval_freq=3e4,
                                 eval_freq=1e2,
                                 deterministic=True,
                                 render=False)

    model.learn(total_timesteps=1e6, callback=[eval_callback, checkpoint_cb])
