import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('-p',type=int)
parser.add_argument('--repeat',type=int, default=1)
parser.add_argument('--action',type=str, default="3")

parser.add_argument('--yaml-dir', type=str, default="./gym_ras/config.yaml")
parser.add_argument('--yaml-tag', type=str, nargs='+', default=[])
parser.add_argument('--section', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--eval-eps', type=int, default=20)
parser.add_argument('--prefill', type=int, default=8000) # <0 means following default settings
parser.add_argument('--logdir', type=str, default="./log")
parser.add_argument('--log-video-every', type=int, default=1000) # <0 means consistent with config file
parser.add_argument('--env-tag', type=str, nargs='+', default=[])

args = parser.parse_args()

if args.p ==1:
    from gym_ras.env.client.surrol import SurrolEnv
    from gym_ras.env.wrapper.visualizer import Visualizer
    from gym_ras.env.wrapper.dsa import DSA
    from gym_ras.env.wrapper.obs import OBS
    env = SurrolEnv(task="needle_pick")
    env = DSA(env)
    env = OBS(env)
    env = Visualizer(env)
    print("obs space: ", env.observation_space)
    print("action space: ", env.action_space)


    done = False
    obs = env.reset()
    print("obs:", obs)
    while not done:
        action = env.get_oracle_action()
        # action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        img = env.render(mode="DSA")
        if env.cv_show(**img):
            break

elif args.p ==2:
    from gym_ras.env.client.surrol import SurrolEnv
    from gym_ras.env.wrapper.visualizer import Visualizer
    import numpy as np
    env = SurrolEnv(task="needle_pick")
    env = Visualizer(env)
    env.reset()
    rgb, mask = env.render(mode="rgb_mask_array")
    print(rgb)
    print(mask)
    
    needle_id = 6
    instrument_id = 1
    # mask = np.uint8(mask==needle_id)
    mask = np.uint8(mask==instrument_id)
    scale_func = lambda _input,old_min,old_max,new_min,new_max: (_input-old_min)/(old_max-old_min)*(new_max-new_min) + new_min
    mask = scale_func(mask, np.min(mask), np.max(mask), 0, 255)
    mask = np.uint8(np.clip(mask, 0, 255))
    print(mask)
    while True:      
        if env.cv_show(np.stack([mask]*3,axis=2)):
            break
if args.p ==3:
    from gym_ras.env.client.surrol import SurrolEnv
    env = SurrolEnv(task="needle_pick")
    pose = env._get_obj_pose("needle", 4)
    print(pose)
if args.p == 4:
    # from surrol.tasks.pick_and_place import PickAndPlace
    # env = PickAndPlace(render_mode="human")
    # from surrol.tasks.match_board_ii import MatchBoardII
    # env = MatchBoardII(render_mode="human")
    # from surrol.tasks.match_board import MatchBoard
    # env = MatchBoard(render_mode="human")
    # from surrol.tasks.needle_the_rings import NeedleRings
    # env = NeedleRings(render_mode="human")
    # from surrol.tasks.gauze_retrieve import GauzeRetrieve
    # env = GauzeRetrieve(render_mode="human")
    from surrol.tasks.peg_transfer import PegTransfer
    env = PegTransfer(render_mode="human")
    import cv2
    
    env.seed = args.s
    obs = env.reset()
    rgb = env.render()

    print(obs)

    done = False
    while not done:
        action = env.get_oracle_action(obs)
        obs, reward, done, info = env.step(action)
        # print(obs, reward, done, info)
        img = env.render(mode="rgb_array")
        print(action, done, info)
        while True:
            
            img = cv2.resize(img, 
                                (2000, 2000),
                                interpolation=cv2.INTER_NEAREST)
            cv2.imshow('preview', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            k = cv2.waitKey(0)
            cv2.setWindowTitle(
                'preview', 'press n to continue, q to quit')
            if k & 0xFF == ord('q'):    # Esc key to stop
                done = True
                break
            elif k & 0xFF == ord('n'):
                break
            else:
                continue
if args.p == 5:
    from gym_ras.env.client.surrol import SurrolEnv
    from gym_ras.env.wrapper import *
    import numpy as np
    from tqdm import tqdm
    env = SurrolEnv(task="needle_pick", pybullet_gui=False)
    env = DSA(env)
    env = VirtualClutch(env)
    env = DiscreteAction(env)
    env = TimeLimit(env)
    env = FSM(env)
    env = OBS(env)
    env = Visualizer(env)
    env.seed = args.s
    print("obs space: ", env.observation_space)
    print("action space: ", env.action_space)

    for _ in tqdm(range(args.repeat)):
        done = False
        obs = env.reset()
        print("obs:", obs)
        while not done:
            action = env.get_oracle_action()
            # action = env.action_space.sample()
            # print(action)
            action = 8
            obs, reward, done, info = env.step(action)
            print(obs.keys())
            print("reward:", reward, "done:", done, "info:", info, "step:", env.timestep, "obs_key:", obs.keys(), "fsm_state:", obs["fsm_state"])
            img = env.render(mode="DSA")
            print(img.keys())
            obs.update(img)
            
            if info['is_success']:
                break
            img_break = env.cv_show(imgs=obs)
            if img_break:
                break
        if img_break:
            break
if args.p == 6:
    from gym_ras.env.client.surrol import SurrolEnv
    from gym_ras.env.wrapper import *
    from pympler.tracker import SummaryTracker
    tracker = SummaryTracker()
    import tracemalloc
    tracemalloc.start()
    from tqdm import tqdm
    env = SurrolEnv(task="needle_pick", pybullet_gui=False)
    env = DSA(env)
    env = VirtualClutch(env)
    env = DiscreteAction(env)
    env = TimeLimit(env)
    env = FSM(env)
    env = OBS(env)
    done = False
    tracker.print_diff()
    obs = env.reset()
    for _ in tqdm(range(args.repeat)):
        done = False
        # tracker.print_diff()
        snapshot = tracemalloc.take_snapshot()
        obs = env.reset()
        snapshot1 = tracemalloc.take_snapshot()
        top_stats = snapshot1.compare_to(snapshot, 'lineno')
        print("[ Top 10 differences ]")
        for stat in top_stats[:10]:
            if stat.size_diff < 0:
                continue
            print(stat)

        # print("obs:", obs)
        tracker.print_diff()
        while not done:
            action = env.get_oracle_action()
            # action = env.action_space.sample()
            # print(action)
            action = 8
            obs, reward, done, info = env.step(action)

            # print(obs.keys())
            # print("reward:", reward, "done:", done, "info:", info, "step:", env.timestep, "obs_key:", obs.keys(), "fsm_state:", obs["fsm_state"])
            img = env.render(mode="DSA")
            # print(img.keys())
            obs.update(img)
            
            if info['is_success']:
                break


if args.p == 7:
    from gym_ras import load_yaml, make_env
    from gym_ras.env.wrapper import Visualizer
    import argparse
    from tqdm import tqdm
    from dreamer_fd import common
    yaml_dict = load_yaml(args.yaml_dir)
    if "default" in yaml_dict:
        yaml_config = yaml_dict["default"].copy()
        config = common.Config(yaml_config)
        # print(train_config)
        for tag in args.yaml_tag:
            config = config.update(yaml_dict[tag])
    else:
        raise NotImplementedError

    env_config = common.Config(config.env.flat)
    env = make_env(env_config)
    env =  Visualizer(env)
    env.seed = args.seed
    for _ in tqdm(range(args.repeat)):
        done = False
        obs = env.reset()
        # print("obs:", obs)
        while not done:
            # action = env.action_space.sample()
            # print(action)
            print("==========step", env.timestep, "===================")
            if any(i.isdigit() for i in args.action):
                action = int(args.action)
            elif args.action == "random":
                action = env.action_space.sample()
            elif args.action == "oracle":
                action = env.get_oracle_action()
            else:
                raise NotImplementedError
            print("step....")
            obs, reward, done, info = env.step(action)
            print_obs = obs.copy()
            print_obs = {k: v.shape if k in ["image","rgb","depth"] else v for k,v in print_obs.items()}
            print_obs = [str(k)+ ":" +str(v) for k,v in print_obs.items()]
            print(" | ".join(print_obs))
        
            # print("reward:", reward, "done:", done, "info:", info, "step:", env.timestep, "obs_key:", obs.keys(), "fsm_state:", obs["fsm_state"])
            # print("observation space: ", env.observation_space)
            img = env.render(mode="DSA")
            # print(img.keys())
            # print(img)
            obs.update({"dsa": img["dsa"]})
            # print(action)
            
            if info['is_success']:
                break
            img_break = env.cv_show(imgs=obs)
            if img_break:
                break
        if img_break:
            break


if args.p == 8:
    from gym_ras.api import make_env
    from gym_ras.env.wrapper import Visualizer
    
    
    env, env_config = make_env(tags=args.env_tag, seed=args.seed)
    if env_config.embodied_name == "dVRKEnv":
        env =  Visualizer(env, update_hz=100)
    else:
        env =  Visualizer(env)
    for _ in tqdm(range(args.repeat)):
        done = False
        obs = env.reset()
        # print("obs:", obs)
        while not done:
            # action = env.action_space.sample()
            # print(action)
            print("==========step", env.timestep, "===================")
            if any(i.isdigit() for i in args.action):
                action = int(args.action)
            elif args.action == "random":
                action = env.action_space.sample()
            elif args.action == "oracle":
                action = env.get_oracle_action()
            else:
                raise NotImplementedError
            print("step....")
            obs, reward, done, info = env.step(action)
            print_obs = obs.copy()
            print_obs = {k: v.shape if k in ["image","rgb","depth"] else v for k,v in print_obs.items()}
            print_obs = [str(k)+ ":" +str(v) for k,v in print_obs.items()]
            print(" | ".join(print_obs))
        
            # print("reward:", reward, "done:", done, "info:", info, "step:", env.timestep, "obs_key:", obs.keys(), "fsm_state:", obs["fsm_state"])
            # print("observation space: ", env.observation_space)
            img = env.render()
            # print(img.keys())
            # print(img)
            if "dsa" in img:
                obs.update({"dsa": img["dsa"]})
            # print(action)
            
            if info['is_success']:
                break
            img_break = env.cv_show(imgs=img)
            if img_break:
                break
        if img_break:
            break
    del env
