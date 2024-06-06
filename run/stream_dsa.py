from gym_ras.api import make_env
from gym_ras.env.wrapper import Visualizer

import argparse
from tqdm import tqdm
import time
from pathlib import Path
import cv2
parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

# parser.add_argument('--repeat',type=int, default=1)
# parser.add_argument('--action',type=str, default="3")
# parser.add_argument('--yaml-dir', type=str, default="./gym_ras/config.yaml")
# parser.add_argument('--yaml-tag', type=str, nargs='+', default=[])
parser.add_argument('--env-tag', type=str, nargs='+', default=[])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--vis-tag', type=str, nargs='+', default=[])
parser.add_argument('--save-dir', type=str, default="./data/stream_dsa/")
parser.add_argument('--save-tag', type=str, nargs='+', default=[])
args = parser.parse_args()

env, env_config = make_env(tags=args.env_tag, seed=args.seed)
env =  Visualizer(env, update_hz=-1, keyboard=True)
obs = env.reset()
while True:
    img = env.render()
    print(img.keys())
    if len(args.vis_tag) != 0:
        img = {k:v for k,v in img.items() if k in args.vis_tag}
    
    img_break = env.cv_show(imgs=img)
    if len(args.save_tag)!=0:
        _render_dir = Path(args.save_dir)
        _render_dir.mkdir(exist_ok=True, parents=True)
        for k in args.save_tag:
            file = time.strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(str(_render_dir / (file + '-' + k)) + '.png',
                        cv2.cvtColor(img[k], cv2.COLOR_RGB2BGR))
    if img_break:
        break
