from gym_ras.api import make_env
from gym_ras.env.wrapper import Visualizer
from gym_ras.tool.img_tool import CV2_Visualizer
import argparse
from tqdm import tqdm
import time
from gym_ras.tool.keyboard import Keyboard
import sys       
parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('--env-tag', type=str, nargs='+', default=["dvrk_real_dsa","dvrk_cam_setting","color_seg", "dvrk_full_ws","no_wrapper"])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--vis-tag', type=str, nargs='+', default=["rgb"])
parser.add_argument('--repeat', type=int, default=5)
parser.add_argument('--num', type=int, default=10)
parser.add_argument('--dir', type=str, default="./data/collect_seg_rgb/")
parser.add_argument('--method', type=str, default="auto")

args = parser.parse_args()

env, env_config = make_env(tags=args.env_tag, seed=args.seed)
visualizer = CV2_Visualizer(update_hz=100,
                            vis_tag=args.vis_tag,
                            keyboard=False)
keyboard = Keyboard()
def get_char():
    c = keyboard.get_char()
    if c=="q":
        sys.exit(0)
for i in tqdm(range(args.repeat if args.method=="auto" else 1)):
    get_char()

    for i in tqdm(range(args.num)):
        if args.method == "auto":
            obs = env.reset()
            time.sleep(1)
        elif args.method == "manual":
            get_char()
        else:
            raise NotImplementedError()
        for i in range(3):
            img = env.render()
        if len(args.vis_tag) != 0:
            img = {k:v for k,v in img.items() if k in args.vis_tag}
        _ = visualizer.cv_show(imgs=img)
        visualizer.save_img(img['rgb'], args.dir)
