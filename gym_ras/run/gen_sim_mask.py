import argparse
from pathlib import Path
from datetime import datetime
from gym_ras.api import make_env
from tqdm import tqdm
from gym_ras.tool.img_tool import save_img
from gym_ras.tool.seg_tool import get_coco_annot, get_coco_image, generate_coco_json, get_coco_category
from pycocotools.mask import encode
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--env-tag', type=str, nargs='+', default=[])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--savedir', type=str, default="./data/sim_mask")
parser.add_argument('--train-num', type=int, default=50000)
parser.add_argument('--eval-num', type=int, default=300)
args = parser.parse_args()

savedir =  Path(args.savedir) 
dir_name = str(datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))

env_tag= ["segment", "cam_pose_noise_medium"]
env_tag.extend(args.env_tag)
if len(env_tag)!=0:
    dir_name += "-" + "-".join(env_tag)
savedir = savedir / dir_name
# savedir.mkdir(parents=True, exist_ok=True)



env, env_config = make_env(tags=env_tag, seed=args.seed)




cul = 0

num = {"train": args.train_num, "eval": args.eval_num}
for data_type, data_num in num.items(): 
    annot_id = 1 + cul
    image_id = 1 + cul
    cul += data_num 
    if data_type == "train":
        continue
    annots = []
    images = []
    cats = []
    _ = env.reset()
    img = env.render()
    _mask = img["mask"]
    category_ids = {}
    j =0
    for k,v in _mask.items():
        j+=1
        category_ids[k] = j
        cats.append(get_coco_category(j, k,))

    _savedir = savedir / data_type
    image_dir = _savedir / "images"
    for i in tqdm(range(data_num)):
        _ = env.reset()
        img = env.render()
        img_shape = img["rgb"].shape
        _file_dir = save_img(img["rgb"],str(image_dir), str(image_id), img_format="png")
        images.append(get_coco_image(image_id, _file_dir, img_shape[0], img_shape[1],))

        _mask = img["mask"]
        for k,v in _mask.items():
            if np.sum(v)>0: # empty mask
                annots.append(get_coco_annot(v,annot_id,image_id,category_ids[k],iscrowd=False))
                annot_id+=1
        image_id+=1

    print("generate coco json...")
    generate_coco_json(images, annots, cats, str(_savedir), "annotation",)
    print(f"save to dir: {str(_savedir)}")