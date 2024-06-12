from gym_ras.tool.img_tool import CV2_Visualizer
from gym_ras.tool.rs435 import RS435_ROS_Engine
from gym_ras.tool.seg_tool import DetectronPredictor
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--savedir', type=str, default="./data/stream_cam")
parser.add_argument('--depth-c', type=float, default="0.3")
parser.add_argument('--depth-r', type=float, default="0.1")
parser.add_argument('--seg-dir', type=str, default="")
parser.add_argument('--seg-type', type=str, default="")
parser.add_argument('--vis-nodepth', type=str, default="")
parser.add_argument('--vis-tag', type=str, nargs='+', default=[])
args = parser.parse_args()


engine = RS435_ROS_Engine(depth_remap_center=None if args.depth_c <= 0 else args.depth_c,
                          depth_remap_range=None if args.depth_r <= 0 else args.depth_r)
visualizer = CV2_Visualizer(update_hz=10,
                            render_dir=args.savedir,
                            vis_tag=args.vis_tag,
                            keyboard=True)
is_seg = args.seg_type != ""
if is_seg:
    if args.seg_type == "detectron":
        _dir = Path(args.seg_dir)
        predictor = DetectronPredictor(model_dir=str(_dir / "model_best.pth"),
                                       cfg_dir=str(_dir / 'segment.yaml'))
    elif args.seg_type == "color":
        from gym_ras.tool.seg_tool import ColorObjSegmentor
        _dir = Path(args.seg_dir)

        predictor = ColorObjSegmentor(str(_dir))
else:
    predictor = None
# img = {"rgb":rgb, "depth": depth}
is_quit = False
while not is_quit:
    img = engine.get_image()
    if predictor is not None:
        masks = predictor.predict(img['rgb'])
        # print(masks)
        if len(masks) > 0:
            img.update({"mask": {str(k): v[0] for k, v in masks.items()}})
    # img.pop("depth", None)
    # img.pop("mask", None)
    visualizer.cv_show(img)
    is_quit = visualizer.cv_show(img)

    # print(results)
