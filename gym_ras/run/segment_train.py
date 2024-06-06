import argparse
from pathlib import Path
from datetime import datetime
import os




# import some common libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

# import some common detectron2 utilities
import detectron2
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor, DefaultTrainer, BestCheckpointer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger 
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
from detectron2.checkpoint import Checkpointer

setup_logger()
parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, required=True)
parser.add_argument('--worker', type=int, default=4)
parser.add_argument('--batch', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.00025)
parser.add_argument('--epoch', type=int, default=4e5)
parser.add_argument('--class-num', type=int, default=2)
parser.add_argument('--savedir', type=str, default="./data/segment_model")
parser.add_argument('--eval-every', type=int, default=1000)
args = parser.parse_args()



_dir = Path(args.datadir)
for d in ["train", "eval"]:
    register_coco_instances(f"gym_ras_{d}", {}, str(_dir / d  / "annotation.json"), str(_dir / d / f"images"))





savedir = Path(args.savedir) / str(datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
savedir.mkdir(parents=True, exist_ok=True)
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("gym_ras_train",)
cfg.DATASETS.TEST = ("gym_ras_eval",)
cfg.DATALOADER.NUM_WORKERS = args.worker
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = args.batch
cfg.SOLVER.BASE_LR = args.lr
cfg.SOLVER.MAX_ITER = int(args.epoch)
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.class_num
cfg.OUTPUT_DIR = str(savedir)
cfg.TEST.EVAL_PERIOD = int(args.eval_every)
# cfg.SOLVER.CHECKPOINT_PERIOD = 5000

dump_cfg = cfg.dump()
with open(str(savedir / "segment.yaml"), "w") as file:
    file.write(dump_cfg)


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        coco_evaluator = COCOEvaluator(dataset_name, output_dir=output_folder)
        
        evaluator_list = [coco_evaluator]
        
        return DatasetEvaluators(evaluator_list)

trainer = MyTrainer(cfg)
trainer.register_hooks([BestCheckpointer(eval_period=args.eval_every, 
                            checkpointer=Checkpointer(trainer.model, save_dir=str(savedir), save_to_disk=True),
                            val_metric="bbox/AP")])
trainer.resume_or_load(resume=False)
trainer.train()


# import torch
# torch.save(trainer.model.state_dict(), "./output/segment.pth")
