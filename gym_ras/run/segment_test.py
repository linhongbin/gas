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
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger 
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
import cv2
import matplotlib.pyplot as plt
import random



parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, required=True)
parser.add_argument('--modeldir', type=str, required=True)
parser.add_argument('--vis-num', type=int, required=3)
args = parser.parse_args()

_data_dir = Path(args.datadir)

_model_dir = Path(args.modeldir)
for d in ["train", "eval"]:
    register_coco_instances(f"gym_ras_{d}", {}, str(_dir / d  / "annotation.json"), str(_dir / d / f"images"))

cfg.MODEL.WEIGHTS = str(_model_dir / "model_best.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
cfg.DATASETS.TEST = ("gym_ras_eval", )
predictor = DefaultPredictor(cfg)
from detectron2.utils.visualizer import ColorMode
dataset_dicts = DatasetCatalog.get("suture_test")
for d in random.sample(dataset_dicts, args.vis_num):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=microcontroller_metadata, 
                   scale=10, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize = (14, 10))
    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    plt.show()


#