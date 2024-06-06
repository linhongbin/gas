import pandas as pd 
import argparse
import os
from pathlib import Path
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, default="./log")
parser.add_argument('--step-min', type=int, default=2e4)
args = parser.parse_args()

file_string =["config.yaml", "wm.pkl", "policy.pkl", "metrics.jsonl", "tfevents"]

for root, dirs, files in os.walk(args.logdir): 
    for _dir in dirs:
        if _dir == "train_episodes":
            _root_dir = root
            _is_rm = False
            if not ("metrics.jsonl" in os.listdir(_root_dir)):
                _is_rm = True
            else:
                _file = "metrics.jsonl"
                df = pd.read_json(path_or_buf=os.path.join(_root_dir, _file), lines=True)
                if df["step"].iloc[-1] < args.step_min:
                    _is_rm = True
            if _is_rm:
                shutil.rmtree(root)
                print("delete dir:", root)
