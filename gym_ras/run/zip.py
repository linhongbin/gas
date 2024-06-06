import zipfile
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, default="./log")
parser.add_argument('--outdir', type=str, default="./log/zips")
parser.add_argument('--zip-model',action="store_true")
args = parser.parse_args()

file_string =["config.yaml", "metrics.jsonl", "tfevents"]
if args.zip_model:
    file_string = file_string + ["wm.pkl", "policy.pkl"]
outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

for root, dirs, files in os.walk(args.logdir):  
    for file in files:
        if file == "metrics.jsonl":
            _dir = Path(os.path.join(root, file)).parent
            method_name = _dir.name
            env_name = _dir.parent.name
            zip_dir = str(outdir / (method_name+"@"+env_name +'.zip'))
            print("zip to dir:", zip_dir)
            with zipfile.ZipFile(zip_dir, 'w') as f:
                for fname in os.listdir(_dir):
                    for key in file_string:
                        if fname.find(key)!=-1:
                            f.write(str(_dir/fname),fname)
