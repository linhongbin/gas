import cv2
from pathlib import Path
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--video-dir', type=str, default="./data/surgical_video")
parser.add_argument('--output-dir', type=str, default="./data/surgical_image")
parser.add_argument('--sample-hz', type=int, default=1)
args = parser.parse_args()


ext = ["avi", "mp4"]
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
video_dir = []
for e in ext:
    video_dir.extend(sorted(Path(args.video_dir).glob("*."+e+"*")))

k = 1
for _dir in tqdm(video_dir):
    try:
        cap= cv2.VideoCapture(str(_dir))
        fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV v2.x used "CV_CAP_PROP_FPS"
        # frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        # duration = frame_count/fps
        i=0
        while(cap.isOpened()):
            ret, frame = cap.read()
            # print(fps/args.sample_hz)
            if ret == False:
                break
            if i % int(fps/args.sample_hz) == 0: # this is the line I added to make it only save one frame every 20
                cv2.imwrite(str(output_dir / (str(k) + ".jpg")),frame)
                k+=1
            i+=1
        cap.release()
    except Exception as e:
        print(f"{str(_dir)} got error: {e}")


cv2.destroyAllWindows()