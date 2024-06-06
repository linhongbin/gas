from gym_ras.tool.seg_tool import DetectronPredictor
from gym_ras.api import make_env
from gym_ras.tool.img_tool import CV2_Visualizer





visualizer = CV2_Visualizer()
env, env_config = make_env(tags=['no_wrapper'], seed=0)

_ = env.reset()
img = env.render()
visualizer.cv_show(img)

predictor = DetectronPredictor(model_dir="./data/segment/segment.pth",
                    cfg_dir='./data/segment/segment.yaml')
results = predictor.predict(img["rgb"])
print(results)
img.update({"mask": {str(k): v[0] for k,v in results[0].items()}})
visualizer.cv_show(img)

