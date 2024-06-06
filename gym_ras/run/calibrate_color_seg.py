#refer: https://blog.csdn.net/weixin_42216109/article/details/89520423

import cv2
import numpy as np
import argparse
from pathlib import Path
from gym_ras.tool.seg_tool import ColorSegmentor
from gym_ras.tool.config import Config
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default="")
parser.add_argument('--savedir', type=str, default='./data/dvrk_cal')
parser.add_argument('--hz', type=int, default='30')
args = parser.parse_args()

object_id_map = {"needle":0, "gripper_tip":1,"gripper_base":2, "gripper_link":3,}
display_hz = args.hz
#定义窗口名称
winName='Colors of the rainbow'
#定义滑动条回调函数，此处pass用作占位语句保持程序结构的完整性
def nothing(x):
    pass

if args.file == "":
    from gym_ras.tool.rs435 import RS435_ROS_Engine
    engine = RS435_ROS_Engine()
else:
    engine = None
    img_original=cv2.imread(args.file)
    img_original=cv2.cvtColor(img_original,cv2.COLOR_BGR2RGB) 
#颜色空间的转换


#新建窗口
cv2.namedWindow(winName)
#新建6个滑动条，表示颜色范围的上下边界，这里滑动条的初始化位置即为黄色的颜色范围
cv2.createTrackbar('h_low',winName,0,255,nothing)
cv2.createTrackbar('h_high',winName,255,255,nothing)
cv2.createTrackbar('h2_low',winName,255,255,nothing)
cv2.createTrackbar('h2_high',winName,255,255,nothing)
cv2.createTrackbar('s_low',winName,0,255,nothing)
cv2.createTrackbar('s_high',winName,255,255,nothing)
cv2.createTrackbar('v_low',winName,0,255,nothing)
cv2.createTrackbar('v_high',winName,255,255,nothing)
cv2.createTrackbar('region_mask_size',winName,100,100,nothing)

segmentor = ColorSegmentor()

yaml_dict ={}
for key, mask_id in object_id_map.items():
    cv2.setTrackbarPos('h_low',winName,0)
    cv2.setTrackbarPos('h_high',winName,255)
    cv2.setTrackbarPos('h2_low',winName,255)
    cv2.setTrackbarPos('h2_high',winName,255)
    cv2.setTrackbarPos('s_low',winName,0)
    cv2.setTrackbarPos('s_high',winName,255)
    cv2.setTrackbarPos('v_low',winName,0)
    cv2.setTrackbarPos('v_high',winName,255)
    cv2.setTrackbarPos('region_mask_size',winName,100)
    while(1):
        #函数cv2.getTrackbarPos()范围当前滑块对应的值
        if engine is not None:
            img = engine.get_image()
            img_original = img["rgb"]

        h_low=cv2.getTrackbarPos('h_low',winName)
        h_high=cv2.getTrackbarPos('h_high',winName)
        h2_low=cv2.getTrackbarPos('h2_low',winName)
        h2_high=cv2.getTrackbarPos('h2_high',winName)
        s_low=cv2.getTrackbarPos('s_low',winName)
        s_high=cv2.getTrackbarPos('s_high',winName)
        v_low=cv2.getTrackbarPos('v_low',winName)
        v_high=cv2.getTrackbarPos('v_high',winName)
        region_mask_size=cv2.getTrackbarPos('region_mask_size',winName)

        #得到目标颜色的二值图像，用作cv2.bitwise_and()的掩模
        # img_target=cv2.inRange(img_original,(136, 0, 0), (255, 255, 255))
        # img_hsv=cv2.cvtColor(img_original,cv2.COLOR_RGB2HSV) 
        # mask=cv2.inRange(img_hsv,(h_low,s_low,v_low),(h_high,s_high,v_high))
        # print(img_target)
        #输入图像与输入图像在掩模条件下按位与，得到掩模范围内的原图像
        param = {
        "hsv_h":[h_low, h_high],
        "hsv_h2":[h2_low, h2_high],
        "hsv_s":[s_low, s_high],
        "hsv_v": [v_low, v_high],
        "region_mask_size": region_mask_size / 100,
        "mask_id":mask_id,
            }
        segmentor.set_param(**param)
        mask = segmentor.predict(img_original)
        mask = mask[0]
        img_render = img_original.copy()
        img_render[np.logical_not(mask)] = 0
        # img_specifiedColor=cv2.bitwise_and(img_hsv,img_hsv,mask=img_target)
        # if is_mask:
        #     img_specifiedColor = img_target
        cv2.setWindowTitle(winName, key )
        cv2.imshow(winName, cv2.cvtColor(img_render, cv2.COLOR_RGB2BGR),)
        delay_t = 1000 // display_hz
        if cv2.waitKey(delay_t)==ord('q'):
            sys.exit(0)
        elif cv2.waitKey(delay_t)==ord('c'):
            yaml_dict[key] = segmentor.param.copy()
            break
        # elif cv2.waitKey(delay_t)==ord('m'):
        #     is_mask = not is_mask


savedir = Path(args.savedir)
savedir.mkdir(parents=True, exist_ok=True)
# print(yaml_dict)
config = Config(yaml_dict)
# with open(str(savedir / 'color_seg_cal.yaml'), 'w') as outfile:
config.save(str(savedir / 'color_seg_cal.yaml'))
cv2.destroyAllWindows()