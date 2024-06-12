from PIL import Image
from pathlib import Path
import cv2
import time
import numpy as np
import sys


def save_img(img_arr, save_dir, save_file_name, img_format="png"):
    from pathlib import Path
    _dir = Path(save_dir)
    _dir.mkdir(parents=True, exist_ok=True)
    _dir = _dir / (save_file_name + '.' + img_format)
    im = Image.fromarray(img_arr)
    im.save(_dir)
    return str(_dir)


class CV2_Visualizer():
    """ render image with GUI """

    def __init__(self,
                 update_hz=-1,
                 render_dir="./data/cv_visualizer",
                 vis_channel=[0, 1, 2],
                 is_gray=False,
                 gui_shape=[600, 600],
                 cv_interpolate="area",
                 save_img_key=["rgb"],
                 vis_tag=[],
                 keyboard=False,
                 **kwargs,
                 ):
        self._update_hz = update_hz
        self._render_dir = Path(render_dir)
        self._render_dir = self._render_dir / time.strftime("%Y%m%d-%H%M%S")
        self._vis_channel = vis_channel
        self._is_gray = is_gray
        self._gui_shape = gui_shape
        self._cv_interpolate = cv_interpolate
        self._save_img_key = save_img_key
        self._vis_tag = vis_tag
        self.keyboard = keyboard

    def cv_show(self, imgs, gui_length=1080):

        title = self.__class__.__name__
        # cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.namedWindow(title, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(
            title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        is_quit = False
        if len(self._vis_tag) != 0:
            _imgs = {k: v for k, v in imgs.items() if k in self._vis_tag}
            imgs = _imgs
        while True:
            imgs_dict = {}
            for k, v in imgs.items():
                if k.find("rgb") >= 0:
                    imgs_dict[k] = v
                if k.find("image") >= 0:
                    imgs_dict[k] = v
                elif k.find("depth") >= 0:
                    imgs_dict[k] = np.stack(
                        [v]*3, axis=2) if len(v.shape) == 2 else v
                elif k.find("dsa") >= 0:
                    imgs_dict[k] = v
                    for i in range(v.shape[2]):
                        imgs_dict[k+"@" +
                                  str(i)] = np.stack([v[:, :, i]]*3, axis=2)
                elif k.find("mask") >= 0:
                    if len(v) == 0:
                        break

                    _mask = None
                    for _key, _value in dict(sorted(v.items())).items():
                        # print(_value.shape)
                        _mask = _value if _mask is None else np.logical_or(
                            _mask, _value)
                        _mat = np.zeros(_value.shape, dtype=np.uint8)
                        _mat[_value] = 255
                        imgs_dict[k+"@"+_key] = np.stack([_mat]*3, axis=2)
                    rgb_str = k.replace("mask", "rgb")
                    _background = imgs[rgb_str].copy()
                    _mask_rgb = imgs[rgb_str].copy()
                    _maskss = np.stack([np.bitwise_not(_mask)]*3, axis=2)
                    # print(_maskss.shape)
                    _mask_rgb[_maskss] = 0
                    c = 0.2
                    b = 1
                    imgs_dict[k+"@enhance"] = cv2.addWeighted(
                        _background, c, _mask_rgb, 1-c, b)
                # elif k == "mask":
                #     _img = v
                elif k.find("obs") >= 0:
                    # print(cv2.resize(v,
                    #     (imgs['rgb'].shape[0], imgs['rgb'].shape[1], ),
                    #     interpolation={"nearest": cv2.INTER_NEAREST,
                    #                     "linear": cv2.INTER_LINEAR,
                    #                     "area": cv2.INTER_AREA,
                    #                     "cubic": cv2.INTER_CUBIC,}[self._cv_interpolate]).shape)
                    imgs_dict[k] = cv2.resize(v,
                                              (imgs['rgb'].shape[0],
                                               imgs['rgb'].shape[1], ),
                                              interpolation={"nearest": cv2.INTER_NEAREST,
                                                             "linear": cv2.INTER_LINEAR,
                                                             "area": cv2.INTER_AREA,
                                                             "cubic": cv2.INTER_CUBIC, }[self._cv_interpolate])
                else:
                    continue
            if "rgb" in imgs_dict and "depth" in imgs_dict:
                c = 0.4
                b = 1
                imgs_dict["rgbd@enhance"] = cv2.addWeighted(
                    imgs_dict["rgb"], c, imgs_dict["depth"], 1-c, b)
            if "rgb" in imgs and "depth" in imgs and "mask" in imgs:
                for _key, _value in dict(sorted(imgs['mask'].items())).items():
                    _depth = imgs["depth"]
                    _mat = np.zeros(_depth.shape, dtype=np.uint8)
                    _mat[_value] = _depth[_value]
                    imgs_dict["depth@"+_key] = np.stack([_mat]*3, axis=2)

            # names = []
            frame = None
            cnt = 0
            pic_length = int(gui_length / len(imgs_dict))
            pic_num = len(imgs_dict)
            colum = np.int(np.ceil(np.sqrt(pic_num)))
            total_num = np.int(colum*colum)
            for j in range((total_num-pic_num) % colum):
                imgs_dict[j] = np.zeros((4, 4, 3), dtype=np.uint8)
            img_list = list(imgs_dict.items())
            id_shape = np.array(list(range(len(imgs_dict)))).reshape(-1, colum)

            frames = []
            for j in range(id_shape.shape[0]):
                frame = None
                for i in range(id_shape.shape[1]):
                    k = img_list[id_shape[j][i]][0]
                    v = img_list[id_shape[j][i]][1]
                    gui_width = int(pic_length * v.shape[0] / v.shape[1])
                    # print((pic_length, gui_width, ))

                    _v = cv2.resize(v,
                                    (pic_length, gui_width, ),
                                    interpolation={"nearest": cv2.INTER_NEAREST,
                                                   "linear": cv2.INTER_LINEAR,
                                                   "area": cv2.INTER_AREA,
                                                   "cubic": cv2.INTER_CUBIC, }[self._cv_interpolate])
                    # 图片 添加的文字 位置 字体 字体大小 字体颜色 字体粗细
                    cv2.putText(_v, str(k), (20, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 204, 0), 1)
                    # print(_v.shape)
                    frame = _v if frame is None else np.concatenate(
                        (frame, _v), axis=1)
                frames.append(frame)

            frame = np.concatenate(tuple(frames), axis=0)

            # Display the resulting frame
            # print(frame.shape)
            cv2.imshow(title,
                       cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            _t = 0 if self._update_hz < 0 else np.int32(
                (1/self._update_hz)*1000)

            is_quit = False
            if self.keyboard:
                k = cv2.waitKey(_t)

                if k & 0xFF == ord('q'):    # q key to exit
                    print("press q ..")
                    is_quit = True
                    sys.exit(0)
                elif k & 0xFF == ord('s'):    # s key to save pic
                    for _key in self._save_img_key:
                        self._render_dir.mkdir(exist_ok=True, parents=True)
                        file = time.strftime("%Y%m%d-%H%M%S")
                        cv2.imwrite(str(self._render_dir / (file + '-' + _key)) + '.png',
                                    cv2.cvtColor(imgs[_key], cv2.COLOR_RGB2BGR))
                elif k & 0xFF == ord('g'):    # g key to toggle gray
                    self._is_gray = not self._is_gray
                else:
                    # print("")
                    break
            else:
                _q = cv2.waitKey(1)
                break
        return is_quit

    def save_img(self, img, save_dir):
        _dir = Path(save_dir)
        _dir.mkdir(exist_ok=True, parents=True)
        file = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(str(_dir / (file)) + '.png',
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def __del__(self):
        cv2.destroyAllWindows()
