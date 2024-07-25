"""
refer to domain randomization tech: https://arxiv.org/pdf/2208.04171.pdf
"""

from gym_ras.env.wrapper.base import BaseWrapper
import numpy as np
import gym
import cv2
from copy import deepcopy


class ImageNoise(BaseWrapper):
    def __init__(self, env,
                 skip=True,
                 uniform_noise_range=3,
                 pns_noise_amount=0.05,
                 pns_noise_balance=0.5,
                 gaussian_blur_kernel=3,
                 gaussian_blur_sigma=0.8,
                 cutout_circle_r_low=0.01,
                 cutout_circle_r_high=0.2,
                 cutout_circle_num_low=2,
                 cutout_circle_num_high=8,
                 cutout_rec_w_low=0.1,
                 cutout_rec_w_high=0.2,
                 cutout_rec_h_low=0.1,
                 cutout_rec_h_high=0.2,
                 cutout_rec_num_low=3,
                 cutout_rec_num_high=10,
                 cutout_line_w_low=0.5,
                 cutout_line_w_high=1,
                 cutout_line_h_low=0.9,
                 cutout_line_h_high=1,
                 cutout_line_num_low=3,
                 cutout_line_num_high=10,
                 cutout_all_amount_range=[0.8, 1],
                 cutout_depth_amount_range=[0.5, 1],
                 max_loop=3,
                 image_key=["depth"],
                 **kwargs):
        super().__init__(env)
        self._skip = skip
        self._pns_noise_amount = pns_noise_amount
        self._pns_noise_balance = pns_noise_balance
        self._gaussian_blur_kernel = gaussian_blur_kernel
        self._gaussian_blur_sigma = gaussian_blur_sigma
        self._image_noise_rng = np.random.RandomState(0)
        self._circle = {}
        self._rec = {}
        self._line = {}
        self._circle["radius"] = [cutout_circle_r_low, cutout_circle_r_high]
        self._circle["num"] = [cutout_circle_num_low, cutout_circle_num_high]
        self._rec["width"] = [cutout_rec_w_low, cutout_rec_w_high]
        self._rec["height"] = [cutout_rec_h_low, cutout_rec_h_high]
        self._rec["num"] = [cutout_rec_num_low, cutout_rec_num_high]

        self._line["width"] = [cutout_line_w_low, cutout_line_w_high]
        self._line["height"] = [cutout_line_h_low, cutout_line_h_high]
        self._line["num"] = [cutout_line_num_low, cutout_line_num_high]
        self._cutout_all_amount_range = cutout_all_amount_range
        self._cutout_depth_amount_range = cutout_depth_amount_range
        self._max_loop = max_loop

        self._uniform_noise_range = uniform_noise_range
        self._image_key = image_key

    def render(self,):
        img = self.env.render()
        if not self._skip:
            if self.unwrapped._dr_scale[self.unwrapped.mode] > 0.5:
                img = self._post_process(img)
        return img

    def _post_process(self, img):
        for k in self._image_key:
            img[k] = self._add_pepper_and_salt_nosie(img[k])
        for k in self._image_key:
            img[k] = self._add_uniform_noise(img[k], self._uniform_noise_range)

        for k in self._image_key:
            img[k] = self._add_gaussian_blur(img[k])

        if self._cutout_all_amount_range[0] < 1:
            img = self._cutout_protocol(
                img, mask_min=self._cutout_all_amount_range[0], mask_max=self._cutout_all_amount_range[1], only_depth=False)
        if self._cutout_depth_amount_range[0] < 1:
            img = self._cutout_protocol(
                img, mask_min=self._cutout_depth_amount_range[0], mask_max=self._cutout_depth_amount_range[1], only_depth=True)

        return img

    def _cutout_protocol(self, img, mask_min, mask_max=0.9, only_depth=False):

        return_img = deepcopy(img)
        for loop in range(self._max_loop):
            _img = deepcopy(img)
            mask_amount = {k: np.sum(v) for k, v in img["mask"].items()}

            # cirlce cutout
            _num = {}
            _num["circle"] = self._image_noise_rng.randint(low=self._circle["num"][0],
                                                           high=self._circle["num"][1]+1,)
            _num["rectangle"] = self._image_noise_rng.randint(low=self._circle["num"][0],
                                                              high=self._rec["num"][1]+1,)
            _num["line"] = self._image_noise_rng.randint(low=self._line["num"][0],
                                                         high=self._line["num"][1]+1,)

            for k, v in _num.items():
                for _ in range(v):
                    _img = self._cutout(_img, cutout_type=k,
                                        only_depth=only_depth)

            # # retangle cutout
            # _num = self._image_noise_rng.randint(low=self._cutout_circle["num_low"],
            #                                      high=self._cutout_circle["num_high"]+1,)
            # for _ in range(_num):
            #     _img = self._cutout(_img, cutout_type="rectangle")

            amount_check = True
            for k, v in _img["mask"].items():
                _amount = np.sum(v)
                if (_amount < mask_amount[k]*mask_min) or (_amount > mask_amount[k]*mask_max):
                    amount_check = False
            if amount_check:
                return_img = deepcopy(_img)
                if only_depth:
                    return_img["mask"] = img["mask"]

        return return_img

    def _add_uniform_noise(self, img, range_px):
        _range_px = range_px // 2
        noise = np.random.uniform(-_range_px, _range_px, img.shape)
        img = np.clip(img+noise, 0, 255)
        return np.uint8(img)

    def _add_gaussian_blur(self, img):
        img = deepcopy(img)
        return cv2.GaussianBlur(img, (self._gaussian_blur_kernel, self._gaussian_blur_kernel), self._gaussian_blur_sigma, self._gaussian_blur_sigma)

    def _add_pepper_and_salt_nosie(self, image):
        image = deepcopy(image)
        s_vs_p = self._pns_noise_balance
        amount = self._pns_noise_amount
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        _out = out.reshape(-1)
        # coords = [np.random.randint(0, i - 1, int(2)).tolist() for i in image.shape]
        coords = np.random.randint(0, int(image.size) - 1, int(num_salt))
        _out[coords] = 1
        out = _out.reshape(image.shape)

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        _out = out.reshape(-1)
        # coords = [np.random.randint(0, i - 1, int(2)).tolist() for i in image.shape]
        coords = np.random.randint(0, int(image.size) - 1, int(num_pepper))
        _out[coords] = 0
        out = _out.reshape(image.shape)
        return out

    def _draw_rec(self, image, cx, cy, width, height, angle, rgb_list):
        rot_rectangle = ((np.int(cy*image.shape[1]), np.int(cx*image.shape[0]), ),
                         (np.int(height*image.shape[1]), np.int(width*image.shape[0])), np.int(angle*180))
        box = cv2.boxPoints(rot_rectangle)
        box = np.int0(box)  # Convert into integer values
        rgb_list = np.uint8(np.array(rgb_list)*255).tolist()
        rgb_list.reverse()
        image = cv2.drawContours(image, [box], contourIdx=0, color=tuple(
            rgb_list), thickness=cv2.FILLED)
        return image

    def _draw_line(self, image, cx, cy, width, height, angle, rgb_list, ratio_w2h=0.01):
        self._draw_rec(image, cx, cy, width*ratio_w2h, height, angle, rgb_list)
        return image

    def _draw_circle(self, image, cx, cy, radius, rgb_list):
        cx = np.int(cx*image.shape[0])
        cy = np.int(cy*image.shape[1])
        radius = np.int(min(image.shape[0], image.shape[1])*radius/2)
        rgb_list = np.uint8(np.array(rgb_list)*255).tolist()
        rgb_list.reverse()
        image = cv2.circle(image, (cy, cx), radius,
                           color=tuple(rgb_list), thickness=cv2.FILLED)
        return image

    def _cutout(self, img, cutout_type="circle", only_depth=False):
        img = deepcopy(img)
        w, h, c = img['rgb'].shape

        args = {}
        args["cx"] = self._image_noise_rng.uniform(low=0.0, high=1.0,)
        args["cy"] = self._image_noise_rng.uniform(low=0.0, high=1.0,)
        args["rgb_list"] = self._image_noise_rng.uniform(
            low=0.0, high=1.0, size=3,).tolist()
        if cutout_type == "circle":
            args["radius"] = self._image_noise_rng.uniform(
                low=self._circle["radius"][0], high=self._circle["radius"][1])
            _call = getattr(self, "_draw_circle")

        elif cutout_type == "rectangle":
            args["width"] = self._image_noise_rng.uniform(
                low=self._rec["width"][0], high=self._rec["width"][1])
            args["height"] = self._image_noise_rng.uniform(
                low=self._rec["height"][0], high=self._rec["height"][1])
            args["angle"] = self._image_noise_rng.uniform(low=0, high=1)
            _call = getattr(self, "_draw_rec")
        elif cutout_type == "line":
            args["width"] = self._image_noise_rng.uniform(
                low=self._line["width"][0], high=self._line["width"][1])
            args["height"] = self._image_noise_rng.uniform(
                low=self._line["height"][0], high=self._line["height"][1])
            args["angle"] = self._image_noise_rng.uniform(low=0, high=1)
            _call = getattr(self, "_draw_line")

        if only_depth:
            args["rgb_list"] = [0, 0, 0]
            img["depth"] = _call(image=img["depth"], **args)
        else:
            for k in self._image_key:
                img[k] = _call(image=img[k], **args)

        _mask_dict = {}
        args.update({"rgb_list": [0, 0, 0]})
        for k, v in img["mask"].items():
            _bool_mat = v.copy()
            _mat = np.zeros(_bool_mat.shape, dtype=np.uint8)
            _mat[_bool_mat] = 1

            _mat = _call(image=_mat, **args)
            _mat = _mat != 0
            _mask_dict[k] = _mat
        img.update({"mask": _mask_dict})
        return img

    def _init_rng(self, seed):
        image_noise_seed = np.uint32(seed)
        print("image_noise_seed:", image_noise_seed)
        self._image_noise_rng = np.random.RandomState(image_noise_seed)
        if self.is_wrapper:
            self.env._init_rng(self.env.seed)
