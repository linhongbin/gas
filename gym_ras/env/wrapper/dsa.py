from gym_ras.env.wrapper.base import BaseWrapper
import numpy as np
import cv2


class DSA(BaseWrapper):
    """ timelimit in a rollout """

    def __init__(self,
                 env,
                 zoom_margin_ratio=0.3,
                 zoom_box_obj="psm1",
                 encode_type="general_simple",
                 cv_interpolate="area",
                 zoom_box_fix_length_ratio=0.5,
                 encoding_id_stuff=50,
                 encoding_id_psm1=100,
                 encoding_id_zoom_box=200,
                 zoom_movement_type="continuous",
                 dsa_key=["psm1", "stuff"],
                 out_reward_xy=1e-8,
                 out_reward_z=5e-6,
                 dense_reward=False,
                 **kwargs
                 ):
        super().__init__(env,)
        self._zoom_margin_ratio = zoom_margin_ratio
        self._zoom_box_obj = zoom_box_obj
        self._encode_type = encode_type
        self._cv_interpolate = cv_interpolate
        self._zoom_box_fix_length_ratio = zoom_box_fix_length_ratio
        self._image_encode_id = {"stuff": encoding_id_stuff,
                                 "psm1": encoding_id_psm1, "zoom_box": encoding_id_zoom_box}
        self._zoom_movement_type = zoom_movement_type
        self._dsa_key = dsa_key
        self._out_reward_xy = out_reward_xy
        self._out_reward_z = out_reward_z
        self._dense_reward = dense_reward
        self._reset_vars()

    # def step(self, action):
    #     obs, reward, done, info = self.env.step(action)
    #     if self.is_out_dsa_zoom and info["fsm"] != "done_success":
    #         info["fsm"] = "prog_abnorm_3" # cover the fsm state

    #     return obs, reward, done, info

    # def _mask_or(self, mask, ids):
    #     x = None
    #     for _id in ids:
    #         out = mask == _id
    #         x = out if x is None else x | out
    #     return x
    def render(self, ):
        img = self.env.render()
        req_key = ["rgb", "mask", ]
        pfix = ""
        dsa_list = []
        if all([k+pfix in img for k in req_key]):
            dsa_list.append(self.dsa_process(img, pfix))
        pfix = "_2"
        if all([k+pfix in img for k in req_key]):
            dsa_list.append(self.dsa_process(img, pfix))

        if len(dsa_list) == 1:
            img["dsa"] = dsa_list[0]
        if len(dsa_list) == 2:
            layer2 = dsa_list[0][:, :, 2]
            layer3 = dsa_list[1][:, :, 2]
            _dsas = []
            _l = int(dsa_list[0][:, :, 0].shape[0] / 2)
            for i in range(2):
                _mat = np.zeros((_l, _l,), dtype=np.uint8)
                _mat = np.concatenate((_mat, cv2.resize(dsa_list[i][:, :, 0], dsize=(_l, _l,),
                                                        interpolation={"nearest": cv2.INTER_NEAREST,
                                                                       "linear": cv2.INTER_LINEAR,
                                                                       "area": cv2.INTER_AREA,
                                                                       "cubic": cv2.INTER_CUBIC, }[self._cv_interpolate])), axis=0
                                      )
                _dsas.append(_mat)

            layer1 = np.concatenate(tuple(_dsas), axis=1)
            img["dsa"] = np.stack([layer1, layer2, layer3], axis=2)

        return img

    def dsa_process(self, img, pfix):
        self._out_zoom = False
        if self._encode_type == "raw":
            _mask_mat = np.zeros(img["depth"+pfix].shape)
            for k, v in img["mask"+pfix].items():
                if k in self._dsa_key:
                    _mask_mat[v] = self._image_encode_id[k]
            img_proc = img["rgb"+pfix].copy()
            img_proc[:, :, 0] = _mask_mat
            img_proc[:, :, 1] = np.mean(img["rgb"+pfix], axis=2)
            img_proc[:, :, 2] = img["depth"+pfix]
            img_proc = np.clip(img_proc, 0, 255)
            return np.uint8(img_proc)
        img_dsa = np.zeros(img["rgb"+pfix].shape, np.uint8)
        assert "mask" in img
        if len(img["mask"]) <= 0:
            return img_dsa
        if self._zoom_box_obj not in img["mask"+pfix]:
            return img_dsa
        zoom_mask = img["mask"+pfix][self._zoom_box_obj]

        if not np.any(zoom_mask):  # all mask have to be no zero
            return img_dsa

        zoom_box_x, zoom_box_y, _, _ = self._get_box_from_masks(zoom_mask)
        if self._encode_type == "IROS2023":
            zoom_x_min, zoom_x_max, zoom_y_min, zoom_y_max = zoom_box_x[
                0], zoom_box_x[1], zoom_box_y[0], zoom_box_y[1]
            gripper_mat = self._segment(img["depth"+pfix], gripper_mask)
            _zoom_stuff_mat, _x_stuff, _y_stuff = self._get_zoom_mat(
                stuff_mask, img["depth"+pfix], zoom_box_x, zoom_box_y, margin_ratio=self._zoom_margin_ratio)
            _zoom_gripper_mat, _, _ = self._get_zoom_mat(
                gripper_mask, img["depth"+pfix], zoom_box_x, zoom_box_y, margin_ratio=self._zoom_margin_ratio)
            gripper_mat[_x_stuff[0]:_x_stuff[1],
                        _y_stuff[0]:_y_stuff[1]] = 255
            segs = [gripper_mat, _zoom_stuff_mat, _zoom_gripper_mat]
            # print(np.sum(_zoom_gripper_mat))
            im_pre = np.stack(segs, axis=2)
            img_dsa = im_pre

        elif self._encode_type in ["general"]:
            layer1 = self._segment(
                img["depth"+pfix], gripper_mask | stuff_mask)
            zoom_box_length = self._zoom_box_fix_length_ratio * \
                min(zoom_mask.shape[0], zoom_mask.shape[1])
            # zoom_x_min, zoom_x_max = self._get_legal_min_max_continuous(zoom_box_x[0],zoom_box_x[1], zoom_box_length, zoom_mask.shape[0])
            # zoom_y_min, zoom_y_max = self._get_legal_min_max_continuous(zoom_box_y[0],zoom_box_y[1], zoom_box_length, zoom_mask.shape[1])
            zoom_x_min, zoom_x_max, zoom_y_min, zoom_y_max = self._get_zoom_coordinate(
                zoom_box_x, zoom_box_y, zoom_box_length, zoom_mask.shape)

            layer2 = np.zeros(img["depth"+pfix].shape, dtype=np.uint8)
            _mat = np.zeros(img["depth"+pfix].shape, dtype=np.uint8)
            _mat[stuff_mask] = self._image_encode_id["stuff"]
            layer2 += _mat
            _mat = np.zeros(img["depth"+pfix].shape, dtype=np.uint8)
            _mat[gripper_mask] = self._image_encode_id["psm1"]
            layer2 += _mat
            _mat = np.zeros(img["depth"+pfix].shape, dtype=np.uint8)
            _mat[zoom_x_min:zoom_x_max+1,
                 zoom_y_min:zoom_y_max+1] = self._image_encode_id["zoom_box"]
            layer2 += _mat
            # layer2 = self._zoom_legal(layer1, zoom_x_min, zoom_x_max,zoom_y_min, zoom_y_max)

            gripper_ecode = self._segment(self._scale_arr(
                img["depth"+pfix], 0, 255, 0, 127), gripper_mask)
            stuff_ecode = self._segment(self._scale_arr(
                img["depth"+pfix], 0, 255, 128, 255), stuff_mask)
            encode_mat = np.uint8(gripper_ecode + stuff_ecode)
            layer3 = self._zoom_legal(
                encode_mat, zoom_x_min, zoom_x_max, zoom_y_min, zoom_y_max)

            im_pre = np.stack([layer1, layer2, layer3], axis=2)
            img_dsa = im_pre
        elif self._encode_type in ["general_simple", "general_simple2", "general_simple3"]:
            zoom_box_length = self._zoom_box_fix_length_ratio * \
                min(zoom_mask.shape[0], zoom_mask.shape[1])
            # zoom_x_min, zoom_x_max = self._get_legal_min_max_continuous(zoom_box_x[0],zoom_box_x[1], zoom_box_length, zoom_mask.shape[0])
            # zoom_y_min, zoom_y_max = self._get_legal_min_max_continuous(zoom_box_y[0],zoom_box_y[1], zoom_box_length, zoom_mask.shape[1])
            zoom_x_min, zoom_x_max, zoom_y_min, zoom_y_max = self._get_zoom_coordinate(
                zoom_box_x, zoom_box_y, zoom_box_length, zoom_mask.shape)
            layer1 = np.zeros(img["rgb"+pfix].shape[0:2], dtype=np.uint8)
            _union_mask = None
            for k, v in img["mask"+pfix].items():
                if k in self._dsa_key:
                    _box_r, _box_c, _, _ = self._get_box_from_masks(v)
                    if self._encode_type == 'general_simple3':
                        _ratio = 0.5
                        _l = int(zoom_box_length * _ratio)
                        # print("length", zoom_box_length, _l)
                        _box_r = list(_box_r)
                        _box_c = list(_box_c)
                        _box_r[0], _box_r[1] = self._get_legal_min_max_continuous(
                            _box_r[0], _box_r[1], _l, zoom_mask.shape[0])
                        _box_c[0], _box_c[1] = self._get_legal_min_max_continuous(
                            _box_c[0], _box_c[1], _l, zoom_mask.shape[1])
                    if k == "stuff":
                        stuff_x = _box_r
                        stuff_y = _box_c
                    layer1[_box_r[0]:_box_r[1]+1, _box_c[0]                           :_box_c[1]+1] += self._image_encode_id[k]
                    _union_mask = (
                        _union_mask | v) if _union_mask is not None else v

            self._is_out_dsa_zoom([zoom_x_min, zoom_x_max], [
                                  zoom_y_min, zoom_y_max], stuff_x, stuff_y)

            if self._encode_type == 'general_simple':
                layer1[zoom_x_min:zoom_x_max+1, zoom_y_min:zoom_y_max +
                       1] += self._image_encode_id["zoom_box"]
            elif self._encode_type in ['general_simple2', 'general_simple3',]:
                layer1[zoom_x_min:zoom_x_max+1, zoom_y_min:zoom_y_max +
                       1] = self._image_encode_id["zoom_box"]
            if "depth" in img:
                depth_seg_mat = self._segment(img["depth"+pfix], _union_mask)
                layer2 = self._zoom_legal(
                    depth_seg_mat, zoom_x_min, zoom_x_max, zoom_y_min, zoom_y_max)
            else:
                layer2 = np.zeros(img["rgb"+pfix].shape[0:2], dtype=np.uint8)

            _mask_mat = np.zeros(img["rgb"+pfix].shape[0:2], dtype=np.uint8)
            for k, v in img["mask"+pfix].items():
                if k in self._dsa_key:
                    _mask_mat[v] = self._image_encode_id[k]
            layer3 = self._zoom_legal(
                _mask_mat, zoom_x_min, zoom_x_max, zoom_y_min, zoom_y_max)
            im_pre = np.stack([layer1, layer2, layer3], axis=2)
            img_dsa = im_pre
        elif self._encode_type in ["decompose"]:
            zoom_box_length = self._zoom_box_fix_length_ratio * min(
                zoom_mask.shape[0], zoom_mask.shape[1]
            )
            zoom_x_min, zoom_x_max, zoom_y_min, zoom_y_max = self._get_zoom_coordinate(
                zoom_box_x, zoom_box_y, zoom_box_length, zoom_mask.shape
            )
            _union_mask = None
            for k, v in img["mask"+pfix].items():
                if k in self._dsa_key:
                    _box_r, _box_c, _, _ = self._get_box_from_masks(v)
                    _ratio = 0.5
                    _l = int(zoom_box_length * _ratio)
                    # print("length", zoom_box_length, _l)
                    _box_r = list(_box_r)
                    _box_c = list(_box_c)
                    _box_r[0], _box_r[1] = self._get_legal_min_max_continuous(
                        _box_r[0], _box_r[1], _l, zoom_mask.shape[0])
                    _box_c[0], _box_c[1] = self._get_legal_min_max_continuous(
                        _box_c[0], _box_c[1], _l, zoom_mask.shape[1])
                    if k == "stuff":
                        stuff_x = _box_r
                        stuff_y = _box_c
                    _union_mask = (
                        _union_mask | v) if _union_mask is not None else v
            zoom_cx = (zoom_x_min + zoom_x_max)//2
            zoom_cy = (zoom_y_min + zoom_y_max)//2
            zoom_cz = np.median(img["depth"+pfix][img["mask"+pfix][self._zoom_box_obj]])
            stuff_cx = (stuff_x[0] + stuff_x[1]) // 2
            stuff_cy = (stuff_y[0] + stuff_y[1]) // 2
            stuff_cz = np.median(img["depth"+pfix][img["mask"+pfix]["stuff"]])
            is_out = self._zoom2stuff(
                zoom_cx,
                zoom_cy,
                zoom_cz,
                stuff_cx,
                stuff_cy,
                stuff_cz,
                img["depth"+pfix].shape[0],
                img["depth"+pfix].shape[1],
                self._zoom_box_fix_length_ratio,
            )
            layer1 = np.zeros(img["rgb" + pfix].shape[0:2], dtype=np.uint8)
            layer2 = np.zeros(img["rgb"+pfix].shape[0:2], dtype=np.uint8)
            layer3 = np.zeros(img["rgb" + pfix].shape[0:2], dtype=np.uint8)
            if is_out:
                _w = img["rgb" + pfix].shape[0] 
                _h = img["rgb" + pfix].shape[1] 
                _k = int(_w * self._zoom_box_fix_length_ratio // 2)
                c = lambda i, min, max: np.clip(i, min, max)
                # print(_k)
                layer2[
                    c(zoom_cx - _k, 0, _w - 1) : c(zoom_cx + _k, 0, _w - 1),
                    c(zoom_cy - _k, 0, _h - 1) : c(zoom_cy + _k, 0, _h - 1),
                ] = self._image_encode_id["zoom_box"]
                layer2[
                    c(stuff_cx - _k, 0, _w - 1) : c(stuff_cx + _k, 0, _w - 1),
                    c(stuff_cy - _k, 0, _h - 1) : c(stuff_cy + _k, 0, _h - 1),
                ] = self._image_encode_id["stuff"]
                layer3[c(zoom_cx-_k, 0, _w-1) : c(zoom_cx+_k, 0, _w-1), 
                       c(zoom_cy-_k, 0, _h-1) : c(zoom_cy+_k, 0, _h-1)] = zoom_cz
                layer3[c(stuff_cx-_k, 0, _w-1) : c(stuff_cx+_k, 0, _w-1), 
                       c(stuff_cy-_k, 0, _h-1) : c(stuff_cy+_k, 0, _h-1)] = stuff_cz
            else:
                depth_seg_mat = self._segment(img["depth"+pfix], _union_mask)
                layer2 = self._zoom_legal(
                    depth_seg_mat, zoom_x_min, zoom_x_max, zoom_y_min, zoom_y_max)
                _mask_mat = np.zeros(img["rgb"+pfix].shape[0:2], dtype=np.uint8)
                for k, v in img["mask"+pfix].items():
                    if k in self._dsa_key:
                        _mask_mat[v] = self._image_encode_id[k]
                layer3 = self._zoom_legal(
                    _mask_mat, zoom_x_min, zoom_x_max, zoom_y_min, zoom_y_max)
            im_pre = np.stack([layer1, layer2, layer3], axis=2)
            img_dsa = im_pre

        if self._timestep_prv != self.env.timestep:
            self._zoom_coordinate_prv = (
                zoom_x_min, zoom_x_max, zoom_y_min, zoom_y_max)
            self._timestep_prv = self.env.timestep
        # print(self._zoom_coordinate_prv)
        return img_dsa

    def _is_out_dsa_zoom(self, zoom_x, zoom_y, stuff_x, stuff_y, margin=0.2):
        m1 = margin * (zoom_x[1] - zoom_x[0]) / 2
        m2 = margin * (zoom_y[1] - zoom_y[0]) / 2
        self._out_zoom = (stuff_x[0] > zoom_x[1]-m1) or (stuff_x[1] < zoom_x[0]+m1) or (
            stuff_y[0] > zoom_y[1]-m2) or (stuff_y[1] < zoom_y[0]+m2)

    def _zoom2stuff(
        self,
        zoom_cx,
        zoom_cy,
        zoom_cz,
        stuff_cx,
        stuff_cy,
        stuff_cz,
        width,
        height,
        zoom_box_ratio,
    ):
        lx = int(width * zoom_box_ratio // 2) 
        ly = int(height * zoom_box_ratio // 2) 
        lz = int(255 * zoom_box_ratio // 2) 
        x = np.clip(np.abs(stuff_cx - zoom_cx) - lx, 0, None)
        y = np.clip(np.abs(stuff_cy - zoom_cy) - ly, 0, None)
        z = np.clip(np.abs(stuff_cz - zoom_cz) - lz, 0, None)
        is_out = (x+y+z) >0
        self._out_zoom = is_out
        xy_r = -(x**2 + y**2) * self._out_reward_xy
        z_r = -z * self._out_reward_z
        # print("xy reward:", xy_r, "z reward", z_r)
        self._dsa_reward = xy_r + z_r
        return is_out

    @property
    def reward_dict(
        self,
    ):
        if self._dense_reward:
            di = self.env.reward_dict.copy()
            di["prog_abnorm_3"] = self._dsa_reward
            return di
        else:
            return self.env.reward_dict

    @property
    def is_out_dsa_zoom(self,):
        return self._out_zoom

    @staticmethod
    def _scale_arr(_input, old_min, old_max, new_min, new_max):
        _in = _input
        _in = np.divide(_input-old_min, old_max-old_min)
        _in = np.multiply(_in, new_max-new_min) + new_min
        return _in

    def _zoom_legal(self, in_mat, x_min, x_max, y_min, y_max):
        zoom_mat = in_mat[x_min:x_max+1,
                          y_min:y_max+1]
        zoom_mat = cv2.resize(zoom_mat,
                              dsize=in_mat.shape, interpolation={"nearest": cv2.INTER_NEAREST,
                                                                 "linear": cv2.INTER_LINEAR,
                                                                 "area": cv2.INTER_AREA,
                                                                 "cubic": cv2.INTER_CUBIC, }[self._cv_interpolate])
        return zoom_mat

    def _get_zoom_coordinate(self, zoom_box_x, zoom_box_y, zoom_box_length, zoom_mask_shape):
        if self._zoom_movement_type == "continuous":
            zoom_x_min, zoom_x_max = self._get_legal_min_max_continuous(
                zoom_box_x[0], zoom_box_x[1], zoom_box_length, zoom_mask_shape[0])
            zoom_y_min, zoom_y_max = self._get_legal_min_max_continuous(
                zoom_box_y[0], zoom_box_y[1], zoom_box_length, zoom_mask_shape[1])
        elif self._zoom_movement_type == "discrete":
            if self._zoom_coordinate_prv is None:
                zoom_x_min, zoom_x_max = self._get_legal_min_max_continuous(
                    zoom_box_x[0], zoom_box_x[1], zoom_box_length, zoom_mask_shape[0])
                zoom_y_min, zoom_y_max = self._get_legal_min_max_continuous(
                    zoom_box_y[0], zoom_box_y[1], zoom_box_length, zoom_mask_shape[1])
            else:
                zoom_x_min, zoom_x_max = self._get_legal_min_max_discrete(
                    zoom_box_x[0], zoom_box_x[1], self._zoom_coordinate_prv[0], self._zoom_coordinate_prv[1], zoom_box_length, zoom_mask_shape[0])
                zoom_y_min, zoom_y_max = self._get_legal_min_max_discrete(
                    zoom_box_y[0], zoom_box_y[1], self._zoom_coordinate_prv[2], self._zoom_coordinate_prv[3], zoom_box_length, zoom_mask_shape[1])
        else:
            raise NotImplementedError
        return zoom_x_min, zoom_x_max, zoom_y_min, zoom_y_max

    def _get_legal_min_max_discrete(self, box_min, box_max,  zoom_min_prv, zoom_max_prv, box_length, length, move_scale=0.5, switch_margin=0.1):
        _low = np.int32(zoom_min_prv + switch_margin*box_length)
        _high = np.int32(zoom_max_prv - switch_margin*box_length)
        if box_min >= _low and box_max <= _high:
            zoom_min = zoom_min_prv
            zoom_max = zoom_max_prv
        elif box_min < _low:
            scale = (zoom_min_prv - box_min) // (box_length * move_scale) + 1
            zoom_min = zoom_min_prv - box_length * scale * move_scale
            zoom_max = zoom_max_prv - box_length * scale * move_scale
        elif box_max > _high:
            scale = (zoom_max_prv - box_max)//(box_length * move_scale) + 1
            zoom_min = zoom_min_prv + box_length * scale * move_scale
            zoom_max = zoom_max_prv + box_length * scale * move_scale
        else:
            raise NotImplementedError

        if zoom_min < 0:
            zoom_min = 0
            zoom_max = box_length-1
        elif zoom_max > length-1:
            zoom_min = length-1-box_length
            zoom_max = length-1
        return np.int32(zoom_min), np.int32(zoom_max)

    def _get_legal_min_max_continuous(self, box_min, box_max, box_length, length):
        _l = box_length // 2 if box_length % 2 == 0 else box_length // 2 + 1
        center = (box_min + box_max)//2
        if center-_l <= 0:
            out_min = 0
            out_max = box_length-1
        elif center+_l >= length-1:
            out_min = length-1-box_length
            out_max = length-1
        else:
            out_min = center-_l
            out_max = center+_l

        return np.int32(out_min), np.int32(out_max)

    def _get_zoom_mat(self, mask_mat, depth_mat, box_x, box_y, margin_ratio=0.1, fix_box_ratio=None):
        zoom_mat = self._segment(
            depth_mat, mask_mat)

        _zoom_box_length = max((box_x[1] - box_x[0]),
                               (box_y[1] - box_y[0])) * (1+margin_ratio)
        _zoom_box_cen_x = int((box_x[1] + box_x[0])//2)
        _zoom_box_cen_y = int((box_y[1] + box_y[0])//2)
        if fix_box_ratio is None:
            _l = int(_zoom_box_length // 2)
        else:
            _l = int((depth_mat.shape[0] * fix_box_ratio)//2)
        x_min = np.clip(_zoom_box_cen_x-_l, 0, depth_mat.shape[0]-1)
        x_max = np.clip(_zoom_box_cen_x+_l+1, 0, depth_mat.shape[0]-1)
        y_min = np.clip(_zoom_box_cen_y-_l, 0, depth_mat.shape[0]-1)
        y_max = np.clip(_zoom_box_cen_y+_l+1, 0, depth_mat.shape[0]-1)
        zoom_mat = zoom_mat[x_min:x_max,
                            y_min:y_max]

        try:
            zoom_mat = cv2.resize(zoom_mat,
                                  dsize=depth_mat.shape, interpolation=cv2.INTER_NEAREST)
        except:
            zoom_mat = np.full(
                depth_mat.shape, 0, dtype=np.uint8)
        return zoom_mat, (x_min, x_max), (y_min, y_max)

    def _get_box_from_masks(self, mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        try:
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            _s = mask.shape
            return (rmin, rmax), (cmin, cmax), (rmin/_s[0], rmax/_s[0]), (cmin/_s[1], cmax/_s[1])
        except:
            return (0, 0), (0, 0), (0, 0), (0, 0)  # by default

    def _segment(self, matrix, mask):
        mask_matrix = np.zeros(matrix.shape, dtype=np.uint8)
        # print(matrix)
        # print(mask.shape)
        mask_matrix[mask] = matrix[mask]
        # print(np.sum(mask_matrix))
        return mask_matrix

    def _reset_vars(self):
        self._zoom_coordinate_prv = None
        self._timestep_prv = self.env.timestep
