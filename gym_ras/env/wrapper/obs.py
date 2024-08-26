from gym_ras.env.wrapper.base import BaseWrapper
import numpy as np
import gym
import cv2


class OBS(BaseWrapper):
    """ specify observation """

    def __init__(self, env,
                 image_obs_key=["dsa"],
                 vector_obs_key=["gripper_state", "fsm_state",],
                 direct_map_key=["fsm_state"],
                 direct_render_key=["rgb"],
                 is_vector2image=True,
                 vector2image_type="row",
                 image_resize=[64, 64],
                 cv_interpolate="area",
                 dsa_out_zoom_anamaly=False,
                 action_insertion_anamaly=False,
                 **kwargs,
                 ):
        super().__init__(env,)
        self._image_obs_key = image_obs_key
        self._vector_obs_key = vector_obs_key
        self._is_vector2image = is_vector2image
        self._image_resize = image_resize
        self._direct_map_key = direct_map_key
        self._cv_interpolate = cv_interpolate
        self._direct_render_key = direct_render_key
        self._vector2image_type = vector2image_type
        self._dsa_out_zoom_anamaly = dsa_out_zoom_anamaly
        self._action_insertion_anamaly = action_insertion_anamaly

        obs = self.reset()
        self._obs_shape = {k: v.shape if isinstance(
            v, np.ndarray) else None for k, v in obs.items()}
        self._obs_image = None

    def reset(self,):
        obs = self.env.reset()
        return self._get_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self._action_insertion_anamaly:
            if action == 4 and obs['gripper_state'] < 0:
                _str = "prog_abnorm_1"
                obs['fsm_state'] = self.env.fsm_states.index(_str)
                reward = self.env.reward_dict[_str]
                info["fsm"] = _str
        obs = self._get_obs(obs)
        # if self._dsa_out_zoom_anamaly:
        #     if self.env.is_out_dsa_zoom and info["fsm"] != "done_success":
        #         _str = "prog_abnorm_3"
        #         obs['fsm_state'] = self.env.fsm_states.index(_str)
        #         reward = self.env.reward_dict[_str]
        #         info["fsm"] = _str

        return obs, reward, done, info

    def render(self, ):
        img = self.env.render()
        if self._obs_image is not None:
            img["obs"] = self._obs_image
            # print("sdfasdfasdfasdfasdfas")
        return img

    def _get_obs_low_high(self):
        def low_high_tuple(x): return (x.low, x.high, x.shape)

        def get_bound(
            x, shape): return x if x.shape[0] == shape else x*np.ones(shape)
        _list = [low_high_tuple(self.env.observation_space[k])
                 for k in self._vector_obs_key]
        if len(_list) == 1:
            k = _list[0]
            _low = get_bound(k[0], k[2])
            _high = get_bound(k[1], k[2])
        else:
            _low = np.concatenate(
                tuple([get_bound(k[0], k[2]) for k in _list]), axis=0)
            _high = np.concatenate(
                tuple([get_bound(k[1], k[2]) for k in _list]), axis=0)

        def np_arr_func(x): return x if isinstance(
            x, np.ndarray) else np.array([x])
        return np_arr_func(_low), np_arr_func(_high)

    def _get_obs(self, _obs):
        obs = {}
        for v in self._direct_map_key:
            obs[v] = _obs[v]
        _img = self.env.render()
        if "depth" in _img:
            _img["depth"] = np.stack([_img["depth"]]*3, axis=2)
        for v in self._direct_render_key:
            obs[v] = _img[v]
            if self._image_resize[0] > 0:
                obs[v] = cv2.resize(obs[v],
                                    tuple(self._image_resize),
                                    interpolation={"nearest": cv2.INTER_NEAREST,
                                                   "linear": cv2.INTER_LINEAR,
                                                   "area": cv2.INTER_AREA,
                                    "cubic": cv2.INTER_CUBIC, }[self._cv_interpolate])

        def np_arr_func(x): return x if isinstance(
            x, np.ndarray) else np.array([x])
        if len(self._image_obs_key) > 0:
            obs["image"] = _img[self._image_obs_key[0]] \
                if len(self._image_obs_key) == 1 \
                else np.concatenate(tuple([_img[k] for k in self._image_obs_key]), axis=2)
            if self._image_resize[0] > 0:
                obs["image"] = cv2.resize(obs["image"],
                                          tuple(self._image_resize),
                                          interpolation={"nearest": cv2.INTER_NEAREST,
                                                         "linear": cv2.INTER_LINEAR,
                                                         "area": cv2.INTER_AREA,
                                                         "cubic": cv2.INTER_CUBIC, }[self._cv_interpolate])

        if len(self._vector_obs_key) > 0:
            obs["vector"] = _obs[self._vector_obs_key[0]] \
                if len(self._vector_obs_key) == 1 \
                else np.concatenate(tuple([np_arr_func(_obs[k]) for k in self._vector_obs_key]), axis=0)
            _low, _high = self._get_obs_low_high()
            # print("kkkk",obs["vector"])
            obs["vector"] = self._scale(
                obs["vector"], _low, _high, -np.ones(_low.shape), np.ones(_low.shape))
            if self._is_vector2image:
                obs["image"] = self._vector2image(obs["image"], obs["vector"])
                obs.pop('vector', None)
        self._obs_image = obs["image"]
        return obs

    def _vector2image(self, image_in, vector, fill_channel=0, ):
        image = np.copy(image_in)
        _value_norm = np.clip(self._scale(
            vector, old_min=-1, old_max=1, new_min=0.0, new_max=255.0), 0, 255.0)
        _value_norm = _value_norm.astype(np.uint8)
        # print("jj",_value_norm)
        if self._vector2image_type == "row":
            # print(_value_norm.shape)
            extend_d = 32
            _value_norm_l = np.zeros((_value_norm.shape[0] * extend_d,), dtype=np.uint8)
            # print(_value_norm)
            for i in range(_value_norm.shape[0]):
                _value_norm_l[i * extend_d: (i + 1) * extend_d] = _value_norm[i]
            # print(_value_norm_l)
            _value_norm = _value_norm_l
            _value_norm = np.tile(_value_norm, (image.shape[0], 1))
            _value_norm = np.transpose(_value_norm)
            s = image.shape[0]
            image[s-_value_norm.shape[0]:, :, fill_channel] = _value_norm
        elif self._vector2image_type == "square":
            ROW_SIZE = 6
            for _v in range(_value_norm.shape[0]):
                s = image.shape[0]
                a1 = s-_v*ROW_SIZE-1-ROW_SIZE
                b1 = s-_v*ROW_SIZE-1
                a2 = s-0*ROW_SIZE-1-ROW_SIZE
                b2 = s-0*ROW_SIZE-1
                image[a2:b2, a1:b1, fill_channel] = _value_norm[_v]

        elif self._vector2image_type == "pixel":
            _shape = image.shape
            _x = np.reshape(image[:, :, fill_channel], (-1))
            _x[_x.shape[0]-_value_norm.shape[0]:_x.shape[0]+1] = _value_norm
            _x = np.reshape(_x, image.shape[:2])
            image[:, :, fill_channel] = _x[:, :]
        # print(_value_norm)
        return image

    @staticmethod
    def _scale(_input, old_min, old_max, new_min, new_max):
        _in = _input
        _in = np.divide(_input-old_min, old_max-old_min)
        _in = np.multiply(_in, new_max-new_min) + new_min
        return _in

    @property
    def observation_space(self):
        obs = {}
        for v in self._direct_map_key:
            obs[v] = self.env.observation_space[v]
        for v in self._direct_render_key:
            obs[v] = gym.spaces.Box(0, 255, self._obs_shape[v],
                                    dtype=np.uint8)

        if "image" in self._obs_shape:
            obs['image'] = gym.spaces.Box(0, 255, self._obs_shape["image"],
                                          dtype=np.uint8)

        if "vector" in self._obs_shape and (not self._is_vector2image):
            _low, _high = self._get_obs_low_high()
            obs['vector'] = gym.spaces.Box(-1, 1, self._obs_shape["vector"],
                                           dtype=np.float32)

        return gym.spaces.Dict(obs)
