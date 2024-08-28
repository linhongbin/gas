import gym
from gym_ras.env.embodied.base_env import BaseEnv
import pybullet as p
from surrol.utils.pybullet_utils import get_link_pose
import numpy as np
from gym_ras.tool.common import *
from pathlib import Path
import colorsys


class SurrolEnv(BaseEnv):
    def __init__(self,
                 task,
                 pybullet_gui=False,
                 cid=-1,
                 cam_width=600,
                 cam_height=600,
                 mask_background_id=-3,
                 depth_remap_range=0.2,
                 depth_remap_range_noise=0.0,
                 depth_remap_center_noise=0.0,
                 cam_target_noise=0,
                 cam_distance_noise=0,
                 cam_yaw_noise=0,
                 cam_pitch_noise=0,
                 cam_roll_noise=0,
                 cam_up_axis_noise=0,
                 background_texture_dir="",
                 random_obj_vis=True,
                 cam_mode="rgbdm",
                 cam_num=1,
                 cam1_to_cam2_pose=[0.0, .0, .0, .0, .0, .0],
                 no_depth_link=False,
                 mask_except_gripper=False,
                 stuff_slide_thres=0.0012,
                 reward_done_success=1,
                 reward_done_fail=-0.2,
                 reward_prog_norm=0,
                 reward_prog_abnorm_1=-0.04,
                 reward_prog_abnorm_2=-0.04,
                 reward_prog_abnorm_3=-0.01,
                 disturbance_scale=0.03,
                 cam_dynamic_noise_scale=0.1,
                 gripper_toggle_anolamy=False,
                 no_depth_noise=0.9,
                 dr_scale_train=1,
                 dr_scale_eval=1,
                 stuff_dist=False,
                 **kwargs,
                 ):
        self._cam_width = cam_width
        self._cam_height = cam_height
        self._gripper_toggle_anolamy = gripper_toggle_anolamy
        self._no_depth_noise = no_depth_noise
        self._mask_obj = ["psm1", "stuff"]
        self._dr_scale = {"train": dr_scale_train, "eval": dr_scale_eval, }
        self._mode = "train"
        self._stuff_dist = stuff_dist
        if task == "needle_pick":
            from gym_ras.env.embodied.surrol.needle_pick import NeedlePickMod
            # print(kwargs)
            client = NeedlePickMod(
                render_mode="human" if pybullet_gui else "rgb_array", cid=cid, **kwargs)
        elif task == "gauze_retrieve":
            from gym_ras.env.embodied.surrol.gauze_retrieve import GauzeRetrieveMod
            client = GauzeRetrieveMod(
                render_mode="human" if pybullet_gui else "rgb_array", cid=cid, **kwargs)
        elif task == "peg_transfer":
            from gym_ras.env.embodied.surrol.peg_transfer import PegTransferMod
            client = PegTransferMod(
                render_mode="human" if pybullet_gui else "rgb_array", cid=cid, **kwargs)
        elif task == "grasp_any":
            from gym_ras.env.embodied.surrol.grasp_any import GraspAny
            client = GraspAny(
                render_mode="human" if pybullet_gui else "rgb_array", cid=cid, **kwargs)
        elif task == "grasp_any_v2":
            from gym_ras.env.embodied.surrol.grasp_any_v2 import GraspAnyV2
            client = GraspAnyV2(
                render_mode="human" if pybullet_gui else "rgb_array", cid=cid, **kwargs)
        else:
            raise Exception("Not support")

        super().__init__(client)
        self._cam_dynamic_noise_scale = cam_dynamic_noise_scale
        self._disturbance_scale = disturbance_scale
        self._random_obj_vis = random_obj_vis
        self._view = self.client._view_param
        self._project = self.client._proj_param
        self._mask_background_id = mask_background_id
        self._view["depth_remap_range"] = depth_remap_range * \
            self.client.SCALING
        self._view["depth_remap_range_noise"] = depth_remap_range_noise * \
            self.client.SCALING
        self._view["depth_remap_center_noise"] = depth_remap_center_noise * \
            self.client.SCALING
        self._view["target_noise"] = cam_target_noise * self.client.SCALING
        self._view["distance_noise"] = cam_distance_noise * self.client.SCALING
        self._view["yaw_noise"] = cam_yaw_noise
        self._view["pitch_noise"] = cam_pitch_noise
        self._view["yaw_noise"] = cam_yaw_noise
        self._view["roll_noise"] = cam_roll_noise
        self._view["up_axis_noise"] = cam_up_axis_noise
        self._texture_dir = {
            "tray": Path(__file__).resolve().parent.parent.parent.parent / "asset"
            if background_texture_dir is "" else background_texture_dir,
        }
        self._texture_extension = ["png", "jpeg", "jpg"]
        assert cam_mode in ["rgbdm", "rgbm", "rgbm_2"]
        self._cam_mode = cam_mode
        self._cam_num = cam_num
        self.seed = 0
        self._view_matrix = [None,] * cam_num
        self._cam1_to_cams_pose = [[0.0, 0, 0, 0, 0, 0], cam1_to_cam2_pose]
        self._proj_matrix = [None,] * cam_num
        self._depth_remap_range = [None,] * cam_num
        self._no_depth_link = no_depth_link
        self._mask_except_gripper = mask_except_gripper
        self._stuff_slide_thres = stuff_slide_thres
        self._reward_dict = {
            "done_success": reward_done_success,
            "done_fail": reward_done_fail,
            "prog_norm": reward_prog_norm,
            "prog_abnorm_1": reward_prog_abnorm_1,
            "prog_abnorm_2": reward_prog_abnorm_2,
            "prog_abnorm_3": reward_prog_abnorm_3,
        }

    @property
    def reward_dict(self, ):
        return self._reward_dict

    def _gen_cam_noise(self, bound, center, new_center=None):
        if new_center is None:
            out = self._cam_pose_rng.uniform(-np.abs(bound),
                                             np.abs(bound)) + center
        else:
            b = bound * self._cam_dynamic_noise_scale
            out = self._cam_pose_rng.uniform(-np.abs(b),
                                             np.abs(b)) + new_center
            out = np.clip(out, center-bound, center+bound,)
        return out

    def _reset_cam(self, cam_id, reset=True):
        # add_noise_fuc = lambda x, low, high: np.array(x) + self._cam_pose_rng.uniform(low, high)
        workspace_limits = self.client.workspace_limits1
        target_pos = [workspace_limits[0].mean(
        ),  workspace_limits[1].mean(),  workspace_limits[2][0]]
        roll = self._view["roll"]
        pitch = self._view["pitch"]
        yaw = self._view["yaw"]
        distance = self._view["distance"]

        def ns(str): return None if reset else self._prv_cam_noise[str]
        _dis_noise = self._gen_cam_noise(
            self._view["distance_noise"] * self._dr_scale[self.mode], distance, ns('dis'))
        _roll_noise = self._gen_cam_noise(
            self._view["roll_noise"]*self._dr_scale[self.mode], 0, ns('roll'))
        _pitch_noise = self._gen_cam_noise(
            self._view["pitch_noise"]*self._dr_scale[self.mode], pitch, ns('pitch'))
        _yaw_noise = self._gen_cam_noise(
            self._view["yaw_noise"]*self._dr_scale[self.mode], yaw, ns('yaw'))
        _cam_target_noise_x = self._gen_cam_noise(
            self._view["target_noise"]*self._dr_scale[self.mode], target_pos[0], ns('cam_target_noise_x'))
        _cam_target_noise_y = self._gen_cam_noise(
            self._view["target_noise"]*self._dr_scale[self.mode], target_pos[1], ns('cam_target_noise_y'))
        _up_axis_noise = self._gen_cam_noise(
            self._view["up_axis_noise"]*self._dr_scale[self.mode], 0, ns('up'))
        _depth_remap_range_noise = self._gen_cam_noise(
            self._view["depth_remap_range_noise"]*self._dr_scale[self.mode], 0, None)
        _depth_remap_center_noise = self._gen_cam_noise(
            self._view["depth_remap_center_noise"]*self._dr_scale[self.mode], 0, None)
        T1 = getT(target_pos, [roll, _pitch_noise,
                  _yaw_noise],  rot_type="euler")
        T2 = getT([0, 0, _dis_noise], [0, 0, 0],  rot_type="euler")
        T3 = getT([0, 0, 0], [0, 0, _roll_noise],  rot_type="euler")
        T = TxT([T1, T2])
        # _T_M=T[0:3,0:3]
        # _T_p=T[0:3,3]
        target_pos[0] = _cam_target_noise_x
        target_pos[1] = _cam_target_noise_y
        T_cam = TxT([T, getT(self._cam1_to_cams_pose[cam_id][:3],
                             self._cam1_to_cams_pose[cam_id][3:], rot_type="euler", euler_convension="xyz", euler_Degrees=True)])
        self._view_matrix[cam_id] = p.computeViewMatrix(cameraEyePosition=T_cam[0:3, 3].tolist(),
                                                        cameraTargetPosition=target_pos,
                                                        cameraUpVector=[0, np.sin(np.radians(_up_axis_noise)), np.cos(
                                                            np.radians(_up_axis_noise))],
                                                        )

        self._proj_matrix[cam_id] = p.computeProjectionMatrixFOV(fov=self._project["fov"],
                                                                 aspect=float(
                                                                     self._cam_width) / self._cam_height,
                                                                 nearVal=self._project["nearVal"],
                                                                 farVal=self._project["farVal"])
        _dis = _dis_noise
        _center = _dis + _depth_remap_center_noise
        _range = (self._view["depth_remap_range"] +
                  _depth_remap_range_noise) / 2
        _low = _center - _range
        _high = _center + _range
        self._depth_remap_range[cam_id] = (_low, _high)

        self._prv_cam_noise = {}
        self._prv_cam_noise['dis'] = _dis_noise
        self._prv_cam_noise['roll'] = _roll_noise
        self._prv_cam_noise['pitch'] = _pitch_noise
        self._prv_cam_noise['yaw'] = _yaw_noise
        self._prv_cam_noise['cam_target_noise_x'] = _cam_target_noise_x
        self._prv_cam_noise['cam_target_noise_y'] = _cam_target_noise_y
        self._prv_cam_noise['up'] = _up_axis_noise

    def get_oracle_action(self):
        return self.client.get_oracle_action(self.client._get_obs())

    def _get_obj_pose(self, obj_id: str, link_index: int):
        # assert obj_id in self.obj_ids
        return get_link_pose(self.client.keyobj_ids[obj_id], link_index)

    def reset(self):
        self._init_vars()
        for j in range(self._cam_num):
            self._reset_cam(j)
        _ = self.client.reset()
        if self._random_obj_vis:
            self._random_background_obj_vis()
        obs, reward, done, info = self.client.step(
            np.array([0.0, 0.0, 0.0, 0.0, 1.0]))
        obs["robot_prio"], obs["gripper_state"] = self._get_prio_obs()
        # obs = {}
        # obs["robot_prio"], obs["gripper_state"] = self._get_prio_obs()
        # info = {}
        # info["fsm"] = self.client._fsm()
        self.step_func_prv = (obs, reward, done, info)
        # self._has_constraint_prv = self._get_contact_constraint()
        self._stuff_pos_prv = self._get_obj_pose("stuff", -1)[0]
        return obs

    def step(self, action):
        self.timestep += 1
        if not self.skip:
            if self._cam_dynamic_noise_scale > 0:
                for j in range(self._cam_num):
                    self._reset_cam(j, reset=False)
            _prio, _ = self._get_prio_obs()
            _is_out, action_clip = self._check_new_action(_prio, action[:3])
            _action = action.copy()
            _action[:3] = action_clip
            obs, reward, done, info = self.client.step(_action)

            obs["robot_prio"], obs["gripper_state"] = self._get_prio_obs()

            gripper_toggle = np.abs(
                obs["gripper_state"]-self.step_func_prv[0]["gripper_state"]) > 0.1
            _str = self._get_fsm_prog_abnorm(gripper_toggle, _is_out)
            if _str != "":
                info["fsm"] = _str
            if not self.client._is_grasp_obj() and self._stuff_dist:
                self._disturbance_on_stuff()

            self.step_func_prv = obs, reward, done, info
            self._stuff_pos_prv = self._get_obj_pose("stuff", -1)[0]

        return self.step_func_prv

    def _disturbance_on_stuff(self):
        cur_pos, cur_quat = self._get_obj_pose("stuff", -1)
        cur_M = Quat2M(cur_quat)
        # set random stuff init pose
        ws = self.client.workspace_limits1
        new_high = np.array(
            [ws[0][1]-ws[0][0], ws[1][1]-ws[1][0], 0.00001, 100])
        new_low = -new_high.copy()
        new_low[2] = new_high[2]
        pos_rel = self.client._stuff_pose_rng.uniform(-self._disturbance_scale/2*self._dr_scale[self.mode],
                                                      self._disturbance_scale/2 *
                                                      self._dr_scale[self.mode],
                                                      size=new_low.shape)
        # print(pos_rel)
        _delta = scale_arr(pos_rel, -np.ones(pos_rel.shape),
                           np.ones(pos_rel.shape), new_low, new_high)
        # print(_delta)

        M1 = Euler2M([0, 0, _delta[3]], convension="xyz", degrees=True)
        M2 = np.matmul(M1, cur_M)
        quat = M2Quat(M2)
        pos = cur_pos + _delta[:3]
        pos[:2] = np.clip(pos[:2], ws[:2, 0]+0.08, ws[:2, 1]-0.08)
        p.resetBasePositionAndOrientation(
            self.client.obj_ids['rigid'][0], pos, quat)

    def _get_fsm_prog_abnorm(self, gripper_toggle, is_out):
        if is_out:
            return "prog_abnorm_1"
        _pos = self._get_obj_pose("stuff", -1)[0]
        _pos_delta = np.array(_pos) - np.array(self._stuff_pos_prv)
        # print(np.linalg.norm(_pos_delta[:2]), _pos_delta[:2])
        if (np.linalg.norm(_pos_delta[:2]) > self._stuff_slide_thres * self.client.SCALING) and (not self.client._is_grasp_obj()):
            return "prog_abnorm_2"
        if gripper_toggle and self._gripper_toggle_anolamy:
            return "prog_abnorm_3"
        return ""

    def render(self,):
        imgs = {}
        for j in range(self._cam_num):
            postfix = "_"+str(j+1) if j != 0 else ""
            (_, _, px, depth, mask) = p.getCameraImage(width=self._cam_width,
                                                       height=self._cam_height,
                                                       viewMatrix=self._view_matrix[j],
                                                       projectionMatrix=self._proj_matrix[j],
                                                       shadow=1,
                                                       lightDirection=(
                                                           10, 0, 10),
                                                       renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                                       flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
            px = px[:, :, :3]  # the 4th channel is alpha
            imgs["rgb"+postfix] = px

            # ref: https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/segmask_linkindex.py
            in_obj_data = np.bitwise_and(mask, ((1 << 24) - 1))
            in_link_data = (np.right_shift(mask, 24)) - 1
            in_obj_data[mask <= 0] = self._mask_background_id
            in_link_data[mask <= 0] = self._mask_background_id

            if self._cam_mode.find("d") >= 0:
                far = self._project["farVal"]
                near = self._project["nearVal"]
                depth = far * near / (far - (far - near) * depth)
                depth = np.uint8(np.clip(self._scale(
                    depth, self._depth_remap_range[j][0], self._depth_remap_range[j][1], 0, 255), 0, 255))  # to image
                imgs["depth"+postfix] = depth
                if self._no_depth_link:
                    for k, v in self.client.nodepth_link_ids.items():
                        _mask = self._get_mask(
                            in_obj_data, in_link_data, self.client.keyobj_ids[k], v)
                        if self._no_depth_noise > 0:
                            _noise = np.random.uniform(-122,
                                                       122) * self._no_depth_noise
                            # print("noise", _noise)
                            imgs["depth"+postfix][_mask] = np.clip(
                                imgs["depth"+postfix][_mask] + _noise, 0, 255)
                        else:
                            imgs["depth"+postfix][_mask] = 0

            if self._cam_mode.find("m") >= 0:
                masks = {}
                for _mask_obj in self.client.keyobj_link_ids:
                    _obj_id = [
                        v for k, v in self.client.keyobj_ids.items() if _mask_obj.find(k) >= 0]
                    assert len(_obj_id) != 0
                    _obj_id = _obj_id[0]
                    _obj_link_id = self.client.keyobj_link_ids[_mask_obj]
                    if self._mask_except_gripper and _mask_obj == "psm1":
                        _obj_link_id.extend(
                            self.client.keyobj_link_ids["psm1_except_gripper"])
                    masks[_mask_obj] = self._get_mask(
                        in_obj_data, in_link_data, _obj_id, _obj_link_id)

                imgs["mask"+postfix] = masks

        return imgs

    def _get_mask(self, in_obj_data, in_link_data,  _obj_id, _obj_link_id):
        masks = (in_obj_data == _obj_id) & (
            self._mask_or(in_link_data, _obj_link_id))
        return masks

    def _get_prio_obs(self):
        # norm by workspace
        obs = self.client._get_robot_state(idx=0)  # for psm1
        tip_pos = obs[:3]
        gripper_state = obs[6]
        if gripper_state > np.deg2rad(22):  # discrete
            gripper_state = 1.0
        else:
            gripper_state = -1.0
        return tip_pos, gripper_state

    def _check_new_action(self, state, action):
        pos = action * self.action2pos
        new_state = state + pos
        ws = self.workspace_limit
        _low = ws[:, 0]
        _high = ws[:, 1]
        # print("low exceed", new_state < _low, new_state - _low)
        # print("high exceed", new_state > _high)
        is_out_ws = np.any(new_state < _low) or (np.any(new_state > _high))

        eps = np.ones(_low.shape) * 0.0001  # numerical stable
        new_state_clip = np.clip(new_state, _low+eps, _high-eps)
        action_clip = (new_state_clip - state) / self.action2pos
        return is_out_ws, action_clip

    @staticmethod
    def _scale(input, old_min, old_max, new_min, new_max):
        out = (input-old_min)/(old_max-old_min)*(new_max-new_min) + new_min
        return out

    def _change_obj_vis(self, obj_id, obj_link_id, rgb_list=[1, 1, 1, 1], texture_dir=None):
        # newmetal = p.loadTexture(os.path.join(ASSET_DIR_PATH, "texture/metal2.jpg"))
        if texture_dir is not None:
            textid = p.loadTexture(texture_dir)
            p.changeVisualShape(obj_id, obj_link_id,
                                rgbaColor=rgb_list, textureUniqueId=textid)
        else:
            p.changeVisualShape(obj_id, obj_link_id, rgbaColor=rgb_list,)

    def _random_background_obj_vis(self,):
        _obj_id = {}
        _obj_id.update(self.client.background_obj_ids)
        _obj_id.update(self.client.keyobj_ids)
        _link_id = {}
        _link_id.update(self.client.background_obj_link_ids)
        _link_id.update(self.client.keyobj_link_ids)
        for k in self.client.random_vis_key:
            for link in _link_id[k]:
                color_type = self.client.random_color_range[0]
                if k in self.client.random_color_range[1]:
                    _range = self.client.random_color_range[1][k]
                else:
                    _range = self.client.random_color_range[1]["default"]
                _val = self._background_vis_rng.uniform(_range[0], _range[1])
                _rgb = _val.tolist()
                if color_type == "hsv":
                    _new_val = colorsys.hsv_to_rgb(_val[0], _val[1], _val[2])
                    _rgb = list(_new_val)
                    _rgb.append(_val[-1])
                if not k in self._texture_dir:
                    texture_dir = None
                else:
                    files = []
                    for ext in self._texture_extension:
                        files.extend(
                            sorted(Path(self._texture_dir[k]).glob("*."+ext)))
                    texture_dir = str(
                        files[self._background_vis_rng.randint(len(files))])
                self._change_obj_vis(
                    _obj_id[k], link, _rgb, texture_dir=texture_dir)

    def _init_rng(self, seed):
        cam_pose_seed = np.uint32(seed+1)
        depth_remap_seed = np.uint32(seed+2)
        background_vis_seed = np.uint32(seed+3)
        print("cam_pose_seed:", cam_pose_seed)
        print("depth_remap_seed:", depth_remap_seed)
        self._cam_pose_rng = np.random.RandomState(cam_pose_seed)
        self._depth_remap_rng = np.random.RandomState(depth_remap_seed)
        self._background_vis_rng = np.random.RandomState(background_vis_seed)

    def get_stuff_pose(self):
        needle_id = self.obj_ids['rigid'][0]
        pos, orn = get_link_pose(needle_id, -1)
        return pos, orn

    def to_eval(self):
        self._mode = "eval"

    def to_train(self):
        self._mode = "train"

    @staticmethod
    def _mask_or(mask, ids):
        x = None
        for _id in ids:
            out = mask == _id
            x = out if x is None else x | out
        return x

    @property
    def workspace_limit(self,):
        return self.client.workspace_limits1

    @property
    def action_space(self):
        return self.client.action_space

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self.client.seed(seed)
        self._init_rng(seed)

    @property
    def observation_space(self):
        obs = {}
        obs['gripper_state'] = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
        prio, _ = self._get_prio_obs()
        ws = self.workspace_limit
        _low = ws[:, 0]
        _high = ws[:, 1]
        obs['robot_prio'] = gym.spaces.Box(_low, _high, dtype=np.float32)
        return gym.spaces.Dict(obs)

    @property
    def nodepth_guess_map(self):
        return self.client.nodepth_guess_map

    @property
    def nodepth_guess_uncertainty(self):
        return self.client.nodepth_guess_uncertainty

    @property
    def depth_remap_range(self):
        return self._depth_remap_range

    @property
    def action2pos(self,):
        return 0.01 * self.client.SCALING

    @property
    def mode(self,):
        return self._mode


if __name__ == "__main__":
    env = SurrolEnv(name="needle_pick")
