from surrol.tasks.peg_transfer import PegTransfer
import numpy as np
import pybullet as p
from surrol.utils.pybullet_utils import (
    get_link_pose,
    wrap_angle
)
import os
from gym_ras.tool.common import *


class PegTransferMod(PegTransfer):
    def __init__(self, render_mode=None, cid=-1,
                 fix_goal=True,
                 oracle_pos_thres=3e-3,
                 oracle_rot_thres=3e-1,
                 noise_scale=1,
                 reward_done_success=1,
                 reward_done_fail=-0.1,
                 reward_in_progress=-0.001,
                 reward_progress_fail=-0.01,
                 done_z_thres=0.3,
                 init_pose_ratio_low_gripper=[-0.5, -0.5, -0.5, -0.9],
                 init_pose_ratio_high_gripper=[0.5, 0.5, 0.5, 0.9],
                 init_pose_ratio_low_stuff=[-0.5, -0.5, 0.1, -0.99],
                 init_pose_ratio_high_stuff=[0.5, 0.5, 0.5, 0.99],
                 **kwargs,
                 ):
        self._init_pose_ratio_low_gripper = init_pose_ratio_low_gripper
        self._init_pose_ratio_high_gripper = init_pose_ratio_high_gripper
        self._init_pose_ratio_low_stuff = init_pose_ratio_low_stuff
        self._init_pose_ratio_high_stuff = init_pose_ratio_high_stuff

        self._fix_goal = fix_goal
        super().__init__(render_mode, cid)
        self._view_param = {"distance": 0.8,
                            "yaw": 180, "pitch": -45, "roll": 0,
                            }
        self._proj_param = {"fov": 42, "nearVal": 0.1, "farVal": 1000}
        self._oracle_pos_thres = oracle_pos_thres
        self._oracle_rot_thres = oracle_rot_thres
        self._reward_dict = {"done_success": reward_done_success, "done_fail": reward_done_fail,
                             "in_progress": reward_in_progress, "progress_fail": reward_progress_fail}
        self._done_z_thres = done_z_thres

    def _env_setup(self):
        super()._env_setup()

        # set random gripper init pose
        pos_rel = self._gripper_pose_rng.uniform(
            self._init_pose_ratio_low_gripper, self._init_pose_ratio_high_gripper)
        ws = self.workspace_limits1
        new_low = np.array([ws[0][0], ws[1][0], ws[2][0], -180])
        new_high = np.array([ws[0][1], ws[1][1], ws[2][1], 180])
        pose = scale_arr(pos_rel, -np.ones(pos_rel.shape),
                         np.ones(pos_rel.shape), new_low, new_high)
        M = Quat2M([0.5, 0.5, -0.5, -0.5])
        M1 = Euler2M([0, 0, pose[3]], convension="xyz", degrees=True)
        M2 = np.matmul(M1, M)
        quat = M2Quat(M2)
        pos = pose[:3]
        joint_positions = self.psm1.inverse_kinematics(
            (pos, quat), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)

        # ## set random stuff init pose
        # pos_rel = self._stuff_pose_rng.uniform(self._init_pose_ratio_low_stuff,self._init_pose_ratio_high_stuff)
        # new_low = np.array([ws[0][0],ws[1][0],ws[2][0]+ 0.01, -180])
        # new_high = np.array([ws[0][1],ws[1][1],ws[2][1]- 0.01, 180])
        # pose = scale_arr(pos_rel, -np.ones(pos_rel.shape),np.ones(pos_rel.shape), new_low, new_high)
        # M = Euler2M([0,0,0], convension="xyz",degrees=True)
        # M1 = Euler2M([0,0,pose[3]],convension="xyz",degrees=True)
        # M2 = np.matmul(M1,M)
        # quat = M2Quat(M2)
        # pos = pose[:3]
        # p.resetBasePositionAndOrientation(self.obj_ids['rigid'][0], pos, quat)

    def reset(self):
        obs = super().reset()
        self._init_stuff_z = self._get_stuff_z()
        self._create_waypoint()
        return obs

    def _create_waypoint(self):
        self._WAYPOINTS = [None, None, None, None, None, None]  # six waypoints
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        orn = p.getEulerFromQuaternion(orn_obj)
        orn_eef = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1]
        orn_eef = p.getEulerFromQuaternion(orn_eef)
        yaw = orn[2] if abs(wrap_angle(orn[2] - orn_eef[2])) < abs(wrap_angle(orn[2] + np.pi - orn_eef[2])) \
            else wrap_angle(orn[2] + np.pi)  # minimize the delta yaw

        self._WAYPOINTS[0] = np.array([pos_obj[0]-0.0275, pos_obj[1]+0.005,
                                       pos_obj[2] + 0.045 * self.SCALING, yaw, 0.5])  # above object
        self._WAYPOINTS[1] = np.array([pos_obj[0]-0.0275, pos_obj[1]+0.005,
                                       pos_obj[2] + (0.003 + 0.0102) * self.SCALING, yaw, 0.5])  # approach
        self._WAYPOINTS[2] = np.array([pos_obj[0]-0.0275, pos_obj[1]+0.005,
                                       pos_obj[2] + (0.003 + 0.0102) * self.SCALING, yaw, -0.5])  # grasp
        self._WAYPOINTS[3] = np.array([pos_obj[0]-0.0275, pos_obj[1]+0.005,
                                       pos_obj[2] + 0.045 * self.SCALING, yaw, -0.5])  # lift up

        # pos_peg = get_link_pose(self.obj_ids['fixed'][1], self.obj_id - np.min(self._blocks) + 6)[0]  # 6 pegs
        pos_peg = get_link_pose(self.obj_ids['fixed'][1],
                                self._pegs[self.obj_id - np.min(self._blocks) + 6])[0]  # 6 pegs
        pos_place = [self.goal[0] + pos_obj[0] - pos_peg[0],
                     self.goal[1] + pos_obj[1] - pos_peg[1], self._WAYPOINTS[0][2]]  # consider offset
        self._WAYPOINTS[4] = np.array(
            [pos_place[0]-0.02, pos_place[1]+0.01, pos_place[2], yaw, -0.5])  # above goal
        self._WAYPOINTS[5] = np.array(
            [pos_place[0]-0.02, pos_place[1]+0.01, pos_place[2], yaw, 0.5])  # release

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info["fsm"] = self._fsm()
        return obs, reward, done, info

    def _is_grasp_obj(self):
        return not (self._contact_constraint == None)

    def _get_stuff_z(self):
        stuff_id = self.obj_id
        pos_obj, _ = get_link_pose(stuff_id, -1)
        return pos_obj[2]

    def _fsm(self):
        """ ["in_progress","done_success","done_fail","progress_fail"] """
        obs = self._get_robot_state(idx=0)  # for psm1
        tip = obs[2]
        # if self._is_grasp_obj() and (tip-self.workspace_limits1[2][0])>self._done_z_thres:
        #     return "done_success"
        return "in_progress"

    def action2pos(self, action):
        return action * 0.01 * self.SCALING

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        # four waypoints executed in sequential order
        action = np.zeros(5)
        action[4] = -0.5
        for i, waypoint in enumerate(self._WAYPOINTS):
            if waypoint is None:
                continue
            if i == 4 and (not self._is_grasp_obj()):
                self._create_waypoint()
                return self.get_oracle_action(obs)
            delta_pos = (waypoint[:3] - obs['observation']
                         [:3]) / 0.01 / self.SCALING
            delta_yaw = 0
            if np.abs(delta_pos).max() > 1:
                delta_pos /= np.abs(delta_pos).max()
            scale_factor = 0.4
            delta_pos *= scale_factor
            action = np.array([delta_pos[0], delta_pos[1],
                              delta_pos[2], delta_yaw, waypoint[4]])
            if np.linalg.norm(delta_pos) * 0.01 / scale_factor < self._oracle_pos_thres:
                self._WAYPOINTS[i] = None
            break

        return action

    def seed(self, seed=0):
        super().seed(seed)
        self._init_rng(seed)

    def _init_rng(self, seed):
        stuff_pose_seed = np.uint32(seed)
        gripper_pose_seed = np.uint32(seed-1)
        print("stuff_pose_seed:", stuff_pose_seed)
        print("gripper_pose_seed:", gripper_pose_seed)
        self._stuff_pose_rng = np.random.RandomState(stuff_pose_seed)
        self._gripper_pose_rng = np.random.RandomState(
            gripper_pose_seed)  # use different seed

    @property
    def keyobj_ids(self,):
        return {"psm1": self.psm1.body,
                "stuff": self.obj_id}

    @property
    def keyobj_link_ids(self,):
        return {
            "psm1": [4, 5, 6, 7],
            "psm1_except_gripper": [3,],
            "stuff": [-1],
        }

    @property
    def nodepth_link_ids(self,):
        return {
            "psm1": [6, 7],
        }

    @property
    def background_obj_ids(self,):
        return {
            "tray": self.obj_ids['fixed'][1],
        }

    @property
    def background_obj_link_ids(self,):
        return {
            "tray": [-1, 4],
        }

    @property
    def random_vis_key(self):
        return ["tray", "psm1", "stuff"]

    @property
    def random_rgba_range(self,):
        return {
            "default": [[0, 0, 0, 1], [1, 1, 1, 1]],
        }
