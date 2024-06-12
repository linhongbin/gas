from surrol.tasks.gauze_retrieve import GauzeRetrieve
import numpy as np
import pybullet as p
from surrol.utils.pybullet_utils import (
    get_link_pose,
    wrap_angle
)
import os
from gym_ras.tool.common import *


class GauzeRetrieveMod(GauzeRetrieve):
    # WORKSPACE_LIMITS1 = ((0.50, 0.60), (-0.05, 0.05), (0.675, 0.745))
    WORKSPACE_LIMITS1 = ((0.50, 0.60), (-0.05-0.02, 0.05 +
                         0.02), (0.675+0.009-0.009, 0.745))
    # POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))
    # SCALING = 5.
    # haptic=True

    def __init__(self, render_mode=None, cid=-1,
                 fix_goal=True,
                 oracle_pos_thres=5e-3,
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
                 depth_distance=0.25,
                 **kwargs,
                 ):
        self._init_pose_ratio_low_gripper = init_pose_ratio_low_gripper
        self._init_pose_ratio_high_gripper = init_pose_ratio_high_gripper
        self._init_pose_ratio_low_stuff = init_pose_ratio_low_stuff
        self._init_pose_ratio_high_stuff = init_pose_ratio_high_stuff

        self._fix_goal = fix_goal
        super().__init__(render_mode, cid)
        self._view_param = {"distance": depth_distance*self.SCALING,
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
        self._contact_approx = True
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

        # set random stuff init pose
        pos_rel = self._stuff_pose_rng.uniform(
            self._init_pose_ratio_low_stuff, self._init_pose_ratio_high_stuff)
        new_low = np.array([ws[0][0], ws[1][0], ws[2][0] + 0.01, -180])
        new_high = np.array([ws[0][1], ws[1][1], ws[2][1] - 0.01, 180])
        pose = scale_arr(pos_rel, -np.ones(pos_rel.shape),
                         np.ones(pos_rel.shape), new_low, new_high)
        M = Euler2M([0, 0, 0], convension="xyz", degrees=True)
        M1 = Euler2M([0, 0, pose[3]], convension="xyz", degrees=True)
        M2 = np.matmul(M1, M)
        quat = M2Quat(M2)
        pos = pose[:3]
        p.resetBasePositionAndOrientation(self.obj_ids['rigid'][0], pos, quat)

    def reset(self):
        obs = super().reset()
        self._init_stuff_z = self._get_stuff_z()
        self._create_waypoint()
        return obs

    def _create_waypoint(self):
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        orn = p.getEulerFromQuaternion(orn_obj)
        orn_eef = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1]
        orn_eef = p.getEulerFromQuaternion(orn_eef)
        yaw = 0  # does not need yaw
        self._WAYPOINTS = [None] * 5
        self._WAYPOINTS[0] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102 + 0.005) * self.SCALING, yaw, 0.5])  # approach
        self._WAYPOINTS[1] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, yaw, 0.5])  # approach
        self._WAYPOINTS[2] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, yaw, -0.5])  # grasp
        self._WAYPOINTS[3] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102 + 0.005) * self.SCALING, yaw, -0.5])  # lift
        self._WAYPOINTS[4] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102 + 1) * self.SCALING, yaw, -0.5])  # lift

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info["fsm"] = self._fsm()
        return obs, reward, done, info

    def _is_grasp_obj(self):
        return not (self._contact_constraint == None)

    def _get_stuff_z(self):
        stuff_id = self.obj_ids['rigid'][0]
        pos_obj, _ = get_link_pose(stuff_id, -1)
        return pos_obj[2]

    def _fsm(self):
        """ ["in_progress","done_success","done_fail","progress_fail"] """
        obs = self._get_robot_state(idx=0)  # for psm1
        tip = obs[2]
        if self._is_grasp_obj() and (tip-self.workspace_limits1[2][0]) > self._done_z_thres:
            return "done_success"
        stuff_z = self._get_stuff_z()
        # prevent stuff is lifted without grasping or grasping unrealisticly
        if (not self._is_grasp_obj()) and (stuff_z-self._init_stuff_z) > self.SCALING*0.015:
            return "done_fail"
        return "in_progress"

    def action2pos(self, action):
        return action * 0.01 * self.SCALING

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        scale = 0 if self._fix_goal else self.SCALING
        workspace_limits = self.workspace_limits1
        goal = np.array([workspace_limits[0].mean() + 0.01 * np.random.randn() * scale,
                         workspace_limits[1].mean() + 0.01 *
                         np.random.randn() * scale,
                         workspace_limits[2][1] - 0.04 * self.SCALING])
        return goal.copy()

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
            if np.linalg.norm(delta_pos) * 0.01 / scale_factor < self._oracle_pos_thres and np.abs(delta_yaw) < self._oracle_rot_thres:
                self._WAYPOINTS[i] = None
            break
        print(self._WAYPOINTS)
        print(f"oracle #point: {i}")
        print("delta current:", np.linalg.norm(delta_pos) * 0.01)
        print("delta goal:", self._oracle_pos_thres)
        print(np.abs(delta_yaw),  self._oracle_rot_thres)
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
            "psm1": [6, 7],
            "psm1_except_gripper": [3, 4, 5],
            "stuff": [-1],
        }

    @property
    def nodepth_link_ids(self,):
        return {
            "psm1": [6, 7],
        }

    @property
    def nodepth_guess_map(self,):
        return {
            "psm1": "psm1_except_gripper",
        }

    @property
    def nodepth_guess_uncertainty(self,):
        return {
            "psm1": 0.01*self.SCALING,  # 1cm x simluation scaling
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
    def random_color_range(self,):
        return "hsv", {
            "default": [[0.4, 0.4, 0.4, 1], [0.4, 1, 0.4, 1]],
            "tray": [[0, 0.0, 0.7, 1], [1, 0.2, 1, 1]],
            "psm1": [[0.7, 0.0, 0.7, 1], [1, 0.2, 1, 1]],
            "stuff": [[0.0, 0, 0.5, 1], [1, 0.4, 1, 1]],
        }
