from gym_ras.env.embodied.surrol.grasp_any_base import GraspAnyBase
import numpy as np
import pybullet as p
from surrol.utils.pybullet_utils import get_link_pose, wrap_angle
import os
from gym_ras.tool.common import *
from pathlib import Path


class GraspAny(GraspAnyBase):
    WORKSPACE_LIMITS1 = (
        (0.50, 0.60),
        (-0.05 - 0.02, 0.05 + 0.02),
        (0.675 + 0.0088, 0.745),
    )

    def __init__(
        self,
        render_mode=None,
        cid=-1,
        stuff_name="needle",
        fix_goal=True,
        oracle_pos_thres=1e-3,
        oracle_rot_thres=3e-1,
        done_z_thres=0.3,
        init_pose_ratio_low_gripper=[-0.5, -0.5, -0.5, -0.9],
        init_pose_ratio_high_gripper=[0.5, 0.5, 0.5, 0.9],
        init_pose_ratio_low_stuff=[-0.5, -0.5, 0.1, -0.99, -0.99, -0.99],
        init_pose_ratio_high_stuff=[0.5, 0.5, 0.5, 0.99, 0.99, 0.99],
        depth_distance=0.25,
        stuff_scaling_low=0.75,
        stuff_scaling_high=1.5,
        needle_scaling_low=0.3,
        needle_scaling_high=1.0,
        on_plane=False,
        **kwargs,
    ): 
        if not on_plane:
            _z_level = -0.035
            # _z_level = 0.0
            # self.WORKSPACE_LIMITS1 = (
            #     (0.50, 0.60),
            #     (-0.05 - 0.02, 0.05 + 0.02),
            #     (0.675 + 0.0088 + _z_level, 0.745 + 0.01 + _z_level),
            # )
        else:
            _z_level = 0.0025
        self.POSE_TRAY = ((0.55, 0, 0.6751 + _z_level), (0, 0, 0))
        self._init_pose_ratio_low_gripper = init_pose_ratio_low_gripper
        self._init_pose_ratio_high_gripper = init_pose_ratio_high_gripper
        self._init_pose_ratio_low_stuff = init_pose_ratio_low_stuff
        self._init_pose_ratio_high_stuff = init_pose_ratio_high_stuff
        self._stuff_name = stuff_name
        self._stuff_scaling_low = stuff_scaling_low
        self._stuff_scaling_high = stuff_scaling_high
        self._needle_scaling_low = needle_scaling_low
        self._needle_scaling_high = needle_scaling_high
        self._on_plane = on_plane

        self._fix_goal = fix_goal
        super().__init__(render_mode, cid)
        self._view_param = {
            "distance": depth_distance * self.SCALING,
            "yaw": 180,
            "pitch": -45,
            "roll": 0,
        }
        self._proj_param = {"fov": 42, "nearVal": 0.1, "farVal": 1000}
        self._oracle_pos_thres = oracle_pos_thres
        self._oracle_rot_thres = oracle_rot_thres
        self._done_z_thres = done_z_thres

    def _env_setup(self):
        asset_path = (
            Path(__file__).resolve().parent.parent.parent.parent / "asset" / "urdf"
        )

        if self._on_plane:
            file_dir = {
                "needle": [
                    "needle_40mm_RL.urdf",
                ],
                "box": ["bar2.urdf", "bar.urdf", "box.urdf"],
            }
        else:
            file_dir = {
                "needle": [
                    "needle_40mm_RL.urdf",
                ],
                # "box": ["bar2.urdf", "bar.urdf", "box.urdf"],
                # "box": ["bar2.urdf"],
                # "block_haptic.urdf",
            }
        file_dir = {k: [asset_path / k / _v for _v in v] for k, v in file_dir.items()}

        if self._stuff_name == "any":
            _dirs = []
            for _, v in file_dir.items():
                _dirs.extend(v)
        else:
            _dirs = file_dir[self._stuff_name]

        _dir = _dirs[self._stuff_urdf_rng.randint(len(_dirs))]
        _low = (
            self._stuff_scaling_low
            if str(_dir).find("needle") < 0
            else self._needle_scaling_low
        )
        _high = (
            self._stuff_scaling_high
            if str(_dir).find("needle") < 0
            else self._needle_scaling_high
        )
        scaling = self._stuff_urdf_rng.uniform(_low, _high)
        # scaling = 1

        # print("urdf scaling:",scaling, "dir: ",_dir)

        super()._env_setup(stuff_path=str(_dir), scaling=scaling if self._on_plane else 1, on_plane=self._on_plane)
        # workspace_limits = np.asarray(self.WORKSPACE_LIMITS1) \
        #                    + np.array([0., 0., 0.0102]).reshape((3, 1))  # tip-eef offset with collision margin
        # workspace_limits *= self.SCALING  # use scaling for more stable collistion simulation
        # self.workspace_limits1 = workspace_limits
        # print("jjjjdsfsdf")
        # print(self.WORKSPACE_LIMITS1)
        # print(self.workspace_limits1)
        # self.workspace_limits1[2][0] += self.SCALING * 0.003
        # self.workspace_limits1[1][0] = -self.SCALING * 0.07
        # self.workspace_limits1[1][1] = self.SCALING * 0.07
        # print(self.workspace_limits1)

        # set random gripper init pose
        pos_rel = self._gripper_pose_rng.uniform(
            self._init_pose_ratio_low_gripper, self._init_pose_ratio_high_gripper
        )
        ws = self.workspace_limits1
        new_low = np.array([ws[0][0], ws[1][0], ws[2][0], -180])
        new_high = np.array([ws[0][1], ws[1][1], ws[2][1], 180])
        pose = scale_arr(
            pos_rel, -np.ones(pos_rel.shape), np.ones(pos_rel.shape), new_low, new_high
        )
        M = Quat2M([0.5, 0.5, -0.5, -0.5])
        M1 = Euler2M([0, 0, pose[3]], convension="xyz", degrees=True)
        M2 = np.matmul(M1, M)
        quat = M2Quat(M2)
        pos = pose[:3]
        joint_positions = self.psm1.inverse_kinematics(
            (pos, quat), self.psm1.EEF_LINK_INDEX
        )
        self.psm1.reset_joint(joint_positions)

        # set random stuff init pose
        pos_rel = self._stuff_pose_rng.uniform(
            self._init_pose_ratio_low_stuff, self._init_pose_ratio_high_stuff
        )
        _z_level = 0 if self._on_plane else -0.1 
        new_low = np.array([ws[0][0], ws[1][0], ws[2][0] + _z_level, -180, -180, -180])
        new_high = np.array([ws[0][1], ws[1][1], ws[2][0] + _z_level, 180, 180, 180])

        pose = scale_arr(
            pos_rel, -np.ones(pos_rel.shape), np.ones(pos_rel.shape), new_low, new_high
        )
        M = Euler2M([0, 0, 0], convension="xyz", degrees=True)
        if self._on_plane:
            M1 = Euler2M([0, 0, pose[5]], convension="xyz", degrees=True)
        else:
            _m1 = Euler2M([-90, 0, 0], convension="xyz", degrees=True)
            _m2 = Euler2M([0, -60, 0], convension="xyz", degrees=True)
            _m3 = Euler2M([0, 0, pose[5]], convension="xyz", degrees=True)
            M1 = np.matmul(_m2,_m1,)
            M1 = np.matmul(_m3, M1,)
        M2 = np.matmul(M1, M)
        quat = M2Quat(M2)
        pos = pose[:3]
        p.resetBasePositionAndOrientation(self.obj_ids["rigid"][0], pos, quat)

        if not self._on_plane:
            body_pose = p.getBasePositionAndOrientation(self.obj_ids["fixed"][1])
            # body_pose = p.getLinkState(self.obj_ids['fixed'][1], -1)
            obj_pose = p.getLinkState(self.obj_id, self.obj_link1)
            world_to_body = p.invertTransform(body_pose[0], body_pose[1])
            obj_to_body = p.multiplyTransforms(world_to_body[0],
                                            world_to_body[1],
                                            obj_pose[0], obj_pose[1])       
            self._init_stuff_constraint = p.createConstraint(
                parentBodyUniqueId=self.obj_ids["fixed"][1],
                parentLinkIndex=-1,
                childBodyUniqueId=self.obj_id,
                childLinkIndex=self.obj_link1,
                jointType=p.JOINT_FIXED,
                jointAxis=(0, 0, 0),
                parentFramePosition=obj_to_body[0],
                parentFrameOrientation=obj_to_body[1],
                childFramePosition=(0, 0, 0),
                childFrameOrientation=(0, 0, 0),
            )

            p.changeConstraint(self._init_stuff_constraint, maxForce=20)

    def reset(self):
        obs = super().reset()
        self._init_stuff_z = self._get_stuff_z()
        self._create_waypoint()
        return obs

    def _create_waypoint(self):
        for _ in range(100):
            p.stepSimulation()  # wait stable
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1 if self._on_plane  else 2)
        if not self._on_plane:
            pos_obj[2] += 0.05
        orn = p.getEulerFromQuaternion(orn_obj)
        orn_eef = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1]
        orn_eef = p.getEulerFromQuaternion(orn_eef)
        yaw = (
            orn[2]
            if abs(wrap_angle(orn[2] - orn_eef[2]))
            < abs(wrap_angle(orn[2] + np.pi - orn_eef[2]))
            else wrap_angle(orn[2] + np.pi)
        )  #
        self._WAYPOINTS = [None] * 5
        self._WAYPOINTS[0] = np.array(
            [
                pos_obj[0],
                pos_obj[1],
                pos_obj[2] + (-0.0007 + 0.0102 + 0.005) * self.SCALING,
                yaw,
                0.5,
            ]
        )  # approach
        self._WAYPOINTS[1] = np.array(
            [
                pos_obj[0],
                pos_obj[1],
                pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING,
                yaw,
                0.5,
            ]
        )  # approach
        self._WAYPOINTS[2] = np.array(
            [
                pos_obj[0],
                pos_obj[1],
                pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING,
                yaw,
                -0.5,
            ]
        )  # grasp
        self._WAYPOINTS[3] = np.array(
            [
                pos_obj[0],
                pos_obj[1],
                pos_obj[2] + (-0.0007 + 0.0102 + 0.005) * self.SCALING,
                yaw,
                -0.5,
            ]
        )  # lift
        self._WAYPOINTS[4] = np.array(
            [
                pos_obj[0],
                pos_obj[1],
                pos_obj[2] + (-0.0007 + 0.0102 + 1) * self.SCALING,
                yaw,
                -0.5,
            ]
        )  # lift

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info["fsm"] = self._fsm()
        return obs, reward, done, info

    def _is_grasp_obj(self):
        return not (self._contact_constraint == None)

    def _get_stuff_z(self):
        stuff_id = self.obj_ids["rigid"][0]
        pos_obj, _ = get_link_pose(stuff_id, -1)
        return pos_obj[2]

    def _fsm(self):
        if self._on_plane:
            obs = self._get_robot_state(idx=0)  # for psm1
            tip = obs[2]
            if (
                self._is_grasp_obj()
                and (tip - self.workspace_limits1[2][0]) > self._done_z_thres
            ):
                return "done_success"
            stuff_z = self._get_stuff_z()
            # prevent stuff is lifted without grasping or grasping unrealisticly
            if (not self._is_grasp_obj()) and (
                stuff_z - self._init_stuff_z
            ) > self.SCALING * 0.015:
                return "done_fail"

            return "prog_norm"
        else:
            if self._init_stuff_constraint is not None and self._contact_constraint is None:
                return "prog_norm"
            elif self._init_stuff_constraint is None and self._contact_constraint is not None:
                return "done_success"
            else:
                return "done_fail"
    def _sample_goal(self) -> np.ndarray:
        """Samples a new goal and returns it."""
        scale = 0 if self._fix_goal else self.SCALING
        workspace_limits = self.workspace_limits1
        goal = np.array(
            [
                workspace_limits[0].mean() + 0.01 * np.random.randn() * scale,
                workspace_limits[1].mean() + 0.01 * np.random.randn() * scale,
                workspace_limits[2][1] - 0.04 * self.SCALING,
            ]
        )
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
            delta_pos = (waypoint[:3] - obs["observation"][:3]) / 0.01 / self.SCALING
            delta_yaw = (waypoint[3] - obs["observation"][5]).clip(-0.4, 0.4)
            if np.abs(delta_pos).max() > 1:
                delta_pos /= np.abs(delta_pos).max()
            scale_factor = 0.4
            delta_pos *= scale_factor
            action = np.array(
                [delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, waypoint[4]]
            )
            if (
                np.linalg.norm(delta_pos) * 0.01 / scale_factor < self._oracle_pos_thres
                and np.abs(delta_yaw) < self._oracle_rot_thres
            ):
                self._WAYPOINTS[i] = None
            break

        return action

    def seed(self, seed=0):
        super().seed(seed)
        self._init_rng(seed)

    def _init_rng(self, seed):
        stuff_pose_seed = np.uint32(seed)
        gripper_pose_seed = np.uint32(seed - 1)
        print("stuff_pose_seed:", stuff_pose_seed)
        print("gripper_pose_seed:", gripper_pose_seed)
        self._stuff_pose_rng = np.random.RandomState(stuff_pose_seed)
        self._gripper_pose_rng = np.random.RandomState(
            gripper_pose_seed
        )  # use different seed
        self._stuff_urdf_rng = np.random.RandomState(stuff_pose_seed)

    @property
    def keyobj_ids(
        self,
    ):
        # from gym_ras.tool.pybullet_tool import get_obj_links
        # print(get_obj_links(self.psm1.body))
        return {
            "psm1": self.psm1.body,
            "stuff": self.obj_id,
        }

    @property
    def keyobj_link_ids(
        self,
    ):
        return {
            "psm1": [6, 7],
            "psm1_except_gripper": [3, 4, 5],
            "stuff": [-1],
        }

    @property
    def nodepth_link_ids(
        self,
    ):
        return {
            "psm1": [6, 7],
        }

    @property
    def nodepth_guess_map(
        self,
    ):
        return {
            "psm1": "psm1_except_gripper",
        }

    @property
    def nodepth_guess_uncertainty(
        self,
    ):
        return {
            "psm1": 0.01 * self.SCALING,  # 1cm x simluation scaling
        }

    @property
    def background_obj_ids(
        self,
    ):
        return {
            "tray": self.obj_ids["fixed"][1],
        }

    @property
    def background_obj_link_ids(
        self,
    ):
        return {
            "tray": [-1, 4],
        }

    @property
    def random_vis_key(self):
        return ["tray", "psm1", "stuff"]

    # @property
    # def random_color_range(self,):
    #     return "rgb", {
    #         "default": [[0.4,0.4,0.4,1], [1,1,1,1]],
    #         "tray": [[0.7,0.7,0.7,1], [1,1,1,1]],
    #     }

    @property
    def random_color_range(
        self,
    ):
        return "hsv", {
            "default": [[0.4, 0.4, 0.4, 1], [0.4, 1, 0.4, 1]],
            "tray": [[0, 0.0, 0.7, 1], [1, 0.2, 1, 1]],
            "psm1": [[0.7, 0.0, 0.7, 1], [1, 0.2, 1, 1]],
            "stuff": [[0.0, 0, 0.5, 1], [1, 0.4, 1, 1]],
        }
