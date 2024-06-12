from pathlib import Path
import numpy as np


class RGBD_CAM():
    def __init__(self,
                 device="rs435",
                 image_height=600,
                 image_width=600,
                 depth_remap_center=None,
                 depth_remap_range=None,
                 segment_tool="",
                 segment_model_dir="",
                 mask_noisy_link=True,
                 ):
        self._image_height = image_height
        self._image_width = image_width
        self._depth_remap_center = depth_remap_center
        self._depth_remap_range = depth_remap_range
        self._mask_noisy_link = mask_noisy_link
        if device == "rs435":
            from gym_ras.tool.rs435 import RS435_ROS_Engine
            self._device = RS435_ROS_Engine(image_height=image_height,
                                            image_width=image_width,
                                            depth_remap_center=depth_remap_center,
                                            depth_remap_range=depth_remap_range,)
        else:
            raise NotImplementedError

        if segment_tool == "detectron":
            from gym_ras.tool.seg_tool import DetectronPredictor
            assert segment_model_dir != ""
            _dir = Path(segment_model_dir)
            self._segment = DetectronPredictor(cfg_dir=str(_dir / "segment.yaml"),
                                               model_dir=str(_dir / "model_best.pth"),)
        elif segment_tool == "color":
            from gym_ras.tool.seg_tool import ColorObjSegmentor

            _dir = Path(segment_model_dir)
            self._segment = ColorObjSegmentor(segment_model_dir)
        elif segment_tool == "track_any":
            from gym_ras.tool.seg_tool import TrackAnySegmentor
            self._segment = TrackAnySegmentor()
            self._device._segmentor = self._segment
        else:
            raise NotImplementedError()

    def _process_depth(self, depth, mask_dict):
        _depth = depth.copy()
        if self._mask_noisy_link:
            for k in self._no_depth_link:  # mask out no depth link
                if k in mask_dict:
                    _depth[mask_dict[k]] = 0

        # for v in self.depth_process_link: # mask out anomalous depth
        #     _proc_mask = [mask_dict[k] for k in list(v) if k in mask_dict]
        #     _mask = _proc_mask[0] if _proc_mask == 1 else self._merge_masks(_proc_mask)
        #     _median = np.median(_depth[_mask])
        #     _range = 30
        #     _range_h = int(_range/2)
        #     _mask_1 = _depth < (_median -  _range_h)
        #     _mask_2 = _depth > (_median +  _range_h)
        #     _depth[np.logical_and(_mask,_mask_1)] = 0
        #     _depth[np.logical_and(_mask,_mask_2)] = 0

        return np.uint8(_depth)

    def render(self,):
        img = self._device.get_image()
        masks = img['mask']
        img["mask"] = {}
        if self._segment is not None:
            # masks = self._segment.predict(img['rgb'])

            _mask_dict = {}
            if len(masks) > 0:
                _mask_dict.update(
                    {self._segment_id_map[k]: v[0] for k, v in masks.items()})
            img['depth'] = self._process_depth(img['depth'], _mask_dict)

            _dsa_mask_dict = {}
            for k, v in self._env_mask_mapping.items():
                _mask = None
                for j in v:
                    if j in _mask_dict:
                        _mask = _mask_dict[j] if _mask is None else np.logical_or(
                            _mask, _mask_dict[j])
                if _mask is not None:
                    _dsa_mask_dict[k] = _mask
            img['mask'] = _dsa_mask_dict
        return img

    def _merge_masks(self, masks):
        _mask = None
        for v in masks:
            _mask = v if _mask is None else np.logical_or(_mask, v)
        return _mask

    @property
    def _segment_id_map(self,):
        return {0: "needle", 1: "gripper_tip", 3: "gripper_link", 2: "gripper_base", }

    @property
    def _env_mask_mapping(self,):
        return {"psm1": ["gripper_tip"], "stuff": ["needle"], "psm1_except_gripper": ["gripper_link", "gripper_base"]}

    # @property
    # def depth_process_link(self,):
    #     return [("gripper_base", "gripper_link")]

    @property
    def _no_depth_link(self, ):
        return ["gripper_tip"]
