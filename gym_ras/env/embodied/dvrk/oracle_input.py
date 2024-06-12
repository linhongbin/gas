import numpy as np


class OracleInput():
    def __init__(self, device="ds4"):
        if device == "ds4":
            from gym_ras.tool.ds_util import DS_Controller
            self._device = DS_Controller()
        else:
            raise NotImplementedError

        self.out_act_prv = np.zeros(5)
        self.gripper_open_prv = True

    def get_oracle_action(self,):
        action = self._device.get_discrete_cmd(n_max=9)
        map_action = [np.zeros(5) for i in range(9)]
        for i in range(8):
            map_action[i][4] = 1 if self.gripper_open_prv else -1
            map_action[i][i//2] = 1 if (i % 2) == 1 else -1
        map_action[8][4] = -1 if self.gripper_open_prv else 1
        out_act = np.array(map_action[action])
        print(out_act)
        print(map_action)
        self.gripper_open_prv = out_act[4] > 0
        return out_act
