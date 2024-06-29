import argparse
import dvrk
import numpy as np
from gym_ras.tool.common import TxT, invT, getT, T2Quat, scale_arr, M2Euler, wrapAngle, Euler2M, printT, Quat2M, M2Quat
from gym_ras.tool.kdl_tool import Frame2T, Quaternion2Frame
from gym_ras.tool.config import load_yaml
from gym_ras.env.embodied.dvrk.psm import SinglePSM


parser = argparse.ArgumentParser()
parser.add_argument('--arm', type=str, default='PSM1') 
parser.add_argument('--calfile', type=str, default='./data/dvrk_cal/dvrk_suture_cal.yaml') 
parser.add_argument('--armfile', type=str, default='./data/dvrk_cal/dvrk_cal.yaml') 
args = parser.parse_args()


psm_args = {}

add_args = load_yaml(args.armfile)
psm_args.update(add_args)

arm = SinglePSM(**psm_args)
arm.close_gripper()
cal_dict = load_yaml(args.calfile)
print(cal_dict)
T = np.zeros((4,4), dtype=np.float)
T[0,0] = -1; T[1,1] = 1; T[2,2] = -1; T[3,3] = 1;
T[0,3] = cal_dict['entry_point'][0]; T[1,3] = cal_dict['entry_point'][1]; T[2,3] = cal_dict['entry_point'][2];
print(T)
arm.moveT_local(T)

deltaT = getT([0,0,0], [-45, 0,0], rot_type="euler")
T1 = TxT([T,deltaT])
arm.moveT_local(T1)
