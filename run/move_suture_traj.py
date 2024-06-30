import argparse
import dvrk
import numpy as np
from gym_ras.tool.common import TxT, invT, getT, T2Quat, scale_arr, M2Euler, wrapAngle, Euler2M, printT, Quat2M, M2Quat
from gym_ras.tool.kdl_tool import Frame2T, Quaternion2Frame
from gym_ras.tool.config import load_yaml
from gym_ras.env.embodied.dvrk.psm import SinglePSM
from time import sleep


parser = argparse.ArgumentParser()
parser.add_argument('--arm', type=str, default='PSM1') 
parser.add_argument('--calfile', type=str, default='./data/dvrk_cal/dvrk_suture_cal.yaml') 
parser.add_argument('--armfile', type=str, default='./data/dvrk_cal/dvrk_cal.yaml') 
parser.add_argument('--needle-d', type=float, default=0.02) 
parser.add_argument('--theta-entryoffset', type=float, default=160) 
parser.add_argument('--theta-exitoffset', type=float, default=60) 
parser.add_argument('--insert-interp', type=int, default=50) 

args = parser.parse_args()




psm_args = {}

add_args = load_yaml(args.armfile)
psm_args.update(add_args)

arm = SinglePSM(**psm_args)



cal_dict = load_yaml(args.calfile)


# grasping pose
T = np.zeros((4,4), dtype=np.float)
T[0,0] = -1; T[1,1] = 1; T[2,2] = -1; T[3,3] = 1;
T[0,3] = cal_dict['entry_point'][0]; T[1,3] = cal_dict['entry_point'][1]; T[2,3] = cal_dict['entry_point'][2] + 0.01;
grasp_T = T
print(grasp_T)
arm.moveT_local(T)

arm.open_gripper()
from gym_ras.tool.ds_util import DS_Controller
in_device = DS_Controller()
in_device.led_on(g=0.8,b=0.8)
cmd = in_device.get_discrete_cmd()
in_device.led_off()
arm.close_gripper()
sleep(1)


# theta 
entry_p = np.array(cal_dict['entry_point'])
exit_p = np.array(cal_dict['exit_point'])
e_e_dis = np.linalg.norm(entry_p[:2] - exit_p[:2])
needle_d = args.needle_d
theta = np.arcsin(e_e_dis / needle_d)
center_p = (entry_p + exit_p)/2
center_p[2] = center_p[2] + (needle_d /2 * np.cos(theta))
print("theta: ", np.rad2deg(theta))
print("e_e_dis: ", e_e_dis)



# entry pose
entry_T = grasp_T.copy()
entry_T[:3,3] = center_p
deltaT = getT([0,0,0], [-90, 0,0], rot_type="euler")
deltaT2 = getT([0,0,0], [0, 0, theta + np.deg2rad(args.theta_entryoffset)], rot_type="euler",euler_Degrees=False)
deltaT3 = getT([0,-needle_d/2, 0], [0, 0, 0], rot_type="euler")
entry_T = TxT([entry_T,deltaT,deltaT2, deltaT3])
arm.moveT_local(entry_T, interp_num=10)


# # exit pose
# exit_T = grasp_T.copy()
# exit_T[:3,3] = exit_p
# deltaT = getT([0,0,0], [-90, 0,0], rot_type="euler")
# # deltaT2 = getT([0,0,0], [0, 0, 10], rot_type="euler")
# deltaT2 = getT([0,0,0], [0, 0, -theta + np.deg2rad(args.theta_entryoffset)], rot_type="euler",euler_Degrees=False)
# exit_T = TxT([exit_T,deltaT,deltaT2])
# arm.moveT_local(exit_T, interp_num=10)

insert_Ts = []
interp_theta = np.linspace(theta + np.deg2rad(args.theta_entryoffset), 
                           -theta + np.deg2rad(args.theta_entryoffset) - np.deg2rad(args.theta_exitoffset) , 
                           num=args.insert_interp).tolist()
print(interp_theta)
for k, th in enumerate(interp_theta):
    _T = grasp_T.copy()
    _T[:3,3] = center_p
    deltaT = getT([0,0,0], [-90, 0,0], rot_type="euler")
    deltaT2 = getT([0,0,0], [0, 0, th], rot_type="euler",euler_Degrees=False)
    deltaT3 = getT([0,-needle_d/2, 0], [0, 0, 0], rot_type="euler")
    _T = TxT([_T,deltaT,deltaT2, deltaT3])
    insert_Ts.append(_T)


for _T in insert_Ts:
    arm.moveT_local(_T)