import argparse
import dvrk
from gym_ras.tool.ds_util import DS_Controller
from gym_ras.tool.kdl_tool import Frame2T, T2Frame
import numpy as np
import yaml
from pathlib import Path
from PyKDL import Rotation

parser = argparse.ArgumentParser()
parser.add_argument('--arm', type=str, default='PSM1') 
parser.add_argument('--savedir', type=str, default='./data/dvrk_cal') 
args = parser.parse_args()

assert args.arm in ["PSM1", "PSM2"]
arm1 = dvrk.psm(args.arm)
f = arm1.measured_cp()
T1 = Frame2T(f)
T = np.zeros((4,4), dtype=np.float)
T[0,0] = -1; T[1,1] = 1; T[2,2] = -1; T[3,3] = 1;
T1[0:3,0:3] = T[0:3,0:3]
arm1.move_cp(T2Frame(T1)).wait()
arm1.jaw.close().wait()
in_device = DS_Controller()

def get_pose():
    cmd = in_device.get_discrete_cmd()
    # print("pos:",client.T_g_w_msr.p)
    return arm1.measured_cp()

needle_d = 0.02 # 20mm

data_dict = {}


in_device.led_on(g=0.8,b=0.8)
print("move to suturing entry port")
v1 = get_pose()
in_device.led_on(r=0.8,b=0.8)
print("move to suturing exit port")
v2 = get_pose()
in_device.led_off()
t1 = Frame2T(v1)
t2 = Frame2T(v2)
print(t1)
print(t2)
entry_point = t1[0:3,3]
exit_point = t2[0:3,3]
print(entry_point)
print(exit_point)
data_dict["entry_point"] = entry_point.tolist()
data_dict["exit_point"] = exit_point.tolist()
with open(str(Path(args.savedir) / 'dvrk_suture_cal.yaml'), 'w') as outfile:
    yaml.dump(data_dict, outfile, default_flow_style=False)
# port_d = np.linalg.norm(exit_point - entry_point)
# circle_origin_x = (entry_point[0] + exit_point[0]) /2
# circle_origin_y = (entry_point[1] + exit_point[1]) /2
# circle_origin_z = np.sqrt((needle_d/2) ** 2 - (port_d/2) ** 2)
# circle_origin = np.array([circle_origin_x, circle_origin_y, circle_origin_z])

