import argparse
import dvrk
from gym_ras.tool.ds_util import DS_Controller
import numpy as np
import yaml
import pathlib
from PyKDL import Rotation

parser = argparse.ArgumentParser()
# parser.add_argument('--obs', type=str, default='rgb') # [rgb, depth]
parser.add_argument('--arm', type=str, default='PSM1') 

# parser.add_argument('--origin-x-ratio', type=float, default=0.5) 
# parser.add_argument('--origin-y-ratio', type=float, default=0.5) 
# parser.add_argument('--origin-z-ratio', type=float, default=0.8)
# parser.add_argument('--gripper-initmargin', type=float, default=0.01)
# parser.add_argument('--needle-initmargin', type=float, default=0.02)
# parser.add_argument('--needle-z-initratio', type=float, default=0.3) 
parser.add_argument('--savedir', type=str, default='./data/dvrk_cal')
args = parser.parse_args()

assert args.arm in ["PSM1", "PSM2"]
arm1 = dvrk.psm(args.arm)
in_device = DS_Controller()

def get_pose():
    cmd = in_device.get_discrete_cmd()
    # print("pos:",client.T_g_w_msr.p)
    return arm1.measured_cp()



Ts = {}


in_device.led_on(g=0.8,b=0.8)
print("move to upward right limit")
v1 = get_pose()
print("move to downward right limit")
v2 = get_pose()
delta = v1.p -v2.p
if np.abs(delta.y())>1e-16:
    theta = np.arctan(delta.x()/delta.y()) 
else:
    theta = np.arctan(delta.x()*np.sign(delta.y()) * 1e10) 

theta += np.deg2rad(90-180) # align with simulation setting
print(f"delta theta is {theta} rad, {np.rad2deg(theta)} degree")
# in_device.led_on(b=1)
in_device.led_off()

Ts['world2base_yaw'] = np.rad2deg(theta).tolist()


rot = Rotation.RotZ(theta)




in_device.led_on(g=1)
print("move to left limit")
v1 = rot*(get_pose().p)
print("move to right limit")
v2 = rot*(get_pose().p)
Ts['ws_y'] =sorted([v1.y(), v2.y()])
print("y:", v1, v2)


print(f"Drag {args.arm} to following position in floating mode....")
print("move to forward limit")
in_device.led_on(r=1)
v1 = rot*(get_pose().p)
print("move to backward limit")
v2 = rot*(get_pose().p)
print("x:", v1, v2)
Ts['ws_x'] = sorted([v1.x(), v2.x()])


in_device.led_on(b=1)
print("move to upward limit")
v1 = get_pose()
print("move to downward limit")
v2 = get_pose()
Ts['ws_z'] =sorted([v1.p.z(), v2.p.z()])




in_device.led_on(g=0.8,r=0.8)
print("move to reset pose")
cmd = in_device.get_discrete_cmd()
qs = arm1.measured_jp()
Ts['reset_q'] = qs[:7].tolist()
in_device.led_off()

# ratio_func = lambda x,y, rate: x*(1-rate)+y*rate
# print("=======get results==========")
# print("\"ws_x_low\":{:.3f},".format(Ts['ws_x'][0]))
# print("\"ws_x_high\":{:.3f},".format(Ts['ws_x'][1]))
# print("\"ws_y_low\":{:.3f},".format(Ts['ws_y'][0]))
# print("\"ws_y_high\":{:.3f},".format(Ts['ws_y'][1]))
# print("\"ws_z_low\":{:.6f},".format(Ts['ws_z'][0]))
# print("\"ws_z_high\":{:.3f},".format(Ts['ws_z'][1]))
# init_origin_pos = [ratio_func(Ts['ws_x'][0],Ts['ws_x'][1], args.origin_x_ratio),
#                    ratio_func(Ts['ws_y'][0],Ts['ws_y'][1], args.origin_y_ratio),
#                    ratio_func(Ts['ws_z'][0],Ts['ws_z'][1], args.origin_z_ratio),]

savedir = pathlib.Path(args.savedir)
savedir.mkdir(parents=True, exist_ok=True)
with open(str(savedir / 'dvrk_cal.yaml'), 'w') as outfile:
    yaml.dump(Ts, outfile, default_flow_style=False)

# print()
# print("\"init_orgin_RPY\":[{:.2f},{:.2f}, {:.2f}, np.pi, 0, np.pi/2],".format(*init_origin_pos))

# print()
# print(f"\"manual_set_base_rpy\":[0,0,0, 0,0,{theta}],")

# init_origin_pose = init_origin_pos
# init_origin_pose.extend([np.pi, 0, np.pi/2])


# qs_reset = client.ik_map(RPY2T(*init_origin_pose))
# print()
# print("\"q_dsr_reset\":[{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},],".format(*qs_reset))


# print()
# print("\"init_pos_bound_dict\":{{ 'x':[{:.3f}, {:.3f}],'y':[{:.3f}, {:.3f}],'z':[{:.3f}, {:.3f}]}},".format(\
# Ts['ws_x'][0]+args.gripper_initmargin,
# Ts['ws_x'][1]-args.gripper_initmargin,
# Ts['ws_y'][0]+args.gripper_initmargin,
# Ts['ws_y'][1]-args.gripper_initmargin,
# init_origin_pos[2],
# init_origin_pos[2],
# ))

# print()
# init_z_needle = ratio_func(Ts['ws_z'][0],Ts['ws_z'][1],args.needle_z_initratio)
# print("\"needle_init_pose_bound\":{{ 'low':[{:.3f}, {:.3f},{:.3f}, 0,0, -np.pi,],\n              'high':[{:.3f}, {:.3f},{:.3f}, 0,0., np.pi,],}},".format(\
# Ts['ws_x'][0]+args.needle_initmargin,
# Ts['ws_y'][0]+args.needle_initmargin,
# init_z_needle,
# Ts['ws_x'][1]-args.needle_initmargin,
# Ts['ws_y'][1]-args.needle_initmargin,
# init_z_needle
# ))
# print("==================================")
