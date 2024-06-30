from gym_ras.env.embodied.dvrk.psm import SinglePSM
from gym_ras.tool.common import TxT, getT
arm = SinglePSM()
arm.close_gripper()
t = arm.tip_pose
print("start:", t)
deltaT = getT([0,0,0], [-45,0,0], rot_type="euler")
t1 = TxT([deltaT,t])
t1[0:3,3] = t[0:3,3]
print("end:", t1)
arm.moveT(t1, interp_num=50)





