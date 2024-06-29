from gym_ras.env.embodied.dvrk.psm import SinglePSM
from gym_ras.tool.common import TxT, getT
arm = SinglePSM()
print(arm.tip_pose)
arm.reset_pose()
deltaT = getT([0.01, 0, 0], [-40,40,0,], rot_type="euler")
goalT = TxT([arm.tip_pose, deltaT])
arm.moveT(goalT, interp_num=100)


