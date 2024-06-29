from gym_ras.tool.common import getT
from gym_ras.tool.kdl_tool import *

f = RPY2Frame(1,2,3,4,5,6)
print(f)

t = getT([1,2,3,], [4,5,6], rot_type="euler", euler_Degrees=False)
print(t)
f2  = T2Frame(t)
print(f2)
t2  = Frame2T(f)
print(t2)