from PyKDL import Frame, Rotation, Vector
import numpy as np
def RPY2Frame(x,y, z, R, P, Y):
    return Frame(Rotation.RPY(*[R,P,Y]), Vector(*[x,y,z]))
def Frame2RPY(T: Frame):
    pos = [T.p.x(),T.p.y(),T.p.z()]
    rpy = list(T.M.GetRPY())
    return np.array(pos), np.array(rpy)

#===
def Quaternion2Frame(x,y, z, rx, ry, rz, rw):
    return Frame(Rotation.Quaternion(*[rx, ry, rz, rw]),  Vector(*[x,y,z]))

def Frame2Quaternion(T):
    x = T.p.x()
    y = T.p.y()
    z = T.p.z()
    rx,ry,rz,rw, = T.M.GetQuaternion()
    return [x,y,z, rx,ry,rz,rw]

def Frame2T(T):
    R = T.M
    Rx = R.UnitX()
    Ry = R.UnitY()
    Rz = R.UnitZ()
    t = T.p
    _T = np.array([[Rx[0],Ry[0],Rz[0],t[0]],
                    [Rx[1],Ry[1],Rz[1],t[1]],
                    [Rx[2],Ry[2],Rz[2],t[2]],
                    [0,0,0,1]])
    return _T