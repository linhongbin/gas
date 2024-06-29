from PyKDL import Frame, Rotation, Vector
import numpy as np



def RPY2Frame(x, y, z, R, P, Y):
    return Frame(Rotation.RPY(*[R, P, Y]), Vector(*[x, y, z]))


def Frame2RPY(T: Frame):
    pos = [T.p.x(), T.p.y(), T.p.z()]
    rpy = list(T.M.GetRPY())
    return np.array(pos), np.array(rpy)

# ===


def Quaternion2Frame(x, y, z, rx, ry, rz, rw):
    return Frame(Rotation.Quaternion(*[rx, ry, rz, rw]),  Vector(*[x, y, z]))


def Frame2Quaternion(T):
    x = T.p.x()
    y = T.p.y()
    z = T.p.z()
    rx, ry, rz, rw, = T.M.GetQuaternion()
    return [x, y, z, rx, ry, rz, rw]


def Frame2T(T):
    R = T.M
    Rx = R.UnitX()
    Ry = R.UnitY()
    Rz = R.UnitZ()
    t = T.p
    _T = np.array([[Rx[0], Ry[0], Rz[0], t[0]],
                   [Rx[1], Ry[1], Rz[1], t[1]],
                   [Rx[2], Ry[2], Rz[2], t[2]],
                   [0, 0, 0, 1]])
    return _T

def T2Frame(T):
    x = Vector(T[0,0],T[1,0],T[2,0])
    y = Vector(T[0,1],T[1,1],T[2,1])
    z = Vector(T[0,2],T[1,2],T[2,2])
    p = Vector(T[0,3],T[1,3],T[2,3])
    M = Rotation(x,y,z)
    f = Frame(M, p)
    return f

def gen_interpolate_frames(T_orgin, T_dsr, num):
    """ generate interpolate frames """
    # print("xx:", T_orgin, T_dsr)
    T_delta =  T_orgin.Inverse()* T_dsr
    angle, axis = T_delta.M.GetRotAngle()
    # print("origin", T_orgin)
    # print("goal", T_dsr)
    # print("angle", angle, "axis", axis)
    # print("deltaT", T_delta)
    return [T_orgin * Frame(Rotation.Rot(axis, angle*alpha), alpha*T_delta.p)  for alpha in np.linspace(0, 1,num=num).tolist()]