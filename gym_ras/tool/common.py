import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R
import pandas as pd


def scale_arr(_input, old_min, old_max, new_min, new_max):
    _in = _input
    _in = np.divide(_input-old_min, old_max-old_min)
    _in = np.multiply(_in, new_max-new_min) + new_min
    return _in


def invT(T):
    return inv(T)


def TxT(T_list):
    T = T_list[0]
    for _T in T_list[1:]:
        T = np.matmul(T, _T)
    return T


def getT(pos_list, rot_list, rot_type="quaternion", euler_convension="xyz", euler_Degrees=True):
    if rot_type == "quaternion":
        M = Quat2M(rot_list)
    elif rot_type == "euler":
        M = Euler2M(rot_list, convension=euler_convension,
                    degrees=euler_Degrees)
    else:
        raise NotImplementedError
    T = np.zeros((4, 4))
    T[0:3, 3] = np.array(pos_list)
    T[0:3, 0:3] = M
    T[3, 3] = 1
    return T


def T2Quat(T):
    M = T[0:3, 0:3]
    p = T[0:3, 3]
    quat = M2Quat(M)
    return p, quat


def T2Euler(T, convension="xyz", degrees=True):
    M = T[0:3, 0:3]
    p = T[0:3, 3]
    euler = M2Euler(M, convension=convension, degrees=degrees)
    return p, euler


def M2Quat(M):
    rot = R.from_matrix(M)
    return rot.as_quat()


def M2Euler(M, convension="xyz", degrees=True):
    rot = R.from_matrix(M)
    return rot.as_euler(convension, degrees=degrees)


def Quat2M(in_list):
    r = R.from_quat(in_list)
    return r.as_matrix()


def Euler2M(in_list, convension="xyz", degrees=True):
    M = R.from_euler(convension, in_list, degrees=degrees)
    return M.as_matrix()


def Euler2Quat(in_list, convension="xyz", degrees=True):
    """
    convention can be "xyz" "zyx" "zxz" etc
    """
    r = R.from_euler(convension, in_list, degrees=degrees)
    return r.as_quat()


def Quat2Euler(in_list, convension="xyz", degrees=True):
    """ x,y,z,w convension"""
    r = R.from_quat(in_list)
    return r.as_euler(convension, degrees)


def wrapAngle(_in, degrees=True, angle_range=180.0):
    '''wrap to angle -180 to 180'''
    scale = angle_range if degrees else angle_range/180 * np.pi
    return (_in + scale) % (2 * scale) - scale


def printT(T, prefix_string=None):
    pos, rpy = T2Euler(T, convension="xyz", degrees=True)
    _, quat = T2Quat(T)
    _prefix_string = prefix_string if prefix_string is not None else ""
    print(f"{_prefix_string}: \n pos: {pos}, \n rpy: {rpy}, \n quat: {quat}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')

    parser.add_argument('-p', type=int)
    args = parser.parse_args()
    if args.p == 1:
        M = Quat2M([0.3826834, 0, 0, 0.9238795])
        print(M)
# #         should be
# #         [  1.0000000,  0.0000000,  0.0000000;
# #    0.0000000,  0.7071068, -0.7071068;
# #    0.0000000,  0.7071068,  0.7071068 ]

        q = Euler2Quat([np.pi, 0, 0])
        print(q)

        euler = Quat2Euler([0, 0, 0, 1])
        print(euler)

        # T = getT([0,1,2],[ 0.3826834, 0, 0, 0.9238795 ],rot_type="quaternion" )
        # print(T)
        # T = getT([0,1,2],[np.pi/4,0,0],rot_type="euler" )
        # print(T)
        # print("inverseT", invT(T))


def ema(in_list, period):
    values = np.array(in_list)
    columns = [str(v) for v in range(len(in_list[0]))]
    return pd.DataFrame.ewm(pd.DataFrame(values, columns=columns), span=period).mean().values.tolist()
