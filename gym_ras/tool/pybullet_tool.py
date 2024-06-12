import pybullet as p


def get_obj_links(obj_id):
    v = obj_id
    _link_name_to_index = {p.getBodyInfo(v)[0].decode('UTF-8'): -1, }
    for _id in range(p.getNumJoints(v)):
        _name = p.getJointInfo(v, _id)[12].decode('UTF-8')
        _link_name_to_index[_name] = _id
    return _link_name_to_index
