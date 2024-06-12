from rospy import get_published_topics
import rospy


def ensure_sub_topic(topic_name, raise_exception=True):
    """ wrapper for ros topic name, check if the topic name exists

    raise exception if not exists
    """
    topics = [topic[0] for topic in get_published_topics()]
    result = topic_name in topics
    if (not result) and raise_exception:
        print(topics)
        raise Exception(
            f"topic {topic_name} does not exist, please check if crtk interface is running")
    return topic_name


def safe_init_ros_node(name):
    try:  # avoid multiple call
        _node = rospy.init_node(name)
        return _node
    except:
        return None
