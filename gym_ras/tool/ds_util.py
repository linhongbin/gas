import rospy
import numpy as np
from ds4_driver.msg import Status, Feedback
import time
from gym_ras.tool.ros_tool import safe_init_ros_node, ensure_sub_topic
''' get msg structure
header: 
  seq: 190264
  stamp: 
    secs: 1698384595
    nsecs: 779674053
  frame_id: "ds4"
axis_left_x: -0.0
axis_left_y: -0.0
axis_right_x: -0.0
axis_right_y: -0.0
axis_l2: 0.0
axis_r2: 0.0
button_dpad_up: 0
button_dpad_down: 0
button_dpad_left: 0
button_dpad_right: 0
button_cross: 0
button_circle: 0
button_square: 0
button_triangle: 0
button_l1: 0
button_l2: 0
button_l3: 0
button_r1: 0
button_r2: 0
button_r3: 0
button_share: 0
button_options: 0
button_trackpad: 0
button_ps: 0
imu: 
  header: 
    seq: 0
    stamp: 
      secs: 1698384595
      nsecs: 779674053
    frame_id: "ds4_imu"
  orientation: 
    x: 0.0
    y: 0.0
    z: 0.0
    w: 0.0
  orientation_covariance: [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  angular_velocity: 
    x: 0.0
    y: -0.0117182664095
    z: -0.00213059389263
  angular_velocity_covariance: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  linear_acceleration: 
    x: 0.447028076792
    y: 9.69264814272
    z: 2.03450573532
  linear_acceleration_covariance: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
battery_percentage: 0.0
battery_full_charging: 0
touch0: 
  id: 31
  active: 0
  x: 0.521625876427
  y: 0.515923559666
touch1: 
  id: 15
  active: 0
  x: 0.94059407711
  y: 0.417197465897
plug_usb: 0
plug_audio: 0
plug_mic: 0

'''


class DS_Controller():
    def __init__(self, sig_keys=None,
                 # reset position signals will reset to zero, we mearsure max(abs(value)) of signals < exit_thres if reset
                 exit_thres=0.3,
                 # signal will set to 1 or -1 when a button or stick is pressed, we mearsure max(abs(value)) of signals > enter_thres if press
                 enter_thres=0.7,
                 wait_hz=-1,
                 only_press=False,
                 ):
        print("init ds contorller...", end="")
        safe_init_ros_node("gym-ras")

        topic_name = '/status'
        topics = [topic[0] for topic in rospy.get_published_topics()]
        if not (topic_name in topics):
            raise Exception(
                "topic {} does not exist, please publisher is running".format(topic_name))
        self._sub = rospy.Subscriber(
            ensure_sub_topic("/status"), Status, self.cb)
        self._pub = rospy.Publisher("/set_feedback", Feedback, queue_size=1)

        self.data = {}
        self.sig_keys = sig_keys or {
            "button_dpad_right": 1,
            "button_dpad_left": 1,
            "button_dpad_up": 1,
            "button_dpad_down": 1,
            "button_cross": 1,
            "button_triangle": 1,
            "button_circle": 1,
            "button_square": 1,
            "button_r2": 1,
            "button_l2": 1,
            "axis_left_x": 2,  # dual direction
        }
        self.exit_thres = exit_thres
        self.enter_thres = enter_thres

        time.sleep(1)
        print("finish")
        self._wait_hz = wait_hz
        self._only_press = only_press

    def cb(self, data):
        for k in self.sig_keys.keys():
            self.data[k] = getattr(data, k)
            # self.data['axis_left_x'] = data.axis_left_x
            # self.data['axis_left_y'] = data.axis_left_y
            # self.data['axis_right_x'] = data.axis_right_x
            # self.data['axis_right_y'] = data.axis_right_y

    def get_discrete_cmd(self, n_max=None):
        sig_keys = list(self.sig_keys.keys())
        if n_max is not None:
            _n = 0
            is_trunc = False
            for i, k in enumerate(sig_keys):
                _n += self.sig_keys[k]
                if _n > n_max:
                    # print("stop ", _n, i)
                    is_trunc = True
                    break
            if is_trunc:
                sig_keys = sig_keys[:i]
        act = 0
        is_reset = True
        start = time.time()
        while is_reset:
            sig = np.array([self.data[k] for k in sig_keys])
            if np.max(np.abs(sig)) > self.enter_thres:
                is_reset = False
                idx = np.argmax(np.abs(sig))
                act = 0
                for i in range(idx):
                    act += self.sig_keys[sig_keys[i]]
                sgn = sig < 0
                act += int(sgn[idx])
            if (time.time()-start) > (1 / self._wait_hz) and self._wait_hz > 0:
                break

        self.rumble_on(big_rum=0.0, small_rum=0.5)
        if not self._only_press:
            while not is_reset:
                sig = np.array([self.data[k] for k in sig_keys])
                if np.max(np.abs(sig)) < self.exit_thres:
                    is_reset = True
                if (time.time()-start) > (1 / self._wait_hz) and self._wait_hz > 0:
                    break
        self.rumble_off()
        # print(act)
        return act

    def rumble_on(self, big_rum=0.5, small_rum=0.5):
        self._assert_0_1(big_rum)
        self._assert_0_1(small_rum)
        msg = Feedback()
        msg.set_rumble = True
        msg.rumble_small = big_rum
        msg.rumble_big = small_rum
        self._pub.publish(msg)

    def rumble_off(self):
        msg = Feedback()
        msg.set_rumble = False
        self._pub.publish(msg)

    def led_on(self, r=0.1, g=0.1, b=0.1):
        self._assert_0_1(r)
        self._assert_0_1(g)
        self._assert_0_1(b)
        msg = Feedback()
        msg.set_led = True
        msg.led_r = float(r)
        msg.led_g = float(g)
        msg.led_b = float(b)
        self._pub.publish(msg)

    def led_off(self):
        self.led_on(r=0.0, g=0.0, b=0.0)

    def _assert_0_1(self, x):
        assert x >= 0 and x <= 1


if __name__ == '__main__':
    con = DS_Controller()
    r = rospy.Rate(1)  # 10hz
    rospy.sleep(1)
    while not rospy.is_shutdown():
        # print(con.data['button_r2'])
        action = con.get_discrete_cmd(9)
        print(action)
        r.sleep()
