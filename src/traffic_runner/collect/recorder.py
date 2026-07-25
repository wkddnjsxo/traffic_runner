"""
카메라 프레임 저장 + 매니페스트 기록.

MORAI 가 /image_jpeg/compressed (sensor_msgs/CompressedImage) 로 발행하는 데이터는
이미 JPEG 바이트다. 그대로 파일에 쓰면 되므로 cv_bridge / OpenCV 가 필요 없고,
morai_msgs 도 안 쓰므로 catkin 빌드 없이 /opt/ros/noetic 소싱만으로 동작한다.

라벨 정확성을 위해 지키는 것:
  - 신호 연출이 **완료된 시각 이후에 도착한** 프레임만 저장한다. 연출 직전에
    찍힌 프레임에는 이전 신호가 남아 있어 라벨이 틀린다.
  - 같은 메시지를 두 번 저장하지 않는다 (10Hz 카메라를 20Hz 제어루프에서
    읽으므로 중복이 생긴다).
"""

import os
import threading
import time


MANIFEST_HEADER = [
    "image_path", "label", "label_index", "spot_id", "kind",
    "weather", "hour", "object_seed", "state", "frame_idx",
    "dist_to_end_m", "dist_to_tl_m", "ego_x", "ego_y", "ego_yaw", "speed_mps",
    "tl_id", "tl_color_observed", "objects", "run_id", "stamp",
]


class CameraRecorder(object):
    """
    ROS 카메라 토픽을 구독해 최신 프레임을 들고 있다가, 요청 시 파일로 쓴다.

    구독은 백그라운드에서 계속 돌고, 저장 여부는 호출자가 결정한다
    (주행 중에만, 연출 완료 이후에만 저장하기 위해).
    """

    def __init__(self, topic="/image_jpeg/compressed", node_name="tr_recorder",
                 wait_sec=10.0):
        import rospy
        from sensor_msgs.msg import CompressedImage

        self._rospy = rospy
        self._lock = threading.Lock()
        self._msg = None
        self._msg_stamp = 0.0     # 수신 시각 (walltime)
        self._seq = 0
        self._last_saved_seq = -1
        self.frames_received = 0

        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=True, disable_signals=True)
        self.topic = topic
        self._sub = rospy.Subscriber(topic, CompressedImage, self._cb, queue_size=2)

        print("[recorder] %s 구독, 첫 프레임 대기..." % topic)
        deadline = time.time() + wait_sec
        while self._msg is None:
            if time.time() > deadline:
                raise RuntimeError(
                    "%s 에서 프레임을 못 받았다.\n"
                    "  확인: MORAI 센서 Network Setting 이 ROS 로 Connect 되어 있는가,\n"
                    "        rostopic hz %s 가 나오는가." % (topic, topic))
            time.sleep(0.05)
        print("[recorder] 수신 시작 (%d bytes/frame)" % len(self._msg.data))

    def _cb(self, msg):
        with self._lock:
            self._msg = msg
            self._msg_stamp = time.time()
            self._seq += 1

    def latest(self, not_before=None, require_fresh=True):
        """
        최신 프레임을 (bytes, seq, stamp) 로 준다. 조건에 안 맞으면 None.

        not_before      : 이 시각 이후에 도착한 프레임만 (신호 연출 완료 시각)
        require_fresh   : 이미 저장한 프레임은 다시 주지 않는다
        """
        with self._lock:
            if self._msg is None:
                return None
            if require_fresh and self._seq == self._last_saved_seq:
                return None
            if not_before is not None and self._msg_stamp < not_before:
                return None
            return bytes(self._msg.data), self._seq, self._msg_stamp

    def mark_saved(self, seq):
        with self._lock:
            self._last_saved_seq = seq

    def close(self):
        try:
            self._sub.unregister()
        except Exception:
            pass


class DatasetWriter(object):
    """
    이미지 파일 + manifest.csv 를 쓴다.

    디렉터리 구조:
      dataset/images/<weather>_<hour>/<spot_id>/seed<N>/<state>/000000.jpg
    """

    def __init__(self, root, run_id):
        self.root = os.path.abspath(root)
        self.run_id = run_id
        self.images_dir = os.path.join(self.root, "images")
        self.manifest_path = os.path.join(self.root, "manifest.csv")
        self.count = 0

        if not os.path.isdir(self.images_dir):
            os.makedirs(self.images_dir)

        # 이어쓰기: 헤더는 파일이 새로 생길 때만
        new_file = not os.path.exists(self.manifest_path)
        self._fp = open(self.manifest_path, "a", buffering=1)
        if new_file:
            self._fp.write(",".join(MANIFEST_HEADER) + "\n")

    def dir_for(self, weather, hour, spot_id, seed, state):
        d = os.path.join(self.images_dir, "%s_%s" % (weather, hour),
                         spot_id, "seed%02d" % seed, state)
        if not os.path.isdir(d):
            os.makedirs(d)
        return d

    def save(self, jpeg_bytes, meta):
        """
        프레임 하나를 저장하고 매니페스트에 한 줄 쓴다.

        meta 는 MANIFEST_HEADER 의 image_path 를 뺀 나머지 키를 담는다.
        """
        d = self.dir_for(meta["weather"], meta["hour"], meta["spot_id"],
                         meta["object_seed"], meta["state"])
        fname = "%06d.jpg" % meta["frame_idx"]
        path = os.path.join(d, fname)
        with open(path, "wb") as f:
            f.write(jpeg_bytes)

        rel = os.path.relpath(path, self.root)
        row = [rel] + [_csv(meta.get(k, "")) for k in MANIFEST_HEADER[1:]]
        self._fp.write(",".join(row) + "\n")
        self.count += 1
        return path

    def close(self):
        try:
            self._fp.close()
        except Exception:
            pass


def _csv(v):
    """CSV 한 칸. 쉼표/따옴표가 있으면 감싼다."""
    s = str(v)
    if "," in s or '"' in s or "\n" in s:
        return '"%s"' % s.replace('"', '""')
    return s
