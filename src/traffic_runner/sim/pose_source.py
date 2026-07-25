"""
Ego pose 소스.

지점 캡처는 사람이 MORAI 에서 직접 몰면서 진행하므로, 실행 중인 시뮬레이션을
'재시작 없이' 붙어서 ego 위치만 읽어야 한다. start 를 호출하면 시뮬/UDP 네트워크가
초기화되어 운전 중인 세션이 끊긴다.

두 소스를 지원한다.
  grpc : gRPC 로 attach 해서 ActorState 폴링. ROS 브리지 설정이 필요 없다(기본값).
  ros  : /Ego_topic (morai_msgs/EgoVehicleStatus) 구독. 이미 ROS 브리지를 켜 뒀다면 더 가볍다.

둘 다 dict {x, y, z, yaw_deg, speed_mps} 를 돌려준다.
"""

import math
import os
import sys


class PoseSource(object):
    def read(self):
        raise NotImplementedError

    def close(self):
        pass


# --------------------------------------------------------------------- gRPC
class GrpcPoseSource(PoseSource):
    """
    실행 중인 MORAI 시뮬레이션에 attach 해서 ego 상태를 읽는다.

    auto_ws 의 grpc_inha_univ SDK(api/, proto/)를 sys.path 에 얹어 직접 사용한다.
    auto_scenario_runner 의 파이썬 모듈(utils/, core/ 등)에는 의존하지 않는다
    (같은 이름의 패키지가 있어 import 가 섞이면 곤란하기 때문).
    """

    def __init__(self, grpc_src, host, port, client_key, map_name):
        self.grpc_src = os.path.abspath(grpc_src)
        self.host = host
        self.port = int(port)
        self.client_key = client_key
        self.map_name = map_name

        self._add_paths()
        self._connect()

    def _add_paths(self):
        for p in (self.grpc_src,
                  os.path.join(self.grpc_src, "api"),
                  os.path.join(self.grpc_src, "proto")):
            if not os.path.isdir(p):
                raise RuntimeError(
                    "grpc SDK 경로가 없다: %s\n"
                    "config/runtime.yaml 의 paths.grpc_src 를 확인할 것." % p
                )
            if p not in sys.path:
                sys.path.append(p)

    def _connect(self):
        from api.morai_sim_client import MoraiSimClient
        from proto.morai.simulation.simulation_enum_pb2 import SYNC_MODE_TYPE_UNSPECIFIED
        from simulation_world import SimulationWorld

        # SimulationWorld 생성 시 서버 MGeo 를 받아오지 않게 한다(캡처엔 불필요, 느림).
        try:
            from api.map import Map

            if not getattr(Map, "_tr_mgeo_disabled", False):
                def _skip(self_, map_name):
                    self_.mgeo_data = {}
                Map.get_mgeo_data = _skip
                Map._tr_mgeo_disabled = True
        except Exception:
            pass

        self.client = MoraiSimClient(self.client_key)
        self.client.connect(self.host, self.port)
        if not self.client.is_connected():
            raise RuntimeError("MORAI gRPC 연결 실패: %s:%d" % (self.host, self.port))

        try:
            import grpc

            grpc.channel_ready_future(self.client._sim_adapter._channel).result(timeout=3.0)
        except Exception as exc:
            self.client.disconnect()
            raise RuntimeError(
                "MORAI gRPC 서버(%s:%d)에 닿지 않는다. MORAI 실행 여부, gRPC 서버 ON, "
                "runtime.yaml 의 grpc.host/port 를 확인할 것." % (self.host, self.port)
            ) from exc

        # attach: start() 를 호출하지 않는다. SimulationWorld.__init__ 가 내부에서
        # destroy_all_actors() 를 부르므로 그 동안만 무력화한다(사용자가 깔아둔 액터 보존).
        _orig = SimulationWorld.destroy_all_actors
        SimulationWorld.destroy_all_actors = lambda self_, remove_all=False: None
        try:
            self.world = SimulationWorld(
                self.client._sim_adapter, self.client_key, self.map_name,
                SYNC_MODE_TYPE_UNSPECIFIED, "Ego", "Ego",
            )
        finally:
            SimulationWorld.destroy_all_actors = _orig

        self.client._simulation_world = self.world
        self.ego = self.world.get_ego()
        print("[pose] gRPC attach 완료 (%s:%d, map=%s, 재시작 없음)"
              % (self.host, self.port, self.map_name))

    def read(self):
        state = self.ego.get_actor_state()
        if state is None:
            return None
        # ActorState.velocity 는 로컬프레임 km/h (auto_ws 에서 실측 확인된 사항)
        v = state.velocity
        speed_kmh = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
        return {
            "x": float(state.transform.location.x),
            "y": float(state.transform.location.y),
            "z": float(state.transform.location.z),
            "yaw_deg": float(state.transform.rotation.z),
            "speed_mps": speed_kmh / 3.6,
            "link_id": str(state.vehicle_state.current_link_info.id.value),
            "tl_id": str(getattr(state.vehicle_state.tl_id, "value", "")),
            "tl_color": int(getattr(state.vehicle_state, "tl_color", 0) or 0),
        }

    def close(self):
        try:
            self.client.disconnect()
        except Exception:
            pass


# ---------------------------------------------------------------------- ROS
class RosPoseSource(PoseSource):
    """/Ego_topic (morai_msgs/EgoVehicleStatus) 구독. 최신 메시지를 read() 로 돌려준다."""

    def __init__(self, topic="/Ego_topic", node_name="tr_spot_capture", wait_sec=5.0):
        import rospy
        from morai_msgs.msg import EgoVehicleStatus

        self._rospy = rospy
        self._latest = None

        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=True, disable_signals=True)
        self._sub = rospy.Subscriber(topic, EgoVehicleStatus, self._cb, queue_size=1)

        print("[pose] ROS %s 대기 중..." % topic)
        deadline = rospy.Time.now() + rospy.Duration(wait_sec) if not rospy.is_shutdown() else None
        rate = rospy.Rate(20)
        while self._latest is None and not rospy.is_shutdown():
            if deadline is not None and rospy.Time.now() > deadline:
                raise RuntimeError(
                    "%s 에서 메시지를 못 받았다. MORAI 네트워크 설정(ROS bridge)과 "
                    "토픽 이름을 확인할 것." % topic
                )
            rate.sleep()
        print("[pose] ROS 구독 시작: %s" % topic)

    def _cb(self, msg):
        self._latest = msg

    def read(self):
        msg = self._latest
        if msg is None:
            return None
        vel = msg.velocity
        return {
            "x": float(msg.position.x),
            "y": float(msg.position.y),
            "z": float(msg.position.z),
            # EgoVehicleStatus.heading 은 도 단위 (MORAI 기준 z-yaw 와 동일 정의)
            "yaw_deg": float(msg.heading),
            "speed_mps": math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2),
            "link_id": "",
            "tl_id": "",
            "tl_color": 0,
        }

    def close(self):
        try:
            self._sub.unregister()
        except Exception:
            pass


def create(cfg, source=None):
    """runtime.yaml 설정으로 pose 소스를 만든다."""
    src = (source or cfg.get("capture", {}).get("pose_source", "grpc")).lower()
    if src == "grpc":
        g = cfg["grpc"]
        return GrpcPoseSource(
            grpc_src=cfg["paths"]["grpc_src"],
            host=g["host"], port=g["port"], client_key=g["client_key"],
            map_name=cfg["morai"]["map_name"],
        )
    if src == "ros":
        return RosPoseSource(topic=cfg.get("ros", {}).get("ego_topic", "/Ego_topic"))
    raise ValueError("pose_source 는 grpc | ros 중 하나여야 함 (현재 %r)" % src)
