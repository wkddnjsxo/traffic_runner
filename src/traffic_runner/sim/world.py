"""
MORAI 시뮬레이터 제어 브리지.

실행 중인 시뮬에 재시작 없이 attach 해서 쓴다 (start() 를 부르면 시뮬/UDP 네트워크가
초기화되어 사용자가 세팅해 둔 것이 날아간다).

담당:
  - ego 상태 읽기 / 텔레포트 / 속도·조향 제어
  - 날씨 / 시간 설정
  - 신호등 상태 설정 (tl/controller.py 가 이걸 쓴다)
"""

import math
import os
import sys
import time


class World(object):
    def __init__(self, cfg, client_key=None):
        self.cfg = cfg
        g = cfg["grpc"]
        self.host = g["host"]
        self.port = int(g["port"])
        self.client_key = client_key or g["client_key"]
        self.map_name = cfg["morai"]["map_name"]
        self.grpc_src = os.path.abspath(cfg["paths"]["grpc_src"])

        self._add_paths()
        self._connect()

    # ------------------------------------------------------------------ setup
    def _add_paths(self):
        for p in (self.grpc_src,
                  os.path.join(self.grpc_src, "api"),
                  os.path.join(self.grpc_src, "proto")):
            if not os.path.isdir(p):
                raise RuntimeError("gRPC SDK 경로가 없다: %s" % p)
            if p not in sys.path:
                sys.path.append(p)

    def _connect(self):
        from api.morai_sim_client import MoraiSimClient
        from proto.morai.simulation.simulation_enum_pb2 import SYNC_MODE_TYPE_UNSPECIFIED
        from simulation_world import SimulationWorld

        # SimulationWorld 생성 시 서버 MGeo 를 받아오지 않게 한다 (느리고 불필요).
        try:
            from api.map import Map

            if not getattr(Map, "_tr_mgeo_disabled", False):
                Map.get_mgeo_data = lambda self_, m: setattr(self_, "mgeo_data", {})
                Map._tr_mgeo_disabled = True
        except Exception:
            pass

        self.client = MoraiSimClient(self.client_key)
        self.client.connect(self.host, self.port)
        if not self.client.is_connected():
            raise RuntimeError("MORAI gRPC 연결 실패: %s:%d" % (self.host, self.port))

        import grpc

        try:
            grpc.channel_ready_future(self.client._sim_adapter._channel).result(timeout=3.0)
        except Exception as exc:
            self.client.disconnect()
            raise RuntimeError(
                "MORAI gRPC 서버(%s:%d)에 닿지 않는다. MORAI 실행 / gRPC 서버 ON / "
                "runtime.yaml 의 grpc.host 를 확인할 것." % (self.host, self.port)) from exc

        # attach: start() 를 부르지 않는다. SimulationWorld.__init__ 가 내부에서
        # destroy_all_actors() 를 호출하므로 그 동안만 무력화한다 (기존 액터 보존).
        _orig = SimulationWorld.destroy_all_actors
        SimulationWorld.destroy_all_actors = lambda s, remove_all=False: None
        try:
            self.world = SimulationWorld(
                self.client._sim_adapter, self.client_key, self.map_name,
                SYNC_MODE_TYPE_UNSPECIFIED, "Ego", "Ego")
        finally:
            SimulationWorld.destroy_all_actors = _orig

        self.client._simulation_world = self.world
        self.ego = self.world.get_ego()
        self.adapter = self.client._sim_adapter
        print("[world] attach 완료 (%s:%d, map=%s)" % (self.host, self.port, self.map_name))

    # -------------------------------------------------------------- ego 상태
    def ego_state(self):
        state = self.ego.get_actor_state()
        if state is None:
            return None
        v = state.velocity
        speed_kmh = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        vs = state.vehicle_state
        return {
            "x": float(state.transform.location.x),
            "y": float(state.transform.location.y),
            "z": float(state.transform.location.z),
            "yaw_deg": float(state.transform.rotation.z),
            "speed_mps": speed_kmh / 3.6,
            "speed_kmh": speed_kmh,
            "link_id": str(vs.current_link_info.id.value),
            "tl_id": str(getattr(vs.tl_id, "value", "") or ""),
            "tl_color": int(getattr(vs, "tl_color", 0) or 0),
            "collision_objects": [str(c) for c in getattr(vs, "collision_objects", [])],
        }

    def make_transform(self, x, y, z, yaw_deg):
        from proto.morai.common.type_pb2 import Transform

        tf = Transform()
        tf.location.x = float(x)
        tf.location.y = float(y)
        tf.location.z = float(z)
        tf.rotation.x = 0.0
        tf.rotation.y = 0.0
        tf.rotation.z = float(yaw_deg)
        return tf

    def teleport_ego(self, x, y, z, yaw_deg, settle_sec=0.5):
        """ego 를 지정 pose 로 옮기고 정지시킨다."""
        tf = self.make_transform(x, y, z, yaw_deg)
        ok = self.ego.set_transform(tf)
        try:
            self.ego.set_velocity(0.0)
        except Exception:
            pass
        if settle_sec > 0:
            time.sleep(settle_sec)
        return ok

    # -------------------------------------------------------------- ego 제어
    def set_manual_control(self):
        """
        외부(우리)가 속도/조향을 직접 지령할 수 있는 모드로 바꾼다.

        MORAI 내장 cruise 가 켜져 있으면 우리 지령과 싸우므로 반드시 꺼야 한다.
        """
        from proto.morai.actor.actor_enum_pb2 import VEHICLE_CONTROL_AUTO_MODE

        ok = False
        try:
            ok = self.ego.set_control_mode(VEHICLE_CONTROL_AUTO_MODE)
        except Exception as exc:
            print("[world] set_control_mode 실패: %s" % exc)
        try:
            self.ego.set_cruise_mode(False)
        except Exception:
            pass
        return ok

    def drive(self, speed_mps, steer_rad):
        """
        목표 속도(m/s)와 조향각(rad)을 지령한다.

        LONG_CMD_TYPE_SPEED 를 쓰면 종방향은 시뮬레이터가 알아서 맞춘다
        (throttle/brake PID 를 우리가 만들 필요가 없다).
        """
        from proto.morai.actor.actor_enum_pb2 import LONG_CMD_TYPE_SPEED

        return self.ego.control(
            long_cmd_type=LONG_CMD_TYPE_SPEED,
            throttle=0.0, brake=0.0,
            steer=float(steer_rad),
            velocity=float(speed_mps) * 3.6,   # MORAI 는 km/h 로 받는다
            acceleration=0.0, frame=0)

    def brake(self, amount=1.0, steer_rad=0.0):
        """
        브레이크를 건다.

        ★ LONG_CMD_TYPE_SPEED 로는 브레이크가 안 걸린다 ★
        속도 모드에서 목표 속도를 낮추면 시뮬레이터는 스로틀만 놓고 타력 주행한다
        (실측: 감속도 0.7m/s² — 끝점 1m 앞에서 4.6m/s 로 지나침).
        실제로 세우려면 스로틀/브레이크 모드로 브레이크 값을 직접 넣어야 한다.
        """
        from proto.morai.actor.actor_enum_pb2 import LONG_CMD_TYPE_THROTTLE

        try:
            return self.ego.control(
                long_cmd_type=LONG_CMD_TYPE_THROTTLE,
                throttle=0.0, brake=float(max(0.0, min(1.0, amount))),
                steer=float(steer_rad), velocity=0.0, acceleration=0.0, frame=0)
        except Exception:
            return False

    def brake_stop(self):
        return self.brake(1.0, 0.0)

    # ------------------------------------------------------------ 환경 설정
    _WEATHER = {
        "SUNNY": "WEATHER_TYPE_SUNNY",
        "CLOUDY": "WEATHER_TYPE_CLOUDY",
        "RAINY": "WEATHER_TYPE_RAINY",
        "SNOWY": "WEATHER_TYPE_SNOWY",
        "FOGGY": "WEATHER_TYPE_FOGGY",
    }

    def set_weather(self, name):
        from proto.morai.environment import environment_enum_pb2 as ee

        key = str(name).upper()
        enum_name = self._WEATHER.get(key)
        if enum_name is None or not hasattr(ee, enum_name):
            # 서버가 아는 이름 중에서 찾아본다
            avail = [n for n in dir(ee) if n.startswith("WEATHER_TYPE_")]
            raise ValueError("알 수 없는 날씨 '%s'. 가능: %s" % (name, avail))
        self.world.set_weather(getattr(ee, enum_name))
        return True

    def set_time_hour(self, hour):
        self.world.set_time(int(hour))
        return True

    def get_env(self):
        try:
            w = self.world.get_weather()
            h = self.world.get_time()
            return {"weather": w, "hour": h}
        except Exception:
            return {"weather": None, "hour": None}

    # ------------------------------------------------------------ 신호등
    def set_traffic_light(self, tl_id, color_value, impulse=False, sibling=True):
        """
        신호등 색을 설정한다.

        impulse=False (영구) 여야 주행 내내 유지된다. True 는 일시적이라
        시뮬 자체 신호 스케줄에 곧 덮어쓰인다.
        """
        from proto.morai.infrastructure.traffic_light_pb2 import TrafficLightStateParam
        from proto.morai.common.enum_pb2 import STATUS_CODE_SUCCESS

        param = TrafficLightStateParam()
        param.info.id.value = str(tl_id)
        param.info.color = int(color_value)
        param.is_impulse = bool(impulse)
        param.set_sibling = bool(sibling)

        result = self.adapter.set_traffic_light_state(param)
        return getattr(result, "status", None) == STATUS_CODE_SUCCESS

    def close(self):
        try:
            self.client.disconnect()
        except Exception:
            pass
