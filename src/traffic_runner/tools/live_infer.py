#!/usr/bin/env python3
"""
실시간 신호등 인지 — 카메라 프레임을 추론 서버로 보내 결과를 본다.

카메라(ROS)는 WSL, 모델은 컨테이너에 있다(WSL 은 Python 3.8 이라 cu128 torch 를
못 깐다). 그래서 이 스크립트가 ROS 에서 프레임을 받아 TCP 로 서버에 넘긴다.

시뮬레이터에 gRPC 로도 붙어서 **실제 신호색(정답)** 을 함께 읽는다. 예측과
정답을 나란히 보여주므로 실주행 정확도가 그 자리에서 측정된다.

  # 컨테이너에서 서버부터 띄운다
  cd ~/traffic_runner/train && ./run.sh serve --ckpt runs/<시각>/best.pt

  # 그다음 WSL 에서
  source /opt/ros/noetic/setup.bash
  python3 tools/live_infer.py

  python3 tools/live_infer.py --no-truth     # 정답 비교 없이 예측만
  python3 tools/live_infer.py --hz 5         # 초당 5회만 추론
"""

import argparse
import json
import os
import socket
import struct
import sys
import time

import yaml

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS_ROOT = os.path.dirname(os.path.dirname(PKG_ROOT))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from tl import states as tl_states   # noqa: E402


def color_name(value):
    """MORAI tl_color 값 -> 우리 클래스 이름."""
    for st in tl_states.STATES.values():
        if st.morai_value == value:
            return st.name
    return {0: "unspecified", -1: "off", -2: "not_detected"}.get(value, str(value))


class InferClient(object):
    def __init__(self, host, port, timeout=5.0):
        self.addr = (host, port)
        self.timeout = timeout
        self.sock = None
        self.connect()

    def connect(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect(self.addr)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock = s
        print("[client] 추론 서버 연결 %s:%d" % self.addr)

    def predict(self, jpeg):
        self.sock.sendall(struct.pack(">I", len(jpeg)) + jpeg)
        head = self._recv(4)
        (n,) = struct.unpack(">I", head)
        return json.loads(self._recv(n).decode())

    def _recv(self, n):
        buf = b""
        while len(buf) < n:
            c = self.sock.recv(n - len(buf))
            if not c:
                raise ConnectionError("서버 연결 끊김")
            buf += c
        return buf

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="실시간 신호등 인지",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--config", default=os.path.join(PKG_ROOT, "config", "runtime.yaml"))
    ap.add_argument("--source", choices=["ros", "udp"], default="ros",
                    help="카메라 소스. ros=개발(토픽 구독), udp=대회(MORAI UDP)")
    ap.add_argument("--udp-ip", default="0.0.0.0",
                    help="카메라 UDP 수신 IP (보통 0.0.0.0)")
    ap.add_argument("--udp-port", type=int, default=9090,
                    help="카메라 UDP 포트 (MORAI 센서 Destination Port. 기본 9090)")
    ap.add_argument("--host", default="127.0.0.1", help="추론 서버 주소")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--hz", type=float, default=10.0, help="초당 추론 횟수 상한")
    ap.add_argument("--no-truth", action="store_true",
                    help="시뮬 정답(gRPC) 비교를 끈다 (예측만 출력)")
    ap.add_argument("--compete", action="store_true",
                    help="대회 모드: gRPC 연결 안 함(신호 토픽 없는 맵), 예측만 크게 출력. "
                         "--no-truth 를 켜고 출력을 대회용으로 바꾼다")
    ap.add_argument("--min-prob", type=float, default=0.0,
                    help="예측 확률이 이보다 낮으면 uncertain 으로 표시 (0=끄기)")
    ap.add_argument("--topk", action="store_true", help="상위 후보도 표시")
    # ---- 신호 순환 (특정 클래스 쌍의 혼동을 재려고 쓴다) ----
    ap.add_argument("--cycle", default=None, metavar="STATES",
                    help="이 신호 상태들을 번갈아 연출한다. "
                         "예: green,green_left  (green↔green_left 혼동 측정용)")
    ap.add_argument("--cycle-sec", type=float, default=4.0,
                    help="한 상태를 유지할 시간(s)")
    ap.add_argument("--spot", default=None,
                    help="이 지점의 신호등 전부를 제어 (생략하면 ego 가 감지한 것 하나)")
    ap.add_argument("--spots-dir", default=os.path.join(WS_ROOT, "spots"))
    ap.add_argument("--guard-sec", type=float, default=0.8,
                    help="신호 전환 직후 이 시간은 비교에서 제외한다. 카메라 지연 때문에 "
                         "전환 직후 프레임에는 이전 신호가 찍혀 있다")
    args = ap.parse_args(argv)

    if args.compete:
        args.no_truth = True
        if args.source == "ros" and "--source" not in (argv or sys.argv):
            args.source = "udp"   # 대회 기본은 UDP

    cycle_states = None
    if args.cycle:
        cycle_states = [x.strip() for x in args.cycle.split(",") if x.strip()]
        for st in cycle_states:
            tl_states.get(st)
        if len(cycle_states) < 1:
            print("--cycle 에 상태를 하나 이상 줄 것.")
            return 1

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    for k in ("grpc_src", "mgeo_dir"):
        p = cfg["paths"].get(k)
        if p and not os.path.isabs(p):
            cfg["paths"][k] = os.path.normpath(os.path.join(cfg_dir, p))

    # ---- 카메라 소스: UDP(대회) 또는 ROS(개발) ----
    latest = {"jpeg": None, "seq": 0}
    udp = None

    if args.source == "udp":
        # 대회: MORAI 카메라 UDP. 별도 스레드가 프레임을 조립한다.
        from sim.camera_udp import CameraUDP

        udp = CameraUDP(ip=args.udp_ip, port=args.udp_port)
        print("[udp] %s:%d 수신 대기 (대회 카메라)..." % (args.udp_ip, args.udp_port))
        first = udp.wait_first(timeout=10.0)
        if first is None:
            print("UDP 프레임을 못 받았다. MORAI 카메라 UDP 설정과 포트(%d)를 확인할 것."
                  % args.udp_port)
            return 1
        print("[udp] 수신 시작 (%d bytes)" % first)
    else:
        # 개발: ROS /image_jpeg/compressed
        import rospy
        from sensor_msgs.msg import CompressedImage

        topic = cfg.get("ros", {}).get("image_topic", "/image_jpeg/compressed")

        def cb(msg):
            latest["jpeg"] = bytes(msg.data)
            latest["seq"] += 1

        rospy.init_node("tr_live_infer", anonymous=True, disable_signals=True)
        rospy.Subscriber(topic, CompressedImage, cb, queue_size=1)
        print("[ros] %s 구독, 첫 프레임 대기..." % topic)
        t0 = time.time()
        while latest["jpeg"] is None:
            if time.time() - t0 > 10:
                print("프레임을 못 받았다. MORAI 센서 ROS 연결을 확인할 것.")
                return 1
            time.sleep(0.05)
        print("[ros] 수신 시작 (%d bytes)" % len(latest["jpeg"]))

    # ---- 시뮬 정답 (선택) ----
    world = None
    if not args.no_truth:
        try:
            from sim.world import World
            world = World(cfg)
            print("[truth] gRPC 연결 — 실제 신호색과 비교한다")
        except Exception as exc:
            print("[truth] gRPC 연결 실패, 예측만 표시: %s" % str(exc)[:80])

    # ---- 신호 순환 준비 ----
    tlc = tl_ids = None
    if cycle_states:
        if world is None:
            print("신호 순환에는 gRPC 연결이 필요하다 (--no-truth 와 같이 못 쓴다).")
            return 1
        from tl.controller import TrafficLightController

        tlc = TrafficLightController(world, settle_sec=0.0)
        if args.spot:
            from spot import schema

            sp = schema.load(os.path.join(args.spots_dir, "%s.yaml" % args.spot),
                             strict=False)
            tl_ids = sp.tl_ids
            print("[cycle] %s 의 신호등 %s" % (args.spot, ", ".join(tl_ids)))
        else:
            st = world.ego_state()
            tid = (st or {}).get("tl_id", "")
            if not tid or tid in ("Not Detected", "None"):
                print("ego 가 신호등을 감지하지 못했다. 신호등 앞으로 이동하거나 "
                      "--spot sig_XXX 로 지정할 것.")
                return 1
            tl_ids = [tid]
            print("[cycle] ego 가 감지한 신호등 %s" % tid)
        print("[cycle] %s 를 %.1f초마다 번갈아 연출 (전환 후 %.1f초는 비교 제외)"
              % (" ↔ ".join(cycle_states), args.cycle_sec, args.guard_sec))

    client = InferClient(args.host, args.port)

    period = 1.0 / max(args.hz, 0.5)
    n = ok = compared = skipped_guard = 0
    last_seq = -1
    conf_pairs = {}
    cyc_i = -1
    cyc_at = 0.0
    per_state = {}
    pred_counts = {}
    print("\nCtrl-C 로 종료\n")
    try:
        while True:
            loop = time.time()

            # ---- 신호 순환 ----
            if cycle_states and loop - cyc_at >= args.cycle_sec:
                cyc_i = (cyc_i + 1) % len(cycle_states)
                want = cycle_states[cyc_i]
                okset, info = tlc.apply(tl_ids, want, verify=False)
                cyc_at = loop
                if not okset:
                    print("\n[cycle] 연출 실패: %s" % info["reason"])

            # UDP 소스면 최신 프레임을 udp 에서 당겨온다 (같은 latest 인터페이스)
            if udp is not None:
                j, seq = udp.latest()
                latest["jpeg"], latest["seq"] = j, seq

            if latest["seq"] == last_seq:      # 새 프레임이 없으면 건너뛴다
                time.sleep(0.005)
                continue
            last_seq = latest["seq"]
            jpeg = latest["jpeg"]

            try:
                r = client.predict(jpeg)
            except (ConnectionError, socket.timeout) as exc:
                print("\n서버 연결 문제: %s — 재연결 시도" % exc)
                time.sleep(1.0)
                try:
                    client.connect()
                except Exception:
                    pass
                continue
            n += 1
            pred_counts[r["label"]] = pred_counts.get(r["label"], 0) + 1

            truth = None
            if world is not None:
                st = world.ego_state()
                if st is not None:
                    truth = color_name(st["tl_color"])

            if args.compete:
                # 대회용: 신호 상태를 크게, 확신도 낮으면 표시
                label = r["label"].upper()
                if args.min_prob and r["prob"] < args.min_prob:
                    label = "UNCERTAIN(%s)" % r["label"]
                line = "  ▶ %-14s  %.0f%%   (%.1fms)" % (label, r["prob"] * 100, r["ms"])
                if args.topk:
                    line += "   [" + " ".join("%s %.0f%%" % (a, b * 100)
                                              for a, b in r["top"][1:]) + "]"
            else:
                line = "  예측 %-11s %.3f  (%.1fms)" % (r["label"], r["prob"], r["ms"])
                if args.topk:
                    line += "  [" + " ".join("%s:%.2f" % (a, b)
                                             for a, b in r["top"][1:]) + "]"
            in_guard = bool(cycle_states) and (time.time() - cyc_at) < args.guard_sec
            if truth is not None:
                if in_guard:
                    skipped_guard += 1
                    line += "  | 정답 %-11s (전환 직후, 비교 제외)" % truth
                elif truth in ("not_detected", "unspecified", "off"):
                    line += "  | 정답 %-13s (미감지 구간)" % truth
                else:
                    compared += 1
                    per_state.setdefault(truth, [0, 0])
                    per_state[truth][1] += 1
                    if truth == r["label"]:
                        per_state[truth][0] += 1
                    hit = (truth == r["label"])
                    ok += hit
                    key = (truth, r["label"])
                    conf_pairs[key] = conf_pairs.get(key, 0) + 1
                    line += "  | 정답 %-11s %s  누적 %.1f%% (%d/%d)" % (
                        truth, "✔" if hit else "✘", 100.0 * ok / compared,
                        ok, compared)
            sys.stdout.write("\r\033[K" + line)
            sys.stdout.flush()

            sleep = period - (time.time() - loop)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\n\n=== 요약 ===")
        print("추론 %d회" % n)
        if args.compete and pred_counts:
            print("\n예측 분포:")
            for lab, c in sorted(pred_counts.items(), key=lambda x: -x[1]):
                print("  %-12s %6d회 (%.1f%%)" % (lab, c, 100.0 * c / n))
        if skipped_guard:
            print("전환 직후 제외 %d회 (카메라 지연 구간)" % skipped_guard)
        if per_state:
            print("\n실제 신호별 정확도:")
            for st in sorted(per_state):
                hit, tot = per_state[st]
                print("  %-12s %6.2f%%  (%d/%d)" % (st, 100.0 * hit / tot, hit, tot))
        if compared:
            print("정답 비교 %d회 중 %d회 일치 — 정확도 %.2f%%"
                  % (compared, ok, 100.0 * ok / compared))
            wrong = sorted(((v, k) for k, v in conf_pairs.items() if k[0] != k[1]),
                           reverse=True)
            if wrong:
                print("\n틀린 조합:")
                for v, (t, p) in wrong[:10]:
                    print("  정답 %-11s → 예측 %-11s %d회" % (t, p, v))
            else:
                print("틀린 프레임 없음.")
    finally:
        client.close()
        if udp is not None:
            udp.close()
        if world is not None:
            world.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
