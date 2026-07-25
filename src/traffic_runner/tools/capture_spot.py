#!/usr/bin/env python3
"""
지점(spot) 캡처 툴 — 시작점 / 끝점 / 그 사이 주행 경로를 손으로 찍어 저장한다.

사용 흐름
---------
1. MORAI 를 켜고 맵을 로드한 뒤, ego 를 수집하고 싶은 교차로 접근로 앞에 둔다.
2. 이 툴을 실행한다 (실행 중인 시뮬에 재시작 없이 attach 한다).
3. 시작점에 차를 세우고 [s] → 신호등이 보이기 시작하는 지점부터 녹화가 시작된다.
4. 그대로 손으로 몰아서 교차로를 통과한다. 궤적이 자동으로 기록된다.
   (이 궤적이 나중에 pure pursuit 이 추종할 기준 경로가 된다)
5. 끝점에서 [e] → 신호등 ID / 좌회전 유무 등을 묻고 spots/<spot_id>.yaml 로 저장한다.
6. 3~5 를 지점 수만큼 반복. 신호등이 없는 unknown 구간은 [k] 로 kind 를 바꾼 뒤 캡처.

키
--
  s  시작점 마킹 & 녹화 시작        e  끝점 마킹 & 저장
  x  현재 녹화 취소                m  현재 위치를 경로점으로 강제 추가
  t  신호등 ID 입력/수정            l  좌회전 화살표 유무 토글
  k  kind 토글 (signal/unknown)     n  메모 입력
  p  현재 상태 출력                 L  저장된 지점 목록
  q  종료
"""

import argparse
import math
import os
import sys
import time
from datetime import datetime

import yaml

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/traffic_runner
WS_ROOT = os.path.dirname(os.path.dirname(PKG_ROOT))                    # traffic_runner
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from sim import pose_source                      # noqa: E402
from spot import schema                          # noqa: E402
from tl import states as tl_states               # noqa: E402
from utils.geometry import angle_diff_deg, dist2d, path_length  # noqa: E402
from utils.keyboard import KeyReader             # noqa: E402


HELP = """
[s] 시작점&녹화  [e] 끝점&저장  [x] 취소  [m] 점 추가
[t] 신호등ID  [l] 좌회전유무  [k] kind  [n] 메모
[p] 상태  [L] 목록  [?] 도움말  [q] 종료
"""


#: 시뮬이 "신호등 못 찾음" 을 나타낼 때 쓰는 tl_id 값들
_NO_TL = ("", "Not Detected", "None", "null", "-1")


def _color_name(value):
    """tl_color 정수 -> 사람이 읽는 이름. 우리 클래스가 아니어도 값 그대로 보여준다."""
    if value is None:
        return "-"
    for st in tl_states.STATES.values():
        if st.morai_value == value:
            return st.name
    return {0: "unspecified", -1: "off", -2: "not_detected"}.get(value, str(value))


def load_cfg(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # grpc_src 는 runtime.yaml 기준 상대경로 허용
    grpc_src = cfg["paths"]["grpc_src"]
    if not os.path.isabs(grpc_src):
        cfg["paths"]["grpc_src"] = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(path)), grpc_src)
        )
    return cfg


class Recording(object):
    """녹화 중인 하나의 지점."""

    def __init__(self, start_pose, kind, tl_ids, has_left, note):
        self.start = dict(start_pose)
        self.kind = kind
        self.tl_ids = list(tl_ids)
        self.has_left = has_left
        self.note = note
        self.points = [self._pt(start_pose)]
        self.started_at = time.time()

    @staticmethod
    def _pt(pose):
        return (pose["x"], pose["y"], pose["z"], pose["yaw_deg"])

    def maybe_add(self, pose, interval_m, yaw_interval_deg):
        """
        이동거리 또는 방향변화가 임계를 넘으면 경로점을 추가한다.

        거리 기준만 쓰면 급커브에서 점이 성기게 찍혀 pure pursuit 이 코너를 자른다.
        그래서 yaw 변화도 같이 본다.
        """
        p = self._pt(pose)
        last = self.points[-1]
        moved = dist2d(p, last)
        turned = abs(angle_diff_deg(p[3], last[3]))
        if moved >= interval_m or (moved >= 0.2 and turned >= yaw_interval_deg):
            self.points.append(p)
            return True
        return False

    def force_add(self, pose, min_move_m=0.05):
        """
        현재 위치를 경로점으로 추가한다. 직전 점과 사실상 같은 자리면 무시한다
        (끝점 마킹 시 중복점이 생겨 pure pursuit 의 세그먼트 계산이 0나눗셈에 걸리는 걸 막는다).
        """
        p = self._pt(pose)
        if dist2d(p, self.points[-1]) < min_move_m:
            return False
        self.points.append(p)
        return True

    @property
    def length_m(self):
        return path_length(self.points)

    @property
    def elapsed_s(self):
        return time.time() - self.started_at


class CaptureSession(object):
    def __init__(self, cfg, spots_dir, args):
        self.cfg = cfg
        self.spots_dir = spots_dir
        self.args = args

        cap = cfg.get("capture", {})
        self.interval_m = float(args.interval or cap.get("interval_m", 0.5))
        self.yaw_interval_deg = float(cap.get("yaw_interval_deg", 5.0))
        self.poll_hz = float(cap.get("poll_hz", 20.0))
        self.min_length_m = float(cap.get("min_length_m", 10.0))
        self.arrival_radius_m = float(cap.get("arrival_radius_m", 3.0))

        self.map_name = cfg["morai"]["map_name"]

        # 다음 캡처에 적용될 기본값. t/l/k/n 으로 미리 정해두면 여러 지점에 이어서 쓰인다.
        self.kind = schema.KIND_SIGNAL
        self.tl_ids = []
        self.has_left = None      # None = 아직 미지정 (저장 시 물어봄)
        self.note = ""

        self.rec = None
        self.pose = None
        self.saved = []

        self.src = pose_source.create(cfg, source=args.pose_source)

    # ------------------------------------------------------------------ 표시
    def status_line(self):
        if self.pose is None:
            return "pose 대기 중..."
        p = self.pose
        base = "x=%8.2f y=%8.2f yaw=%7.2f v=%4.1fm/s" % (
            p["x"], p["y"], p["yaw_deg"], p["speed_mps"])
        if p.get("link_id"):
            base += " link=%s" % p["link_id"]
        # 시뮬이 감지한 신호등 ID/색. 이게 set_traffic_light_info 가 받는 ID 형식이다.
        det = self.detected_tl()
        if det:
            base += " TL=%s(%s)" % (det, _color_name(p.get("tl_color")))
        if self.rec is None:
            return "[대기] %s | kind=%s tl=%s left=%s" % (
                base, self.kind, ",".join(self.tl_ids) or "-",
                "?" if self.has_left is None else ("Y" if self.has_left else "N"))
        d = dist2d(Recording._pt(p), (self.rec.start["x"], self.rec.start["y"], 0, 0))
        return "[녹화] %s | 점=%d 길이=%5.1fm 시작점거리=%5.1fm" % (
            base, len(self.rec.points), self.rec.length_m, d)

    def print_state(self):
        print("\n--- 현재 설정 ---")
        print("  map          : %s" % self.map_name)
        print("  kind         : %s" % self.kind)
        print("  traffic light: %s" % (", ".join(self.tl_ids) or "(미지정)"))
        print("  has_left     : %s" % ("(미지정)" if self.has_left is None else self.has_left))
        print("  note         : %s" % (self.note or "(없음)"))
        if self.kind == schema.KIND_SIGNAL:
            st = tl_states.states_for(bool(self.has_left))
            print("  수집 상태(%d): %s" % (len(st), ", ".join(st)))
        else:
            print("  수집 상태(1): unknown")
        print("  녹화중       : %s" % ("예 (점 %d, %.1fm)" % (len(self.rec.points), self.rec.length_m)
                                       if self.rec else "아니오"))
        print("  저장 위치    : %s" % self.spots_dir)
        print("-----------------")

    def print_list(self):
        spots = schema.load_all(self.spots_dir)
        if not spots:
            print("\n저장된 지점 없음 (%s)" % self.spots_dir)
            return
        print("\n--- 저장된 지점 %d개 ---" % len(spots))
        for s in spots:
            print("  %-10s kind=%-7s len=%6.1fm pts=%4d states=%d tl=%s"
                  % (s.spot_id, s.kind, s.path_length_m, len(s.path),
                     len(s.states()), ",".join(s.tl_ids) or "-"))
        n_env = len(self.cfg.get("collect", {}).get("weathers", ["SUNNY", "FOGGY"])) * \
            len(self.cfg.get("collect", {}).get("hours", [11, 13, 15]))
        n_seed = int(self.cfg.get("collect", {}).get("object_seeds", 1))
        total, _ = schema.combination_count(spots, n_env, n_seed)
        print("  → 환경 %d조합 × 객체seed %d 기준 총 주행 횟수: %d회" % (n_env, n_seed, total))
        print("-----------------")

    # ---------------------------------------------------------------- 액션
    def start_recording(self, kr):
        if self.rec is not None:
            print("\n이미 녹화 중이다. [e] 로 끝내거나 [x] 로 취소할 것.")
            return
        if self.pose is None:
            print("\npose 를 아직 못 읽었다. 잠시 후 다시.")
            return
        if self.pose["speed_mps"] > 0.5:
            print("\n주의: 차가 움직이는 중(%.1f m/s)이다. 시작점은 정지 상태에서 찍는 게 정확하다."
                  % self.pose["speed_mps"])
        self.rec = Recording(self.pose, self.kind, self.tl_ids, self.has_left, self.note)
        print("\n▶ 녹화 시작 @ (%.2f, %.2f, yaw=%.1f). 끝점까지 그대로 주행할 것."
              % (self.pose["x"], self.pose["y"], self.pose["yaw_deg"]))

    def cancel_recording(self):
        if self.rec is None:
            print("\n녹화 중이 아니다.")
            return
        print("\n✕ 녹화 취소 (점 %d개, %.1fm 버림)" % (len(self.rec.points), self.rec.length_m))
        self.rec = None

    def finish_recording(self, kr):
        if self.rec is None:
            print("\n녹화 중이 아니다. 먼저 [s].")
            return
        if self.pose is None:
            print("\npose 를 못 읽어 저장할 수 없다.")
            return

        self.rec.force_add(self.pose)
        length = self.rec.length_m
        if length < self.min_length_m:
            print("\n경로가 너무 짧다 (%.1fm < %.1fm)." % (length, self.min_length_m))
            if not kr.confirm("그래도 저장할까?", default=False):
                return

        kind = self.rec.kind
        tl_ids = list(self.rec.tl_ids)
        has_left = self.rec.has_left
        note = self.rec.note

        # 저장 직전에 빠진 값만 묻는다.
        if kind == schema.KIND_SIGNAL:
            if not tl_ids:
                det = self.detected_tl()
                raw = kr.prompt(
                    "신호등 ID (쉼표로 여러개%s, 나중에 YAML 에서 채우려면 엔터): "
                    % (", 감지된 값=%s" % det if det else ""))
                if not raw and det:
                    raw = det
                tl_ids = [t.strip() for t in raw.split(",") if t.strip()]
            if has_left is None:
                has_left = kr.confirm("이 신호등에 좌회전 화살표가 있나?", default=False)

        default_id = schema.next_spot_id(self.spots_dir, kind)
        spot_id = kr.prompt("spot_id [%s]: " % default_id, default_id)
        if not schema.SPOT_ID_RE.match(spot_id):
            print("spot_id 형식이 잘못됐다. '%s' 로 저장한다." % default_id)
            spot_id = default_id
        if os.path.exists(os.path.join(self.spots_dir, "%s.yaml" % spot_id)):
            if not kr.confirm("'%s' 가 이미 있다. 덮어쓸까?" % spot_id, default=False):
                spot_id = default_id
                print("→ '%s' 로 저장한다." % spot_id)
        if not note:
            note = kr.prompt("메모(엔터로 건너뜀): ")

        end_pose = self.pose
        data = {
            "spot_id": spot_id,
            "map": self.map_name,
            "kind": kind,
            "note": note,
            "traffic_light": {
                "ids": tl_ids,
                "link_id": end_pose.get("link_id", ""),
                "has_left": bool(has_left) if has_left is not None else False,
            },
            "start": {k: round(float(self.rec.start[k]), 4)
                      for k in ("x", "y", "z", "yaw_deg")},
            "end": {k: round(float(end_pose[k]), 4)
                    for k in ("x", "y", "z", "yaw_deg")},
            "arrival_radius_m": self.arrival_radius_m,
            "path_length_m": round(length, 2),
            "capture": {
                "source": self.args.pose_source or self.cfg.get("capture", {}).get("pose_source", "grpc"),
                "interval_m": self.interval_m,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "path": [[round(v, 4) for v in p] for p in self.rec.points],
        }
        if kind == schema.KIND_UNKNOWN:
            data["traffic_light"] = {"ids": [], "link_id": "", "has_left": False}

        out = schema.save(data, self.spots_dir)
        warns = schema.validate(data, strict=False)
        n_states = len(schema.Spot(data).states())
        print("\n✔ 저장: %s" % out)
        print("   길이 %.1fm, 점 %d개, 수집 상태 %d종" % (length, len(self.rec.points), n_states))
        for w in warns:
            print("   ⚠ %s" % w)
        self.saved.append(spot_id)
        self.rec = None

    def detected_tl(self):
        """
        시뮬레이터가 지금 ego 앞에서 감지한 신호등 ID.

        이 값이 곧 set_traffic_light_info() 가 받는 ID 형식이다. MGeo 의
        traffic_light_id 와 형식이 같은지 아직 미확인이라, 신호등 앞으로 몰고 가서
        이 표시를 보고 실제 ID 를 수집하면 된다.
        """
        if self.pose is None:
            return ""
        tid = str(self.pose.get("tl_id") or "").strip()
        return "" if tid in _NO_TL else tid

    def edit_tl_ids(self, kr):
        det = self.detected_tl()
        hint = "현재=%s" % (",".join(self.tl_ids) or "-")
        if det:
            hint += ", 감지된 값=%s (엔터로 사용)" % det
        raw = kr.prompt("신호등 ID (쉼표 구분, %s): " % hint)
        if not raw and det:
            raw = det
        self.tl_ids = [t.strip() for t in raw.split(",") if t.strip()]
        if self.rec is not None:
            self.rec.tl_ids = list(self.tl_ids)
        print("→ 신호등 ID = %s" % (", ".join(self.tl_ids) or "(없음)"))

    def toggle_left(self):
        self.has_left = True if self.has_left is None else (not self.has_left)
        if self.rec is not None:
            self.rec.has_left = self.has_left
        st = tl_states.states_for(self.has_left)
        print("\n→ has_left = %s → 수집 상태 %d종: %s" % (self.has_left, len(st), ", ".join(st)))

    def toggle_kind(self):
        self.kind = (schema.KIND_UNKNOWN if self.kind == schema.KIND_SIGNAL
                     else schema.KIND_SIGNAL)
        if self.rec is not None:
            self.rec.kind = self.kind
        print("\n→ kind = %s" % self.kind)

    def edit_note(self, kr):
        self.note = kr.prompt("메모: ", self.note)
        if self.rec is not None:
            self.rec.note = self.note

    # ---------------------------------------------------------------- 루프
    def run(self):
        period = 1.0 / self.poll_hz
        print(HELP)
        last_draw = 0.0
        with KeyReader() as kr:
            if not kr.enabled:
                print("경고: TTY 가 아니라 키 입력을 못 받는다. 터미널에서 직접 실행할 것.")
                return 2
            while True:
                loop_start = time.time()
                try:
                    pose = self.src.read()
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    print("\npose 읽기 실패: %s" % exc)
                    time.sleep(0.5)
                    continue
                if pose is not None:
                    self.pose = pose
                    if self.rec is not None:
                        self.rec.maybe_add(pose, self.interval_m, self.yaw_interval_deg)

                key = kr.poll(timeout=0.0)
                if key:
                    if key == "q":
                        if self.rec is not None and not kr.confirm(
                                "녹화 중이다. 버리고 종료할까?", default=False):
                            continue
                        break
                    elif key == "s":
                        self.start_recording(kr)
                    elif key == "e":
                        self.finish_recording(kr)
                    elif key == "x":
                        self.cancel_recording()
                    elif key == "m":
                        if self.rec is not None and self.pose is not None:
                            if self.rec.force_add(self.pose):
                                print("\n· 점 추가 (총 %d)" % len(self.rec.points))
                            else:
                                print("\n· 직전 점과 같은 자리라 추가하지 않음")
                    elif key == "t":
                        self.edit_tl_ids(kr)
                    elif key == "l":
                        self.toggle_left()
                    elif key == "k":
                        self.toggle_kind()
                    elif key == "n":
                        self.edit_note(kr)
                    elif key == "p":
                        self.print_state()
                    elif key == "L":
                        self.print_list()
                    elif key == "?":
                        print(HELP)
                    last_draw = 0.0  # 상태줄 즉시 갱신

                now = time.time()
                if now - last_draw >= 0.1:
                    sys.stdout.write("\r\033[K" + self.status_line())
                    sys.stdout.flush()
                    last_draw = now

                sleep = period - (time.time() - loop_start)
                if sleep > 0:
                    time.sleep(sleep)

        print("\n\n캡처 종료. 이번 세션 저장: %d개 %s"
              % (len(self.saved), (", ".join(self.saved) or "")))
        return 0

    def close(self):
        try:
            self.src.close()
        except Exception:
            pass


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="MORAI 지점(시작점/끝점/경로) 캡처 툴",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=HELP)
    ap.add_argument("--config", default=os.path.join(PKG_ROOT, "config", "runtime.yaml"),
                    help="runtime.yaml 경로")
    ap.add_argument("--spots-dir", default=os.path.join(WS_ROOT, "spots"),
                    help="지점 YAML 저장 디렉터리")
    ap.add_argument("--pose-source", choices=["grpc", "ros"], default=None,
                    help="ego pose 소스 (기본: runtime.yaml 의 capture.pose_source)")
    ap.add_argument("--interval", type=float, default=None,
                    help="경로점 기록 간격(m). 기본 0.5")
    args = ap.parse_args(argv)

    cfg = load_cfg(args.config)
    if not os.path.isdir(args.spots_dir):
        os.makedirs(args.spots_dir)

    session = None
    try:
        session = CaptureSession(cfg, args.spots_dir, args)
        return session.run()
    except KeyboardInterrupt:
        print("\n중단됨.")
        return 130
    finally:
        if session is not None:
            session.close()


if __name__ == "__main__":
    sys.exit(main())
