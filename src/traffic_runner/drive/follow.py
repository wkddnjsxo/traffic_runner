"""
경로 추종 주행 1회 (시작점 → 끝점).

test_run.py(단일 지점 시험)와 collect.py(전체 수집)가 같은 코드를 쓴다.
주행 중 카메라 프레임 저장까지 여기서 담당한다.
"""

import math
import sys
import time

from drive.pure_pursuit import PurePursuit


class DriveParams(object):
    """주행 파라미터 묶음."""

    def __init__(self, speed=8.0, control_hz=20.0, lookahead_k=0.6,
                 lookahead_min=4.0, max_steer=35.0, max_cte=6.0,
                 arrival_radius=1.0, end_hold=0.5, verbose=True,
                 save_from_m=70.0, slowdown_m=18.0, stop_speed=1.5,
                 stop_timeout=4.0, brake_deadband=0.4, brake_gain=3.0,
                 save_until_m=None):
        self.speed = speed
        self.control_hz = control_hz
        self.lookahead_k = lookahead_k
        self.lookahead_min = lookahead_min
        self.max_steer = max_steer
        self.max_cte = max_cte
        self.arrival_radius = arrival_radius
        self.end_hold = end_hold
        self.verbose = verbose
        # 신호등까지 이 거리 안에 들어와야 프레임을 저장한다.
        # 더 멀면 신호등이 몇 픽셀밖에 안 되어 사람도 색을 구분 못 하는데,
        # 거기에 라벨이 붙으면 모델이 신호등이 아니라 배경을 보고 맞히는 법을 배운다.
        # None 이면 전부 저장.
        self.save_from_m = save_from_m
        # 신호등이 이 거리보다 가까워지면 저장을 멈춘다 (뒷축 기준).
        # 카메라는 위로 20° 기울어 있고 세로 화각 51.1° 라 앙각 45.5° 위는 화면 밖.
        # 신호등 5m 높이 기준 카메라 3.73m 안쪽이면 이탈하고, 카메라가 뒷축보다
        # 1.90m 앞이므로 뒷축 기준 5.63m 이다.
        # 실측: 8.1m 완벽 / 5.0m 상단 걸침 / 1.5m 신호등 없음(보행자 신호만).
        self.save_until_m = save_until_m
        # 끝점 앞 이 거리부터 감속을 시작한다. 전속으로 달리다 도달 판정 시점에
        # 브레이크를 잡으면 관성으로 끝점을 몇 m 지나친다 (실측: 끝점 3.6m 전에
        # 6.4m/s). 끝점 너머 5m 부터 객체가 있으므로 지나치면 충돌 위험도 있다.
        self.slowdown_m = slowdown_m
        # 끝점에서의 목표 속도
        self.stop_speed = stop_speed
        # 정지 대기 상한. 이 안에 못 서면 그냥 넘어간다.
        self.stop_timeout = stop_timeout
        # 목표보다 이만큼 빠르면 브레이크를 건다 (m/s)
        self.brake_deadband = brake_deadband
        # 초과 속도 이만큼당 브레이크 1.0 (작을수록 급제동)
        self.brake_gain = brake_gain

    @classmethod
    def from_args(cls, args):
        return cls(speed=args.speed, control_hz=args.control_hz,
                   lookahead_k=args.lookahead_k, lookahead_min=args.lookahead_min,
                   max_steer=args.max_steer, max_cte=args.max_cte,
                   arrival_radius=args.arrival_radius, end_hold=args.end_hold,
                   verbose=getattr(args, "verbose", True),
                   save_from_m=getattr(args, "save_from", 70.0),
                   slowdown_m=getattr(args, "slowdown", 18.0),
                   stop_speed=getattr(args, "stop_speed", 1.5),
                   save_until_m=getattr(args, "save_until", None))


#: 도달 판정 후 브레이크로 더 나아가는 거리 (실측 기반 여유)
COAST_ALLOWANCE_M = 0.2


def arrival_radius_for(cfg, stop_gap_m=None):
    """
    앞범퍼가 끝점에서 stop_gap_m 앞에 서도록 하는 도달 반경을 구한다.

    MORAI 의 ego 좌표는 뒷축 중심이라, 좌표가 끝점 앞에 있어도 차체는 정지선을
    3m 넘어간다(Ioniq5 기준 뒷축→앞범퍼 3.845m). 그래서 차량 기하를 반영해
    반경을 잡아야 눈으로 봐도 정지선에 선다.

        arrival_radius = (뒷축→앞범퍼) + (원하는 여유) + (제동 중 더 가는 거리)
    """
    front = float(cfg.get("morai", {}).get("ego_front_from_origin_m", 3.845))
    if stop_gap_m is None:
        stop_gap_m = float(cfg.get("collect", {}).get("stop_gap_m", 0.5))
    return front + float(stop_gap_m) + COAST_ALLOWANCE_M


def settle_at_end(world, hold_sec, hz=20.0, stop_timeout=4.0, stop_speed=0.3,
                  on_frame=None):
    """
    차를 실제로 세운 뒤 hold_sec 만큼 멈춰 있는다.

    고정 시간만 브레이크를 잡으면 안 된다. 6.4m/s 에서 완전히 서는 데 1초 넘게
    걸리는데 0.5초 뒤에 넘어가면 차가 굴러가는 채로 다음 주행이 시작된다.
    그래서 **속도가 실제로 0 에 가까워질 때까지** 잡고, 그 다음에 hold 한다.

    on_frame(state) 를 주면 제동 중(=아직 움직이는 동안)에도 프레임을 저장한다.
    도달 판정 시점과 실제 정지 위치는 제동거리(실측 약 0.2m)만큼 어긋나는데,
    이 구간을 안 찍으면 마지막 이미지가 정지 위치와 달라진다.
    완전히 선 뒤에는 부르지 않는다 — 같은 그림이 반복 저장되면 정지 시점 화면만
    과대표집되어 학습이 치우친다.

    반환: 정지까지 걸린 시간(s). 상한을 넘겨 못 섰으면 None.
    """
    period = 1.0 / hz
    t0 = time.time()
    stopped_at = None
    while time.time() - t0 < stop_timeout:
        world.brake_stop()
        state = world.ego_state()
        if state is not None and abs(state.get("speed_mps", 0.0)) <= stop_speed:
            stopped_at = time.time() - t0
            break
        if on_frame is not None and state is not None:
            on_frame(state)
        time.sleep(period)

    deadline = time.time() + max(0.0, hold_sec)
    while time.time() < deadline:
        world.brake_stop()
        time.sleep(min(period, max(0.0, deadline - time.time())))
    return stopped_at


def drive_spot(world, spot, params, rec=None, writer=None, ctx=None,
               not_before=None, tl_points=None, should_stop=None,
               expect_color=None):
    """
    시작점에서 끝점까지 pure pursuit 로 주행한다.

    rec/writer 가 주어지면 주행 중 카메라 프레임을 저장한다.
    not_before 이후에 도착한 프레임만 저장한다 — 신호 연출 직전에 찍힌 프레임에는
    이전 신호가 남아 있어 라벨이 틀리기 때문이다.

    expect_color 를 주면 주행 내내 ego 가 보는 신호등 색을 감시한다. 기대와 다른
    색이 보이면 라벨이 틀린 것이므로 그 주행을 실패로 돌린다.

    주행 전 한 번만 확인하는 것보다 이 방식이 낫다. 시작점은 신호등 링크에서
    멀어 MORAI 가 신호등을 연결하지 못하는 경우가 많고(실측: 95m 지점 미감지),
    정작 프레임을 저장하는 구간(70m 이내)에서는 감지가 잘 되기 때문이다.

    반환 (ok, info). 도달/타임아웃/경로이탈 중 하나로 끝난다.
    """
    pp = PurePursuit(spot.path, lookahead_k=params.lookahead_k,
                     lookahead_min=params.lookahead_min,
                     max_steer_deg=params.max_steer)

    end = spot.end
    # ★ 도달 반경 = 끝점에 얼마나 바짝 세울지 ★
    # 이 반경 안에 들어오면 '도달' 로 치고 그 지점에서 정지·저장을 끝낸다.
    # 크게 잡으면(예전 4.55m) 끝점 한참 앞에서 저장이 끊긴다. 끝점 이미지까지
    # 저장하려면 작아야 한다. 뒷축 기준점이 끝점 stop_radius 안에 들어올 때까지
    # 주행하며 계속 저장한다.
    arrival_r = params.arrival_radius if params.arrival_radius else 0.8
    # 경로 길이에 여유를 곱해 타임아웃을 잡는다 (아주 느려도 끝나게)
    timeout = max(20.0, (spot.path_length_m / max(params.speed * 0.3, 1.0)) + 20.0)

    t0 = time.time()
    period = 1.0 / params.control_hz
    ticks = 0
    counters = {"saved": 0}
    skipped_far = 0
    verified_frames = 0
    mismatch = None
    max_cte = 0.0
    tl_seen = {}
    last_print = 0.0

    def save_frame(state):
        """조건이 맞으면 현재 프레임을 저장한다. 저장했으면 True."""
        if rec is None or writer is None:
            return False
        x_, y_ = state["x"], state["y"]
        d_end = math.hypot(end["x"] - x_, end["y"] - y_)
        d_tl = (min(math.hypot(t[0] - x_, t[1] - y_) for t in tl_points)
                if tl_points else None)
        if d_tl is not None:
            if params.save_from_m is not None and d_tl > params.save_from_m:
                return False
            if params.save_until_m is not None and d_tl < params.save_until_m:
                return False
        got = rec.latest(not_before=not_before)
        if got is None:
            return False
        jpeg, seq, stamp = got
        meta = dict(ctx or {})
        meta.update({
            "frame_idx": counters["saved"],
            "dist_to_end_m": round(d_end, 2),
            "dist_to_tl_m": round(d_tl, 2) if d_tl is not None else "",
            "ego_x": round(x_, 3), "ego_y": round(y_, 3),
            "ego_yaw": round(state["yaw_deg"], 2),
            "speed_mps": round(state["speed_mps"], 2),
            "tl_id": state.get("tl_id", ""),
            "tl_color_observed": state["tl_color"],
            "stamp": "%.3f" % stamp,
        })
        writer.save(jpeg, meta)
        rec.mark_saved(seq)
        counters["saved"] += 1
        return True

    def finish(ok, reason, dist_end=None):
        # 정지까지 주행 루프 안에서 이미 저장했으므로, 여기서는 멈춰만 있는다
        # (완전 정지 대기). 추가 저장은 하지 않는다 — 선 채로 같은 그림이 반복
        # 저장되면 정지 시점 화면만 과대표집된다.
        settle_at_end(world, params.end_hold, stop_timeout=params.stop_timeout)
        saved = counters["saved"]
        return ok, {"reason": reason, "elapsed": time.time() - t0, "ticks": ticks,
                    "saved": saved, "skipped_far": skipped_far,
                    "verified_frames": verified_frames, "mismatch": mismatch,
                    "max_cte": max_cte, "tl_seen": tl_seen, "dist_end": dist_end}

    while True:
        loop_t = time.time()
        elapsed = loop_t - t0

        if should_stop is not None and should_stop():
            return finish(False, "중단 요청")

        state = world.ego_state()
        if state is None:
            time.sleep(0.05)
            continue

        x, y = state["x"], state["y"]
        dist_end = math.hypot(end["x"] - x, end["y"] - y)
        # 신호등까지의 '진짜' 거리. 끝점과 신호등은 최대 12m 어긋난다(실측).
        dist_tl = (min(math.hypot(t[0] - x, t[1] - y) for t in tl_points)
                   if tl_points else None)

        # 끝점에 도달했고 + 실제로 멈췄으면 종료.
        # ★ 도달 판정만으로 끝내면 안 된다 ★
        # 도달 반경(차량 앞범퍼가 정지선에 서도록 4.55m)에서 루프를 나가면,
        # 그 이후 정지까지의 구간이 저장되지 않는다. 그런데 신호등은 정지선보다
        # 더 뒤에 있어(sig_006 은 끝점=신호등 6.8m), 그 미저장 구간이 바로
        # '차가 실제로 서서 신호를 보는' 가장 중요한 프레임이다.
        # 그래서 도달 후에도 계속 주행/저장하며 완전히 설 때까지 간다.
        arrived = dist_end <= arrival_r
        if arrived and state["speed_mps"] <= params.stop_speed * 0.5:
            world.brake_stop()
            return finish(True, "끝점 도달·정지", dist_end)
        if elapsed > timeout:
            return finish(False, "타임아웃 (%.0fs)" % timeout, dist_end)

        steer, info = pp.step(x, y, state["yaw_deg"], state["speed_mps"])
        max_cte = max(max_cte, info["cross_track_m"])
        if info["cross_track_m"] > params.max_cte:
            return finish(False, "경로 이탈 (%.1fm)" % info["cross_track_m"], dist_end)

        # 끝점이 가까워지면 감속한다. 끝점(arrival_r)에서 속도가 0 이 되도록,
        # 그 앞에서는 stop_speed 이상을 유지해 끝점까지 실제로 도달하게 한다.
        target = pp.speed_for(steer, params.speed)
        if params.slowdown_m and dist_end < params.slowdown_m:
            ratio = max(0.0, min(1.0, (dist_end - arrival_r)
                                 / max(params.slowdown_m - arrival_r, 1e-6)))
            target = min(target, params.stop_speed + (target - params.stop_speed) * ratio)
            # 아직 끝점 전이면 기어가서라도 도달, 도달했으면 0 을 향해 브레이크
            target = max(target, params.stop_speed) if not arrived else 0.0

        # 속도 모드는 목표를 낮춰도 브레이크를 안 걸고 타력 주행만 한다
        # (실측 감속도 0.7m/s²). 목표보다 빠르면 브레이크를 직접 넣는다.
        excess = state["speed_mps"] - target
        if excess > params.brake_deadband:
            world.brake(min(1.0, excess / max(params.brake_gain, 1e-6)), steer)
        else:
            world.drive(target, steer)

        c = state["tl_color"]
        tl_seen[c] = tl_seen.get(c, 0) + 1
        ticks += 1

        # 라벨 검증: ego 가 보는 색이 연출한 색과 같은가.
        # -2(미감지)는 판단 불가라 넘어간다 — 시작점처럼 신호등 링크에서 먼 구간,
        # 정지선을 지나 신호등이 머리 위로 넘어간 구간에서 나온다.
        if expect_color is not None and c not in (-2, 0):
            if c == expect_color:
                verified_frames += 1
            else:
                mismatch = c
                return finish(False, "라벨 불일치: 기대 %d, 관측 %d"
                              % (expect_color, c), dist_end)

        # 신호등이 너무 멀거나(몇 픽셀밖에 안 됨) 너무 가까우면(화면 위로 벗어남)
        # 저장하지 않는다. unknown 지점(tl_points 없음)은 거리 개념이 없어 항상 저장.
        out_of_range = dist_tl is not None and (
            (params.save_from_m is not None and dist_tl > params.save_from_m) or
            (params.save_until_m is not None and dist_tl < params.save_until_m))
        if out_of_range:
            skipped_far += 1
        else:
            save_frame(state)

        if params.verbose and loop_t - last_print > 0.5:
            sys.stdout.write(
                "\r\033[K    진행 %5.1f%%  신호등까지 %s  cte=%4.2f  v=%4.1fm/s  "
                "저장 %d%s"
                % (info["progress"] * 100,
                   "%5.1fm" % dist_tl if dist_tl is not None else "  -   ",
                   info["cross_track_m"], state["speed_mps"], counters["saved"],
                   " (범위밖)" if out_of_range else ""))
            sys.stdout.flush()
            last_print = loop_t

        sleep = period - (time.time() - loop_t)
        if sleep > 0:
            time.sleep(sleep)
