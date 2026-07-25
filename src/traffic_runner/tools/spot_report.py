#!/usr/bin/env python3
"""
캡처한 지점들을 검수한다. 수집 실행 전에 이걸로 한 번 돌려볼 것.

  - 각 지점의 경로 길이 / 점 개수 / 수집할 신호 상태 종류
  - 아직 안 채운 값(신호등 ID, has_left 등) 경고
  - 전체 수집 조합 수와 예상 소요 시간

  python3 tools/spot_report.py [--strict] [--plot out.png]
"""

import argparse
import math
import os
import sys

import yaml

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS_ROOT = os.path.dirname(os.path.dirname(PKG_ROOT))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from spot import mgeo as mgeo_mod      # noqa: E402
from spot import schema                # noqa: E402
from tl import states as tl_states     # noqa: E402
from utils.geometry import dist2d      # noqa: E402


def tl_in_view_ratio(spot, tl_points, half_fov_deg, max_range_m):
    """
    경로를 따라가며 카메라 화각 안에 신호등이 들어오는 프레임 비율.

    unknown 지점 검증용이다. '정지선을 지나지 않는다' 만으로는 부족하다 —
    옆길 신호등이 정면에 잡히면 unknown 라벨이 오염된다(실측: 정지선을 하나도
    안 지나는 지점인데 프레임의 59% 에 신호등이 잡혔다).
    """
    n_bad = 0
    closest = None
    which = None
    for p in spot.path:
        best = None
        for tx, ty, tid in tl_points:
            dx, dy = tx - p[0], ty - p[1]
            d = math.hypot(dx, dy)
            if d > max_range_m:
                continue
            ang = abs((math.degrees(math.atan2(dy, dx)) - p[3] + 180) % 360 - 180)
            if ang <= half_fov_deg and (best is None or d < best[0]):
                best = (d, tid)
        if best is not None:
            n_bad += 1
            if closest is None or best[0] < closest:
                closest, which = best[0], best[1]
    ratio = (100.0 * n_bad / len(spot.path)) if spot.path else 0.0
    return ratio, closest, which


def check_path_sanity(spot, max_gap_m=5.0):
    """경로에 튀는 점(큰 gap)이 없는지 본다. 텔레포트/센서 튐이 섞이면 여기서 잡힌다."""
    issues = []
    pts = spot.path
    for i in range(len(pts) - 1):
        d = dist2d(pts[i], pts[i + 1])
        if d > max_gap_m:
            issues.append("path[%d]->[%d] 간격 %.1fm (튄 점 의심)" % (i, i + 1, d))
    # 시작/끝 pose 가 path 양 끝과 맞는지
    s, e = spot.start, spot.end
    if pts:
        if dist2d((s["x"], s["y"]), pts[0]) > 1.0:
            issues.append("start pose 가 path[0] 과 %.1fm 떨어짐"
                          % dist2d((s["x"], s["y"]), pts[0]))
        if dist2d((e["x"], e["y"]), pts[-1]) > 1.0:
            issues.append("end pose 가 path[-1] 과 %.1fm 떨어짐"
                          % dist2d((e["x"], e["y"]), pts[-1]))
    return issues


def main(argv=None):
    ap = argparse.ArgumentParser(description="캡처된 지점 검수 리포트")
    ap.add_argument("--spots-dir", default=os.path.join(WS_ROOT, "spots"))
    ap.add_argument("--config", default=os.path.join(PKG_ROOT, "config", "runtime.yaml"))
    ap.add_argument("--strict", action="store_true",
                    help="미완성 지점이 하나라도 있으면 실패(exit 1)")
    ap.add_argument("--avg-speed", type=float, default=8.0,
                    help="주행 1회 소요시간 추정용 평균속도(m/s). 기본 8 (약 29km/h)")
    ap.add_argument("--hfov", type=float, default=65.0,
                    help="카메라 수평화각(도). MORAI 센서 설정값과 맞출 것")
    ap.add_argument("--tl-range", type=float, default=120.0,
                    help="신호등이 화면에서 식별 가능한 최대거리(m)")
    ap.add_argument("--no-view-check", action="store_true",
                    help="unknown 지점 화각 오염 검사 생략 (MGeo 로드를 건너뛴다)")
    args = ap.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    col = cfg.get("collect", {})
    weathers = col.get("weathers", ["SUNNY", "FOGGY"])
    hours = col.get("hours", [11, 13, 15])
    n_env = len(weathers) * len(hours)
    n_seed = int(col.get("object_seeds", 1))

    spots = schema.load_all(args.spots_dir, strict=False)
    if not spots:
        print("지점이 하나도 없다: %s" % args.spots_dir)
        print("→ tools/capture_spot.py 로 먼저 캡처할 것.")
        return 1

    tl_points = None
    if not args.no_view_check:
        mgeo_dir = cfg["paths"]["mgeo_dir"]
        if not os.path.isabs(mgeo_dir):
            mgeo_dir = os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(args.config)), mgeo_dir))
        try:
            mg = mgeo_mod.MGeo(mgeo_dir)
            tl_points = [(n["point"][0], n["point"][1], n["traffic_light_id"])
                         for n in mg.nodes.values() if n.get("traffic_light_id")]
        except Exception as exc:
            print("MGeo 로드 실패, 화각 검사 생략: %s" % exc)

    print("=" * 74)
    print("지점 검수 리포트  (%s)" % args.spots_dir)
    print("=" * 74)

    n_bad = 0
    total_drive_time_s = 0.0
    for s in spots:
        warns = schema.validate(s.data, strict=False)
        issues = check_path_sanity(s)
        states = s.states()
        n_runs = n_env * n_seed * len(states)
        drive_s = (s.path_length_m / max(args.avg_speed, 0.1)) * n_runs
        total_drive_time_s += drive_s

        mark = "  " if not (warns or issues) else "! "
        print("%s%-10s %-7s %6.1fm  pts=%4d  states=%d %-46s"
              % (mark, s.spot_id, s.kind, s.path_length_m, len(s.path),
                 len(states), "[" + ",".join(states) + "]"))
        print("             tl=%s  has_left=%s  주행 %d회 (약 %.0f분)"
              % (",".join(s.tl_ids) or "-", s.has_left, n_runs, drive_s / 60.0))
        for w in warns:
            print("             ⚠ %s" % w)
            n_bad += 1
        for i in issues:
            print("             ⚠ %s" % i)

        # unknown 지점: 신호등이 화면에 잡히면 라벨이 오염된다
        if s.is_unknown and tl_points:
            ratio, closest, which = tl_in_view_ratio(
                s, tl_points, args.hfov / 2.0, args.tl_range)
            if s.verified_no_tl:
                print("             화각 검사: 사람이 화면으로 확인함 ✔"
                      + ("  (기하 예측은 %.0f%% 였음 — 가림/거리로 실제로는 안 보임)" % ratio
                         if ratio > 0 else ""))
            elif ratio > 0:
                print("             ⚠ 화각(±%.0f°) 안에 신호등이 잡히는 프레임 %.1f%% "
                      "(최근접 %.0fm %s) — unknown 라벨 오염"
                      % (args.hfov / 2.0, ratio, closest, which))
                n_bad += 1
            else:
                print("             화각 검사 ✔ (신호등 안 잡힘)")

    total, _ = schema.combination_count(spots, n_env, n_seed)
    print("-" * 74)
    print("지점 %d개 | 환경 %d조합(%s × %s) | 객체 seed %d"
          % (len(spots), n_env, "/".join(weathers), "/".join(str(h) for h in hours), n_seed))
    print("총 주행 횟수: %d회" % total)
    print("순수 주행시간 추정: 약 %.1f시간 (평균 %.1fm/s, 리셋/세팅 오버헤드 제외)"
          % (total_drive_time_s / 3600.0, args.avg_speed))
    print()
    print("클래스별 커버리지 (주행 횟수 기준):")
    per_class = {n: 0 for n in tl_states.CLASS_NAMES}
    for s in spots:
        for st in s.states():
            per_class[st] += n_env * n_seed
    for name in tl_states.CLASS_NAMES:
        bar = "#" * min(40, per_class[name])
        flag = "  ← 0! 지점을 더 캡처할 것" if per_class[name] == 0 else ""
        print("  %-11s %4d  %s%s" % (name, per_class[name], bar, flag))
    print("=" * 74)

    if n_bad:
        print("\n미완성 항목 %d건. spots/*.yaml 을 열어 채울 것." % n_bad)
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
