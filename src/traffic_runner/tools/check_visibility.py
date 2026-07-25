#!/usr/bin/env python3
"""
지점별 '신호등이 카메라에 제대로 보이는가' 검사.

수집 전에 돌리면 sig_006 같은 문제를 미리 잡는다. sig_006 은 시작 구간에서
신호등이 화각 오른쪽 밖으로 벗어나 있었고(실측), 그 구간 학습 정확도가
46.8% — 찍기 수준이었다. 라벨은 맞는데 화면에 정보가 없는 프레임이었다.

세 가지를 본다.
  1. 수평 화각    : 신호등이 좌우 화각(HFOV) 안에 있는가
  2. 수직 화각    : 너무 가까워 화면 위로 벗어나지 않는가
  3. 정면성       : 신호등을 앞에서 보는가, 비스듬히/뒤에서 보는가
                    (신호등은 접근로 정면에서만 램프가 보인다)

  python3 tools/check_visibility.py
  python3 tools/check_visibility.py --spot sig_006 --verbose
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

from spot import mgeo as mgeo_mod   # noqa: E402
from spot import schema             # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="신호등 가시성 검사")
    ap.add_argument("--config", default=os.path.join(PKG_ROOT, "config", "runtime.yaml"))
    ap.add_argument("--spots-dir", default=os.path.join(WS_ROOT, "spots"))
    ap.add_argument("--spot", default=None, help="이 지점만 (쉼표 구분)")
    ap.add_argument("--hfov", type=float, default=65.0, help="카메라 수평화각(도)")
    ap.add_argument("--cam-forward", type=float, default=1.90,
                    help="카메라가 뒷축보다 앞선 거리(m)")
    ap.add_argument("--cam-height", type=float, default=1.20)
    ap.add_argument("--cam-pitch", type=float, default=20.0, help="위로 기울인 각(도)")
    ap.add_argument("--tl-height", type=float, default=4.4,
                    help="신호등 높이(m). 실측 역산값 4.36")
    ap.add_argument("--facing-tol", type=float, default=60.0,
                    help="접근 방향과 이 각도 이상 어긋나면 신호등 뒷면/옆면으로 본다")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    mgeo_dir = cfg["paths"]["mgeo_dir"]
    if not os.path.isabs(mgeo_dir):
        mgeo_dir = os.path.normpath(os.path.join(cfg_dir, mgeo_dir))
    mg = mgeo_mod.MGeo(mgeo_dir)

    tl_pos = {}
    for n in mg.nodes.values():
        t = n.get("traffic_light_id")
        if t:
            tl_pos.setdefault(t, []).append((n["point"][0], n["point"][1]))

    W, H = 1280, 960
    vfov = 2 * math.degrees(math.atan(math.tan(math.radians(args.hfov / 2)) * H / W))
    top = args.cam_pitch + vfov / 2
    half_h = args.hfov / 2

    spots = schema.load_all(args.spots_dir)
    if args.spot:
        want = {s.strip() for s in args.spot.split(",")}
        spots = [s for s in spots if s.spot_id in want]

    print("=" * 78)
    print("신호등 가시성 검사  (HFOV %.0f°, VFOV %.1f°, 상단 앙각 %.1f°)"
          % (args.hfov, vfov, top))
    print("=" * 78)
    print("%-9s %8s %10s %10s %10s  %s"
          % ("spot", "프레임", "화각밖(좌우)", "화각밖(위)", "뒷면/옆면", "판정"))

    problems = []
    for sp in spots:
        if sp.is_unknown:
            continue
        pts = [p for t in sp.tl_ids for p in tl_pos.get(t, [])]
        if not pts:
            print("%-9s  신호등 좌표가 mgeo 에 없어 검사 불가" % sp.spot_id)
            continue

        # 신호등이 향하는 방향 = 접근로 진행방향의 반대 (다가오는 차를 향한다)
        approach_yaw = math.radians(float(sp.end.get("yaw_deg", 0.0)))

        n = out_h = out_v = back = 0
        first_ok_dist = None
        for x, y, z, yaw in sp.path:
            d = min(math.hypot(p[0] - x, p[1] - y) for p in pts)
            tx, ty = min(pts, key=lambda p: math.hypot(p[0] - x, p[1] - y))
            # 카메라 위치 (뒷축보다 앞)
            cyaw = math.radians(yaw)
            cx = x + math.cos(cyaw) * args.cam_forward
            cy = y + math.sin(cyaw) * args.cam_forward
            dx, dy = tx - cx, ty - cy
            dist = math.hypot(dx, dy)
            n += 1

            bearing = math.degrees(math.atan2(dy, dx))
            off_h = abs((bearing - yaw + 180) % 360 - 180)
            elev = math.degrees(math.atan2(args.tl_height - args.cam_height,
                                           max(dist, 0.1)))
            # 정면성: ego 진행방향이 접근로 방향과 얼마나 맞는가
            facing = abs((yaw - math.degrees(approach_yaw) + 180) % 360 - 180)

            bad = False
            if off_h > half_h:
                out_h += 1
                bad = True
            if elev > top:
                out_v += 1
                bad = True
            if facing > args.facing_tol:
                back += 1
                bad = True
            if not bad and first_ok_dist is None:
                first_ok_dist = d

        pct_h, pct_v, pct_b = (100.0 * out_h / n, 100.0 * out_v / n, 100.0 * back / n)
        worst = max(pct_h, pct_v, pct_b)
        verdict = "OK" if worst < 5 else ("⚠ 주의" if worst < 20 else "✘ 문제")
        if worst >= 5:
            problems.append((sp.spot_id, pct_h, pct_v, pct_b, first_ok_dist))
        print("%-9s %8d %9.1f%% %9.1f%% %9.1f%%  %s"
              % (sp.spot_id, n, pct_h, pct_v, pct_b, verdict))

    print("-" * 78)
    if problems:
        print("문제 지점 %d개:" % len(problems))
        for sid, h, v, b, ok_d in problems:
            reasons = []
            if h >= 5:
                reasons.append("좌우 화각 밖 %.0f%%" % h)
            if v >= 5:
                reasons.append("화면 위로 벗어남 %.0f%%" % v)
            if b >= 5:
                reasons.append("신호등 뒷면/옆면 %.0f%%" % b)
            print("  %-9s %s" % (sid, ", ".join(reasons)))
            if ok_d:
                print("            → 신호등이 제대로 보이기 시작하는 거리: 약 %.0fm" % ok_d)
        print("\n대처: tools/prune_frames.py 로 그 구간을 걷어내거나,")
        print("      make_spot.py 로 시작점을 앞으로 당겨 재수집할 것.")
    else:
        print("전 지점 이상 없음.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
