#!/usr/bin/env python3
"""
시작점 / 끝점을 직접 지정해서 지점(spot)을 만든다.

두 점만 주면 그 사이를 MGeo 도로망을 따라 이어준다 (직선으로 잇지 않는다 —
곡선 도로에서 차선을 벗어나 pure pursuit 이 따라갈 수 없기 때문).
신호등도 경로가 지나는 정지선에서 자동으로 찾아 채운다.

  # 좌표로 직접 지정
  python3 tools/make_spot.py --start -60.3 -26.9 --end -6.3 -105.8

  # 지금 ego 가 있는 자리를 시작점으로 (MORAI 실행 중이어야 함)
  python3 tools/make_spot.py --start ego --end -6.3 -105.8

  # 이름/신호등/좌회전유무 직접 지정
  python3 tools/make_spot.py --start -60.3 -26.9 --end -6.3 -105.8 \
      --id my_spot_01 --tl C1256W000081 --has-left

  # 미리보기만
  python3 tools/make_spot.py --start -60.3 -26.9 --end -6.3 -105.8 --dry-run

좌표는 MORAI 화면의 ego 위치를 읽거나, tools/where.py 로 확인하면 된다.
"""

import argparse
import math
import os
import sys
from datetime import datetime

import yaml

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS_ROOT = os.path.dirname(os.path.dirname(PKG_ROOT))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from spot import mgeo as mgeo_mod       # noqa: E402
from spot import schema                 # noqa: E402
from tl import states as tl_states      # noqa: E402
from utils.geometry import resample     # noqa: E402


def resolve_point(spec, cfg, label):
    """'ego' 면 시뮬에서 현재 ego 위치를 읽고, 아니면 좌표 두 개를 파싱한다."""
    if len(spec) == 1 and str(spec[0]).lower() == "ego":
        from sim import pose_source

        src = pose_source.create(cfg, source="grpc")
        try:
            pose = src.read()
            if pose is None:
                raise RuntimeError("ego pose 를 못 읽었다")
            print("[%s] ego 위치 사용: (%.2f, %.2f)" % (label, pose["x"], pose["y"]))
            return float(pose["x"]), float(pose["y"])
        finally:
            src.close()
    if len(spec) < 2:
        raise SystemExit("--%s 는 'X Y' 두 개 또는 'ego' 여야 한다 (받은 값: %s)"
                         % (label, spec))
    return float(spec[0]), float(spec[1])


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="시작점/끝점을 직접 지정해 지점 생성",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--start", nargs="+", required=True, metavar="X Y|ego")
    ap.add_argument("--end", nargs="+", required=True, metavar="X Y|ego")
    ap.add_argument("--id", default=None, help="spot_id (기본: 자동 번호)")
    ap.add_argument("--note", default="", help="메모")
    ap.add_argument("--tl", default=None,
                    help="신호등 ID (쉼표 구분). 생략하면 경로에서 자동 탐지")
    ap.add_argument("--has-left", dest="has_left", action="store_true", default=None,
                    help="좌회전 화살표 있음 (생략 시 MGeo movement 로 자동 판정)")
    ap.add_argument("--no-left", dest="has_left", action="store_false",
                    help="좌회전 화살표 없음")
    ap.add_argument("--kind", choices=["signal", "unknown", "auto"], default="auto",
                    help="signal=신호등 구간 / unknown=신호등 없는 구간 / "
                         "auto=경로에 신호등 정지선이 있으면 signal (기본)")
    ap.add_argument("--unknown", action="store_true",
                    help="--kind unknown 과 같음")
    ap.add_argument("--interval", type=float, default=0.5, help="경로점 간격(m)")
    ap.add_argument("--arrival-radius", type=float, default=3.0)
    ap.add_argument("--max-snap", type=float, default=15.0,
                    help="점을 도로에 스냅할 최대 허용거리(m)")
    ap.add_argument("--config", default=os.path.join(PKG_ROOT, "config", "runtime.yaml"))
    ap.add_argument("--spots-dir", default=os.path.join(WS_ROOT, "spots"))
    ap.add_argument("--mgeo-dir", default=None)
    ap.add_argument("--max-detour", type=float, default=2.5,
                    help="경로/직선거리 비가 이 값을 넘으면 경고 (기본 2.5)")
    ap.add_argument("--force", action="store_true",
                    help="경고가 있어도 저장")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    grpc_src = cfg["paths"]["grpc_src"]
    if not os.path.isabs(grpc_src):
        cfg["paths"]["grpc_src"] = os.path.normpath(os.path.join(cfg_dir, grpc_src))

    mgeo_dir = args.mgeo_dir or cfg["paths"]["mgeo_dir"]
    if not os.path.isabs(mgeo_dir):
        mgeo_dir = os.path.normpath(os.path.join(cfg_dir, mgeo_dir))
    mg = mgeo_mod.MGeo(mgeo_dir)

    start = resolve_point(args.start, cfg, "start")
    end = resolve_point(args.end, cfg, "end")
    print("시작점 (%.2f, %.2f) → 끝점 (%.2f, %.2f)" % (start[0], start[1], end[0], end[1]))

    pts, meta = mg.route_between(start, end, max_snap_dist=args.max_snap)
    if pts is None:
        print("\n경로 생성 실패: %s" % meta["error"])
        print("힌트: MORAI 는 일방통행 링크라 방향이 중요하다. 시작점과 끝점을 바꿔보거나,")
        print("      두 점이 실제로 이어진 도로 위에 있는지 확인할 것.")
        return 1

    pts = resample(mgeo_mod.with_yaw(pts), args.interval)
    if len(pts) < 2:
        print("경로가 너무 짧다.")
        return 1

    length = sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
                 for i in range(len(pts) - 1))

    # ---- kind 결정: signal(신호등 구간) / unknown(신호등 없는 구간) ----
    kind_arg = "unknown" if args.unknown else args.kind
    # 경로가 지나는 정지선의 신호등 (auto 판정과 자동 채움에 쓰인다)
    found_tls = mg.traffic_lights_along(meta["links"])

    if kind_arg == "auto":
        kind = schema.KIND_SIGNAL if found_tls else schema.KIND_UNKNOWN
        print("  kind 자동판정 : %s (경로상 신호등 %d개)" % (kind, len(found_tls)))
    else:
        kind = schema.KIND_SIGNAL if kind_arg == "signal" else schema.KIND_UNKNOWN

    if kind == schema.KIND_UNKNOWN:
        tl_ids, has_left = [], False
        if found_tls:
            print("  ⚠ kind=unknown 인데 경로가 신호등 %s 정지선을 지난다. "
                  "신호등이 화면에 보이면 unknown 라벨이 오염된다." % ",".join(found_tls))
    else:
        if args.tl:
            tl_ids = [t.strip() for t in args.tl.split(",") if t.strip()]
        else:
            tl_ids = found_tls
        has_left = args.has_left
        if has_left is None:
            has_left = False
            for lid in meta["links"]:
                link = mg.links.get(lid)
                if link and mg.has_left_arrow(link["to_node_idx"]):
                    has_left = True
                    break

    spot_id = args.id or schema.next_spot_id(args.spots_dir, kind)

    data = {
        "spot_id": spot_id,
        "map": cfg["morai"]["map_name"],
        "kind": kind,
        "note": args.note,
        "traffic_light": {
            "ids": tl_ids,
            "link_id": meta["links"][-1] if meta["links"] else "",
            "has_left": bool(has_left),
        },
        "start": _pose(pts[0]),
        "end": _pose(pts[-1]),
        "arrival_radius_m": args.arrival_radius,
        "path_length_m": round(length, 2),
        "capture": {
            "source": "manual+mgeo",
            "interval_m": args.interval,
            "links": meta["links"],
            "requested_start": [round(start[0], 3), round(start[1], 3)],
            "requested_end": [round(end[0], 3), round(end[1], 3)],
            "snap_start_m": round(meta["snap_start_m"], 2),
            "snap_end_m": round(meta["snap_end_m"], 2),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "path": [[round(v, 4) for v in p] for p in pts],
    }

    # ---- 상식 검사: 일방통행 때문에 엉뚱하게 멀리 돌아가지 않았는가 ----
    straight = math.hypot(end[0] - start[0], end[1] - start[1])
    detour = (length / straight) if straight > 1.0 else 1.0
    problems = []
    if detour > args.max_detour:
        problems.append(
            "경로가 직선거리의 %.1f배다 (%.0fm vs 직선 %.0fm). MORAI 도로는 일방통행이라 "
            "방향이 반대면 맵을 크게 돌아간다. 시작점과 끝점을 바꿔볼 것."
            % (detour, length, straight))
    if kind == schema.KIND_SIGNAL and len(tl_ids) > 1:
        problems.append(
            "경로가 신호등 %d개(%s) 정지선을 지난다. 이 지점을 쓰면 그 신호등들이 "
            "**전부 같은 상태로** 연출된다. 의도한 게 아니면 끝점을 첫 신호등 앞으로 당길 것."
            % (len(tl_ids), ", ".join(tl_ids)))

    states = schema.Spot(data).states()
    print("\n생성 결과")
    print("  spot_id     : %s (%s)" % (spot_id, kind))
    print("  경로 길이   : %.1fm (점 %d개, %d개 링크 경유)"
          % (length, len(pts), len(meta["links"])))
    print("  도로 스냅   : 시작 %.2fm, 끝 %.2fm 이동" % (meta["snap_start_m"], meta["snap_end_m"]))
    print("  신호등      : %s" % (", ".join(tl_ids) or "(없음)"))
    print("  has_left    : %s" % has_left)
    print("  수집 상태(%d): %s" % (len(states), ", ".join(states)))
    print("  직선거리 대비: %.1f배 (직선 %.0fm)" % (detour, straight))
    if meta["snap_start_m"] > 5.0 or meta["snap_end_m"] > 5.0:
        print("  ⚠ 스냅 거리가 크다. 지정한 점이 도로에서 많이 떨어져 있다.")
    if not tl_ids and kind == schema.KIND_SIGNAL:
        print("  ⚠ kind=signal 인데 경로에 신호등 정지선이 없다. "
              "--kind unknown 이 맞는 구간 아닌가? (또는 --tl 로 직접 지정)")
    for p in problems:
        print("  ⚠ %s" % p)

    if args.dry_run:
        print("\n[dry-run] 파일을 쓰지 않았다.")
        return 0

    if problems and not args.force:
        print("\n위 경고 때문에 저장하지 않았다. 의도한 것이면 --force 를 붙일 것.")
        return 1

    out = schema.save(data, args.spots_dir)
    print("\n✔ 저장: %s" % out)
    return 0


def _pose(p):
    return {"x": round(p[0], 4), "y": round(p[1], 4),
            "z": round(p[2], 4), "yaw_deg": round(p[3], 3)}


if __name__ == "__main__":
    sys.exit(main())
