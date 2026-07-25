#!/usr/bin/env python3
"""
MGeo 에서 지점(spot)을 자동 생성한다.

신호등 정지선으로 끝나는 링크를 찾아, 거기서 뒤로 approach_len_m 만큼 거슬러 올라간
경로를 시작점~끝점으로 만든다. 좌회전 화살표 유무도 교차로 movement 로 판정한다.

  # 미리보기 (파일 안 씀)
  python3 tools/gen_spots_from_mgeo.py --dry-run

  # 신호등 지점 전부 + unknown 지점 8개 생성
  python3 tools/gen_spots_from_mgeo.py --unknown 8

  # 접근로 길이 100m, 차선마다 따로 생성
  python3 tools/gen_spots_from_mgeo.py --approach-len 100 --all-lanes

손으로 찍은 지점(capture_spot.py 산출물)과 같은 디렉터리에 섞여도 된다.
파일명 접두사가 다르므로(gen_ vs sig_/unk_) 충돌하지 않는다.
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

from spot import mgeo as mgeo_mod          # noqa: E402
from spot import schema                    # noqa: E402
from tl import states as tl_states         # noqa: E402
from utils.geometry import resample        # noqa: E402


def build_signal_spots(mg, args):
    """신호등 접근로마다 spot 을 만든다."""
    approaches = mg.approach_links()

    # 신호등 하나에 접근로(차선)가 여럿이면 기본은 대표 1개만 쓴다.
    by_tl = {}
    for l in approaches:
        tl_id = mg.nodes[l["to_node_idx"]]["traffic_light_id"]
        by_tl.setdefault(tl_id, []).append(l)

    selected = []
    for tl_id, links in sorted(by_tl.items()):
        if args.all_lanes:
            selected.extend((tl_id, l) for l in links)
        else:
            # ego_lane 1 (1차로) 우선, 없으면 가장 긴 링크
            links = sorted(links, key=lambda l: (str(l.get("ego_lane")) != "1",
                                                 -l.get("link_length", 0)))
            selected.append((tl_id, links[0]))

    spots = []
    for tl_id, link in selected:
        node_idx = link["to_node_idx"]
        pts, meta = mg.build_approach_path(link, args.approach_len, args.end_offset)
        if len(pts) < 2:
            continue
        pts = resample(mgeo_mod.with_yaw(pts), args.interval)
        if len(pts) < 2:
            continue

        has_left = mg.has_left_arrow(node_idx)
        movements = sorted(mg.movements_at(node_idx))
        length = sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
                     for i in range(len(pts) - 1))

        spots.append({
            "spot_id": "gen_%s" % _safe(tl_id) + ("_%s" % _safe(link["idx"][-4:])
                                                  if args.all_lanes else ""),
            "map": args.map_name,
            "kind": schema.KIND_SIGNAL,
            "note": "auto: TL=%s approach=%s movements=%s%s"
                    % (tl_id, link["idx"], "/".join(movements) or "-",
                       " TRUNCATED" if meta["truncated"] else ""),
            "traffic_light": {
                "ids": [tl_id],
                "link_id": link["idx"],
                "has_left": has_left,
            },
            "start": _pose(pts[0]),
            "end": _pose(pts[-1]),
            "arrival_radius_m": args.arrival_radius,
            "path_length_m": round(length, 2),
            "capture": {
                "source": "mgeo",
                "interval_m": args.interval,
                "links": meta["links"],
                "movements": movements,
                "truncated": meta["truncated"],
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "path": [[round(v, 4) for v in p] for p in pts],
        })
    return spots


def build_unknown_spots(mg, args, count):
    """
    신호등에서 충분히 떨어진 도로 구간을 unknown 지점으로 만든다.

    끝점이 어떤 신호등에서도 min_tl_dist_m 이상 떨어진 링크만 쓴다.
    (신호등이 화면 구석에라도 보이면 unknown 라벨이 오염된다)
    """
    cands = []
    for l in mg.links.values():
        if l["link_type"] != mgeo_mod.ROAD_LINK:
            continue
        node = mg.nodes.get(l["to_node_idx"])
        if not node or node.get("traffic_light_id"):
            continue
        end = l["points"][-1]
        d = mg.distance_to_nearest_tl(end[0], end[1])
        if d < args.min_tl_dist:
            continue
        cands.append((d, l))

    # 신호등에서 먼 순으로, 서로 떨어진 것들을 고른다
    cands.sort(key=lambda t: -t[0])
    picked, used_pts = [], []
    for d, l in cands:
        end = l["points"][-1]
        if any(math.hypot(end[0] - p[0], end[1] - p[1]) < args.unknown_spacing
               for p in used_pts):
            continue
        picked.append((d, l))
        used_pts.append(end)
        if len(picked) >= count:
            break

    spots = []
    for i, (d, link) in enumerate(picked, 1):
        pts, meta = mg.build_approach_path(link, args.approach_len, 0.0)
        if len(pts) < 2:
            continue
        pts = resample(mgeo_mod.with_yaw(pts), args.interval)
        if len(pts) < 2:
            continue
        length = sum(math.hypot(pts[j + 1][0] - pts[j][0], pts[j + 1][1] - pts[j][1])
                     for j in range(len(pts) - 1))
        spots.append({
            "spot_id": "gen_unk_%03d" % i,
            "map": args.map_name,
            "kind": schema.KIND_UNKNOWN,
            "note": "auto: 신호등에서 %.0fm 떨어진 구간 (link=%s)" % (d, link["idx"]),
            "traffic_light": {"ids": [], "link_id": link["idx"], "has_left": False},
            "start": _pose(pts[0]),
            "end": _pose(pts[-1]),
            "arrival_radius_m": args.arrival_radius,
            "path_length_m": round(length, 2),
            "capture": {
                "source": "mgeo",
                "interval_m": args.interval,
                "links": meta["links"],
                "dist_to_nearest_tl_m": round(d, 1),
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "path": [[round(v, 4) for v in p] for p in pts],
        })
    return spots


def _pose(p):
    return {"x": round(p[0], 4), "y": round(p[1], 4),
            "z": round(p[2], 4), "yaw_deg": round(p[3], 3)}


def _safe(s):
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in str(s))


def main(argv=None):
    ap = argparse.ArgumentParser(description="MGeo 에서 지점 자동 생성",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--config", default=os.path.join(PKG_ROOT, "config", "runtime.yaml"))
    ap.add_argument("--spots-dir", default=os.path.join(WS_ROOT, "spots"))
    ap.add_argument("--mgeo-dir", default=None, help="기본: runtime.yaml 의 paths.mgeo_dir")
    ap.add_argument("--approach-len", type=float, default=80.0,
                    help="정지선에서 뒤로 거슬러 올라갈 거리(m). 기본 80")
    ap.add_argument("--end-offset", type=float, default=5.0,
                    help="정지선 앞 몇 m 에서 끝낼지. 기본 5 (교차로 진입 방지)")
    ap.add_argument("--interval", type=float, default=0.5, help="경로점 간격(m)")
    ap.add_argument("--arrival-radius", type=float, default=3.0)
    ap.add_argument("--all-lanes", action="store_true",
                    help="신호등당 접근 차선 전부 생성 (기본: 대표 1개)")
    ap.add_argument("--unknown", type=int, default=0, help="생성할 unknown 지점 개수")
    ap.add_argument("--min-tl-dist", type=float, default=80.0,
                    help="unknown 지점 끝점이 신호등에서 떨어져야 할 최소거리(m)")
    ap.add_argument("--unknown-spacing", type=float, default=100.0,
                    help="unknown 지점끼리 최소 간격(m)")
    ap.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않고 요약만 출력")
    args = ap.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    args.map_name = cfg["morai"]["map_name"]

    mgeo_dir = args.mgeo_dir
    if mgeo_dir is None:
        mgeo_dir = cfg["paths"]["mgeo_dir"]
        if not os.path.isabs(mgeo_dir):
            mgeo_dir = os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(args.config)), mgeo_dir))
    mg = mgeo_mod.MGeo(mgeo_dir)
    print("MGeo 로드: 링크 %d, 노드 %d (%s)" % (len(mg.links), len(mg.nodes), mgeo_dir))

    spots = build_signal_spots(mg, args)
    print("신호등 지점 %d개 생성" % len(spots))
    if args.unknown:
        unk = build_unknown_spots(mg, args, args.unknown)
        print("unknown 지점 %d개 생성 (요청 %d)" % (len(unk), args.unknown))
        spots += unk

    n_left = sum(1 for s in spots if s["traffic_light"]["has_left"])
    n_trunc = sum(1 for s in spots if s["capture"].get("truncated"))
    lens = [s["path_length_m"] for s in spots]
    print("  좌회전 화살표 지점 : %d (7종 수집)" % n_left)
    print("  화살표 없음        : %d (4종 수집)"
          % (len([s for s in spots if s["kind"] == schema.KIND_SIGNAL]) - n_left))
    if lens:
        print("  경로 길이          : min=%.0fm 중앙=%.0fm max=%.0fm"
              % (min(lens), sorted(lens)[len(lens) // 2], max(lens)))
    if n_trunc:
        print("  ⚠ 목표 길이 미달(선행링크 없음): %d개" % n_trunc)

    col = cfg.get("collect", {})
    n_env = len(col.get("weathers", [])) * len(col.get("hours", []))
    n_seed = int(col.get("object_seeds", 1))
    total = sum(n_env * n_seed * len(schema.Spot(s).states()) for s in spots)
    print("  → 환경 %d조합 × 객체seed %d 기준 총 주행 %d회" % (n_env, n_seed, total))

    if args.dry_run:
        print("\n[dry-run] 파일을 쓰지 않았다. 상위 10개:")
        for s in spots[:10]:
            print("  %-24s %-7s %6.1fm left=%-5s tl=%s"
                  % (s["spot_id"], s["kind"], s["path_length_m"],
                     s["traffic_light"]["has_left"], ",".join(s["traffic_light"]["ids"]) or "-"))
        return 0

    if not os.path.isdir(args.spots_dir):
        os.makedirs(args.spots_dir)
    for s in spots:
        schema.save(s, args.spots_dir)
    print("\n%d개 저장: %s" % (len(spots), args.spots_dir))
    print("다음: python3 tools/spot_report.py 로 검수")
    return 0


if __name__ == "__main__":
    sys.exit(main())
