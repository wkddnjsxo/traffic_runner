#!/usr/bin/env python3
"""
지금 ego 가 어디 있는지 알려준다. 시작점/끝점 좌표를 손으로 지정할 때 쓴다.

  python3 tools/where.py            # 한 번 출력
  python3 tools/where.py --watch    # 계속 갱신 (Ctrl-C 로 종료)

출력에는 make_spot.py 에 그대로 붙여넣을 수 있는 인자도 같이 나온다.
실행 중인 시뮬에 재시작 없이 붙으므로 운전 중에도 안전하다.
"""

import argparse
import os
import sys
import time

import yaml

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from sim import pose_source            # noqa: E402
from spot import mgeo as mgeo_mod      # noqa: E402

_NO_TL = ("", "Not Detected", "None", "null", "-1")


def main(argv=None):
    ap = argparse.ArgumentParser(description="현재 ego 위치 출력")
    ap.add_argument("--config", default=os.path.join(PKG_ROOT, "config", "runtime.yaml"))
    ap.add_argument("--watch", action="store_true", help="계속 갱신")
    ap.add_argument("--hz", type=float, default=5.0)
    ap.add_argument("--no-mgeo", action="store_true", help="MGeo 조회 생략(빠름)")
    args = ap.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    for key in ("grpc_src", "mgeo_dir"):
        p = cfg["paths"][key]
        if not os.path.isabs(p):
            cfg["paths"][key] = os.path.normpath(os.path.join(cfg_dir, p))

    mg = None
    if not args.no_mgeo:
        mg = mgeo_mod.MGeo(cfg["paths"]["mgeo_dir"])

    src = pose_source.create(cfg, source="grpc")
    try:
        if not args.watch:
            _show(src.read(), mg, oneshot=True)
            return 0
        period = 1.0 / max(args.hz, 0.5)
        print("Ctrl-C 로 종료.\n")
        while True:
            _show(src.read(), mg, oneshot=False)
            time.sleep(period)
    except KeyboardInterrupt:
        print("\n종료.")
        return 0
    finally:
        src.close()


def _show(pose, mg, oneshot):
    if pose is None:
        print("ego pose 를 못 읽었다.")
        return
    tl = str(pose.get("tl_id") or "").strip()
    tl = "" if tl in _NO_TL else tl

    line = "x=%9.3f  y=%9.3f  z=%7.3f  yaw=%8.3f  v=%5.1fm/s" % (
        pose["x"], pose["y"], pose["z"], pose["yaw_deg"], pose["speed_mps"])
    if pose.get("link_id"):
        line += "  link=%s" % pose["link_id"]
    if tl:
        line += "  TL=%s" % tl

    if oneshot:
        print(line)
        if mg is not None:
            _mgeo_info(pose, mg)
        print("\nmake_spot.py 인자로 쓰려면:")
        print("  --start %.3f %.3f      (또는 --end %.3f %.3f)"
              % (pose["x"], pose["y"], pose["x"], pose["y"]))
    else:
        sys.stdout.write("\r\033[K" + line)
        sys.stdout.flush()


def _mgeo_info(pose, mg):
    link, proj = mg.nearest_link(pose["x"], pose["y"], max_dist=30.0)
    if link is None:
        print("  MGeo: 근처 도로 링크를 못 찾았다 (30m 이내)")
        return
    node = mg.nodes.get(link["to_node_idx"], {})
    tl = node.get("traffic_light_id")
    print("  MGeo: link=%s (%.2fm 떨어짐, type=%s)"
          % (link["idx"], proj["dist"], link["link_type"]))
    print("        정지선 신호등=%s  movements=%s"
          % (tl or "(없음)", "/".join(sorted(mg.movements_at(link["to_node_idx"]))) or "-"))
    if tl:
        print("        좌회전 화살표=%s" % mg.has_left_arrow(link["to_node_idx"]))
    print("        가장 가까운 신호등까지 %.1fm"
          % mg.distance_to_nearest_tl(pose["x"], pose["y"]))


if __name__ == "__main__":
    sys.exit(main())
