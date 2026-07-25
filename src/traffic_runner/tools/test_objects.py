#!/usr/bin/env python3
"""
객체 배치를 확인한다.

  # 계획만 출력 (MORAI 불필요)
  python3 tools/test_objects.py --spot sig_004 --plan-only

  # 실제로 스폰해서 눈으로 확인 (엔터 누르면 정리)
  python3 tools/test_objects.py --spot sig_004 --seed 1

  # seed 를 바꿔가며 훑어보기
  python3 tools/test_objects.py --spot sig_004 --seeds 1,2,3 --hold 5

객체는 항상 끝점 '너머' 에만 놓인다. ego 주행 구간(시작점~끝점)에는 놓지 않으므로
충돌하지 않는다.
"""

import argparse
import os
import sys
import time

import yaml

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS_ROOT = os.path.dirname(os.path.dirname(PKG_ROOT))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from collect import objects as obj_mod     # noqa: E402
from spot import mgeo as mgeo_mod          # noqa: E402
from spot import schema                    # noqa: E402


def load_cfg(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg_dir = os.path.dirname(os.path.abspath(path))
    for key in ("grpc_src", "mgeo_dir"):
        p = cfg["paths"].get(key)
        if p and not os.path.isabs(p):
            cfg["paths"][key] = os.path.normpath(os.path.join(cfg_dir, p))
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="객체 배치 확인",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--spot", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seeds", default=None, help="여러 seed 를 쉼표로")
    ap.add_argument("--hold", type=float, default=0.0,
                    help="각 seed 를 몇 초 유지할지 (0이면 엔터 대기)")
    ap.add_argument("--plan-only", action="store_true", help="스폰 없이 계획만")
    ap.add_argument("--spots-dir", default=os.path.join(WS_ROOT, "spots"))
    ap.add_argument("--config", default=os.path.join(PKG_ROOT, "config", "runtime.yaml"))
    args = ap.parse_args(argv)

    cfg = load_cfg(args.config)
    obj_cfg = cfg.get("objects", {})

    path = os.path.join(args.spots_dir, "%s.yaml" % args.spot)
    if not os.path.exists(path):
        print("지점 없음: %s" % path)
        return 1
    spot = schema.load(path, strict=False)
    mg = mgeo_mod.MGeo(cfg["paths"]["mgeo_dir"])

    # 모델 배정은 전체 지점 기준으로 계산해야 균등해진다
    all_spots = schema.load_all(args.spots_dir)
    n_seeds = int(cfg.get("collect", {}).get("object_seeds", 3))
    all_plans = obj_mod.plan_all(all_spots, n_seeds, mg, obj_cfg)

    seeds = ([int(s) for s in args.seeds.split(",") if s.strip()]
             if args.seeds else [args.seed])

    print("=" * 70)
    print("객체 배치: %s  (끝점 %.1f~%.1fm 너머)"
          % (spot.spot_id, obj_cfg.get("min_ahead_m", 5.0),
             obj_cfg.get("max_ahead_m", 25.0)))
    print("=" * 70)

    plans = {}
    for seed in seeds:
        pl, meta = all_plans.get((spot.spot_id, seed), ([], {}))
        plans[seed] = pl
        print("seed %d: %s" % (seed, obj_mod.describe(pl)))
        for p in pl:
            print("    %-38s %-9s %-4s (%8.2f, %8.2f) yaw=%6.1f  전방%.0fm 횡%+.1f"
                  % (p.model, p.category, "동일" if p.lane == "same" else "반대",
                     p.x, p.y, p.yaw_deg, p.ahead_m, p.lateral_m))
        for note in meta.get("notes", []):
            print("    ⚠ %s" % note)

    if args.plan_only:
        return 0

    from sim.world import World

    world = World(cfg)
    spawner = obj_mod.ObjectSpawner(world)
    try:
        # 확인하기 좋게 ego 를 시작점에 둔다
        st = spot.start
        world.teleport_ego(st["x"], st["y"], st["z"], st["yaw_deg"], settle_sec=0.3)
        print("\nego 를 시작점에 배치했다. MORAI 화면에서 확인할 것.")

        for seed in seeds:
            spawner.clear()
            pl = plans[seed]
            n, failed = spawner.spawn(pl)
            print("\n[seed %d] %d/%d 스폰" % (seed, n, len(pl)))
            for model, why in failed:
                print("    ✘ %s: %s" % (model, why))
            if args.hold > 0:
                time.sleep(args.hold)
            else:
                try:
                    input("    엔터 → 다음 (Ctrl-C 종료) ")
                except EOFError:
                    break
    except KeyboardInterrupt:
        print("\n중단됨.")
    finally:
        print("\n객체 정리 중...")
        spawner.clear()
        world.close()
        print("완료.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
