#!/usr/bin/env python3
"""
지점 하나로 수집 시나리오를 실제로 돌려본다 (이미지 저장 없음 — 카메라는 나중).

확인하는 것:
  1. 날씨/시간 설정이 먹는가
  2. 신호등 연출이 먹는가 + ego 가 그 색을 실제로 보는가 (라벨 검증)
  3. 시작점 텔레포트가 되는가
  4. pure pursuit 로 끝점까지 주행이 되는가
  5. 상태를 바꿔가며 반복이 되는가

  # 신호 상태 하나만, 주행 없이 (가장 안전한 첫 시험)
  python3 tools/test_run.py --spot sig_002 --states red --no-drive

  # 한 상태로 주행까지
  python3 tools/test_run.py --spot sig_002 --states red

  # 그 지점의 모든 상태 순회 (실제 수집 루프와 같은 형태)
  python3 tools/test_run.py --spot sig_002

  # 환경까지 바꿔가며
  python3 tools/test_run.py --spot sig_002 --weather SUNNY --hour 11
"""

import argparse
import math
import os
import random
import sys
import time

import yaml

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS_ROOT = os.path.dirname(os.path.dirname(PKG_ROOT))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from drive.follow import DriveParams, arrival_radius_for, drive_spot   # noqa: E402
from sim.world import World                     # noqa: E402
from spot import schema                         # noqa: E402
from tl import states as tl_states              # noqa: E402
from tl.controller import TrafficLightController, _name_of   # noqa: E402


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
        description="지점 하나로 수집 시나리오 시험 주행",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--spot", required=True, help="spot_id (spots/<id>.yaml)")
    ap.add_argument("--spots-dir", default=os.path.join(WS_ROOT, "spots"))
    ap.add_argument("--config", default=os.path.join(PKG_ROOT, "config", "runtime.yaml"))
    ap.add_argument("--states", default=None,
                    help="시험할 신호 상태 (쉼표 구분). 생략하면 그 지점의 전체")
    ap.add_argument("--weather", default=None, help="SUNNY / FOGGY 등")
    ap.add_argument("--hour", type=int, default=None, help="11 / 13 / 15")
    ap.add_argument("--no-drive", action="store_true",
                    help="주행 없이 신호등 연출 + 텔레포트만 확인")
    ap.add_argument("--no-shuffle", action="store_true", help="상태 순서 섞지 않음")
    ap.add_argument("--speed", type=float, default=8.0, help="목표 속도(m/s)")
    ap.add_argument("--control-hz", type=float, default=20.0)
    ap.add_argument("--lookahead-k", type=float, default=0.6)
    ap.add_argument("--lookahead-min", type=float, default=4.0)
    ap.add_argument("--max-steer", type=float, default=35.0)
    ap.add_argument("--max-cte", type=float, default=6.0,
                    help="경로에서 이만큼 벗어나면 중단(m)")
    ap.add_argument("--arrival-radius", type=float, default=None, metavar="M",
                    help="도달 판정 반경. 생략하면 차량 기하로 자동 계산한다 "
                         "(뒷축→앞범퍼 3.845m + stop-gap + 제동여유)")
    ap.add_argument("--stop-gap", type=float, default=None, metavar="M",
                    help="앞범퍼가 끝점에서 이만큼 앞에 서도록 (기본: runtime.yaml "
                         "collect.stop_gap_m)")
    ap.add_argument("--save-from", type=float, default=None, metavar="M",
                    help="신호등까지 이 거리 안에서만 프레임 저장 (기본: runtime.yaml "
                         "collect.save_from_m). 0 이면 전부 저장")
    ap.add_argument("--save-until", type=float, default=None, metavar="M",
                    help="신호등이 이보다 가까우면 저장 중단 (기본: runtime.yaml "
                         "collect.save_until_m). 너무 가까우면 신호등이 화면 위로 "
                         "벗어난다. 0 이면 제한 없음")
    ap.add_argument("--slowdown", type=float, default=18.0, metavar="M",
                    help="끝점 앞 이 거리부터 감속 시작 (0이면 감속 없음)")
    ap.add_argument("--stop-speed", type=float, default=1.5, metavar="MPS",
                    help="끝점에서의 목표 속도(m/s)")
    ap.add_argument("--end-hold", type=float, default=0.5,
                    help="끝점 도달 후 정지 상태로 대기할 시간(s). 기본 0.5")
    ap.add_argument("--settle", type=float, default=0.6,
                    help="텔레포트/연출 후 안정화 대기(s)")
    ap.add_argument("--strict-verify", action="store_true",
                    help="ego 가 신호등을 감지 못하면 실패로 처리 (라벨 미검증 주행 방지)")
    ap.add_argument("--no-sibling", action="store_true",
                    help="형제 신호등 동시 변경 끄기")
    ap.add_argument("--save", action="store_true",
                    help="주행 중 카메라 프레임을 dataset/ 에 저장 (ROS 필요)")
    ap.add_argument("--dataset-dir", default=os.path.join(WS_ROOT, "dataset"))
    ap.add_argument("--objects", type=int, default=None, metavar="SEED",
                    help="객체 seed. 지정하면 그 배치를 깔고 주행한다")
    ap.add_argument("--verbose", action="store_true", default=True)
    ap.add_argument("--quiet", dest="verbose", action="store_false")
    args = ap.parse_args(argv)

    cfg = load_cfg(args.config)
    if args.save_from is None:
        args.save_from = cfg.get("collect", {}).get("save_from_m", 70.0)
    if not args.save_from:
        args.save_from = None
    if args.save_until is None:
        args.save_until = cfg.get("collect", {}).get("save_until_m", 0)
    if not args.save_until:
        args.save_until = None
    if args.arrival_radius is None:
        args.arrival_radius = arrival_radius_for(cfg, args.stop_gap)

    spot_path = os.path.join(args.spots_dir, "%s.yaml" % args.spot)
    if not os.path.exists(spot_path):
        avail = [f[:-5] for f in sorted(os.listdir(args.spots_dir))
                 if f.endswith(".yaml")] if os.path.isdir(args.spots_dir) else []
        print("지점을 못 찾았다: %s" % spot_path)
        print("있는 지점: %s" % (", ".join(avail) or "(없음)"))
        return 1
    spot = schema.load(spot_path, strict=False)

    states = ([s.strip() for s in args.states.split(",") if s.strip()]
              if args.states else spot.states())
    for s in states:
        tl_states.get(s)
    if not args.no_shuffle and len(states) > 1:
        random.shuffle(states)

    print("=" * 70)
    print("시험 주행: %s" % spot.spot_id)
    print("  경로     : %.1fm, 점 %d개" % (spot.path_length_m, len(spot.path)))
    print("  신호등   : %s (has_left=%s)" % (", ".join(spot.tl_ids) or "-", spot.has_left))
    print("  시험 상태: %s" % ", ".join(states))
    print("  주행     : %s" % ("안 함 (--no-drive)" if args.no_drive
                               else "pure pursuit, 목표 %.1fm/s" % args.speed))
    print("=" * 70)

    # 신호등 실제 좌표 (mgeo). 프레임마다 진짜 거리를 기록하는 데 쓴다.
    tl_points = []
    if spot.tl_ids:
        try:
            from spot import mgeo as _mgeo

            _mg = _mgeo.MGeo(cfg["paths"]["mgeo_dir"])
            for n in _mg.nodes.values():
                if n.get("traffic_light_id") in spot.tl_ids:
                    tl_points.append((n["point"][0], n["point"][1]))
            missing = [t for t in spot.tl_ids
                       if not any(nn.get("traffic_light_id") == t
                                  for nn in _mg.nodes.values())]
            if missing:
                print("[mgeo] 좌표를 못 찾은 신호등: %s (거리 기록은 나머지 기준)"
                      % ", ".join(missing))
        except Exception as exc:
            print("[mgeo] 신호등 좌표 로드 실패: %s" % exc)

    world = World(cfg)
    tlc = TrafficLightController(world, sibling=not args.no_sibling,
                                 settle_sec=args.settle, strict=args.strict_verify)

    rec = writer = None
    run_id = time.strftime("%Y%m%d_%H%M%S")
    if args.save:
        from collect.recorder import CameraRecorder, DatasetWriter

        rec = CameraRecorder(topic=cfg.get("ros", {}).get(
            "image_topic", "/image_jpeg/compressed"))
        writer = DatasetWriter(args.dataset_dir, run_id)
        print("[save] %s → %s" % (rec.topic, writer.root))

    spawner = None
    placements = []
    obj_mod = None
    if args.objects is not None:
        from collect import objects as obj_mod
        from spot import mgeo as mgeo_mod

        mg = mgeo_mod.MGeo(cfg["paths"]["mgeo_dir"])
        all_spots = schema.load_all(args.spots_dir)
        n_seeds = max(int(cfg.get("collect", {}).get("object_seeds", 3)),
                      args.objects + 1)
        plans = obj_mod.plan_all(all_spots, n_seeds, mg, cfg.get("objects", {}))
        placements, _ = plans.get((spot.spot_id, args.objects), ([], {}))
        spawner = obj_mod.ObjectSpawner(world)

    results = []
    try:
        # ---- 환경 ----
        if args.weather:
            world.set_weather(args.weather)
            print("[env] 날씨 = %s" % args.weather)
        if args.hour is not None:
            world.set_time_hour(args.hour)
            print("[env] 시간 = %d시" % args.hour)
        if args.weather or args.hour is not None:
            time.sleep(args.settle)
        print("[env] 현재: %s" % world.get_env())

        if spawner is not None:
            n, failed = spawner.spawn(placements)
            print("[objects] seed %d: %d/%d 스폰 — %s"
                  % (args.objects, n, len(placements),
                     obj_mod.describe(placements)))
            for model, why in failed:
                print("    ✘ %s: %s" % (model, why))

        if not args.no_drive:
            ok = world.set_manual_control()
            print("[ego] 외부 제어 모드 전환: %s" % ("성공" if ok else "실패(계속 시도)"))

        st = spot.start
        for n, state_name in enumerate(states, 1):
            print("\n--- [%d/%d] %s ---" % (n, len(states), state_name))

            # 1) 시작점으로 이동
            world.teleport_ego(st["x"], st["y"], st["z"], st["yaw_deg"],
                               settle_sec=args.settle)
            pos = world.ego_state()
            err = math.hypot(pos["x"] - st["x"], pos["y"] - st["y"])
            print("  텔레포트: (%.2f, %.2f) 오차 %.2fm" % (pos["x"], pos["y"], err))
            if err > 2.0:
                print("  ⚠ 텔레포트 오차가 크다. 시작점이 도로 밖이거나 충돌 중일 수 있다.")

            # 2) 신호등 연출 + 검증
            applied_at = time.time()
            if spot.tl_ids and state_name != "unknown":
                ok, info = tlc.apply(spot.tl_ids, state_name)
                obs = info["observed"]
                print("  신호등 연출: %s → %s | ego 관측=%s %s"
                      % (", ".join(spot.tl_ids), state_name,
                         _name_of(obs) if obs is not None else "-",
                         "✔" if ok else "✘ " + info["reason"]))
                applied_at = time.time()
                if not ok:
                    results.append((state_name, False, info["reason"]))
                    continue
            else:
                print("  신호등 연출 생략 (%s)"
                      % ("unknown 지점" if not spot.tl_ids else "unknown 상태"))

            if args.no_drive:
                results.append((state_name, True, "연출만 확인"))
                continue

            # 3) 주행
            ctx = {
                "label": state_name, "label_index": tl_states.label_index(state_name),
                "spot_id": spot.spot_id, "kind": spot.kind,
                "weather": args.weather or world.get_env().get("weather", "?"),
                "hour": args.hour if args.hour is not None else world.get_env().get("hour", "?"),
                "object_seed": args.objects if args.objects is not None else 0,
                "state": state_name,
                "objects": " ".join(p.model for p in placements) or "none",
                "run_id": run_id,
            }
            ok, dinfo = drive_spot(world, spot, DriveParams.from_args(args),
                                   rec=rec, writer=writer, ctx=ctx,
                                   not_before=applied_at, tl_points=tl_points)
            if args.verbose:
                sys.stdout.write("\r\033[K")
            seen = ", ".join("%s×%d" % (_name_of(c), n_)
                             for c, n_ in sorted(dinfo["tl_seen"].items(),
                                                 key=lambda t: -t[1]))
            print("  주행: %s | %.1fs, %d틱, 최대이탈 %.2fm%s"
                  % ("✔ " + dinfo["reason"] if ok else "✘ " + dinfo["reason"],
                     dinfo["elapsed"], dinfo["ticks"], dinfo["max_cte"],
                     ", 저장 %d장 (멀어서 건너뜀 %d)"
                     % (dinfo.get("saved", 0), dinfo.get("skipped_far", 0))
                     if args.save else ""))
            print("  주행 중 ego 가 본 신호: %s" % (seen or "-"))
            results.append((state_name, ok, dinfo["reason"]))

    except KeyboardInterrupt:
        print("\n\n중단됨. ego 정지.")
        try:
            world.brake_stop()
        except Exception:
            pass
        return 130
    finally:
        try:
            world.brake_stop()
        except Exception:
            pass
        if spawner is not None:
            spawner.clear()
        if writer is not None:
            print("\n[save] 총 %d장 → %s" % (writer.count, writer.manifest_path))
            writer.close()
        if rec is not None:
            rec.close()
        world.close()

    print("\n" + "=" * 70)
    if tlc.skipped_verifications:
        print("⚠ 라벨 검증을 건너뛴 주행 %d회 (ego 가 신호등 미감지). "
              "--strict-verify 로 실패 처리할 수 있다." % tlc.skipped_verifications)
    n_ok = sum(1 for _, ok, _ in results if ok)
    print("결과: %d/%d 성공" % (n_ok, len(results)))
    for name, ok, reason in results:
        print("  %-12s %s  %s" % (name, "✔" if ok else "✘", reason))
    print("=" * 70)
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
