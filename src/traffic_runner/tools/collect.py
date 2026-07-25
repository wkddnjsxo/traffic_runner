#!/usr/bin/env python3
"""
학습 데이터셋 자동 수집 — 전체 매트릭스 주행.

  환경(날씨×시간) → 지점 → 객체seed → 신호상태(랜덤 순서)

  # 무엇을 얼마나 돌릴지 먼저 확인 (시뮬 불필요)
  python3 tools/collect.py --plan

  # 실제 수집 (중단되면 같은 명령으로 재개된다)
  python3 tools/collect.py

  # 일부만
  python3 tools/collect.py --weather SUNNY --hour 11
  python3 tools/collect.py --spots sig_004,sig_005

  # 처음부터 다시 (진행상황 무시)
  python3 tools/collect.py --restart

중단(Ctrl-C)해도 완료한 조합은 dataset/progress.txt 에 남아, 재실행하면 이어서 한다.
"""

import argparse
import math
import os
import signal
import sys
import time

import yaml

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS_ROOT = os.path.dirname(os.path.dirname(PKG_ROOT))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from collect import matrix, objects as obj_mod         # noqa: E402
from drive.follow import DriveParams, arrival_radius_for, drive_spot   # noqa: E402
from spot import mgeo as mgeo_mod                      # noqa: E402
from spot import schema                                # noqa: E402
from tl import states as tl_states                     # noqa: E402
from tl.controller import TrafficLightController, _name_of   # noqa: E402


_stop = {"flag": False}


def _on_sigint(sig, frame):
    if _stop["flag"]:
        print("\n강제 종료.")
        sys.exit(130)
    _stop["flag"] = True
    print("\n\n중단 요청됨 — 현재 주행을 마치고 정리합니다. (한 번 더 누르면 강제 종료)")


def load_cfg(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg_dir = os.path.dirname(os.path.abspath(path))
    for key in ("grpc_src", "mgeo_dir"):
        p = cfg["paths"].get(key)
        if p and not os.path.isabs(p):
            cfg["paths"][key] = os.path.normpath(os.path.join(cfg_dir, p))
    return cfg


def fmt_dur(sec):
    if sec < 60:
        return "%.0f초" % sec
    if sec < 3600:
        return "%.0f분" % (sec / 60)
    return "%.1f시간" % (sec / 3600)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="학습 데이터셋 자동 수집",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--config", default=os.path.join(PKG_ROOT, "config", "runtime.yaml"))
    ap.add_argument("--spots-dir", default=os.path.join(WS_ROOT, "spots"))
    ap.add_argument("--dataset-dir", default=os.path.join(WS_ROOT, "dataset"))
    ap.add_argument("--plan", action="store_true", help="계획만 출력하고 종료")
    ap.add_argument("--restart", action="store_true", help="진행상황 무시하고 처음부터")
    ap.add_argument("--no-save", action="store_true", help="이미지 저장 없이 주행만")
    # 범위 제한
    ap.add_argument("--weather", default=None, help="이 날씨만 (쉼표로 여러개)")
    ap.add_argument("--hour", default=None, help="이 시간만 (쉼표로 여러개)")
    ap.add_argument("--spots", default=None, help="이 지점만 (쉼표로 여러개)")
    ap.add_argument("--seeds", default=None, help="이 객체 seed 만 (쉼표로 여러개)")
    ap.add_argument("--states", default=None,
                    help="이 신호 상태만 수집 (쉼표로 여러개). 예: left,red_left,green_left")
    ap.add_argument("--limit", type=int, default=None, help="이번 실행에서 최대 N개 조합만")
    # 주행
    ap.add_argument("--speed", type=float, default=8.0)
    ap.add_argument("--control-hz", type=float, default=20.0)
    ap.add_argument("--lookahead-k", type=float, default=0.6)
    ap.add_argument("--lookahead-min", type=float, default=4.0)
    ap.add_argument("--max-steer", type=float, default=35.0)
    ap.add_argument("--max-cte", type=float, default=6.0)
    ap.add_argument("--arrival-radius", type=float, default=None, metavar="M",
                    help="도달 판정 반경. 생략하면 차량 기하로 자동 계산한다 "
                         "(뒷축→앞범퍼 3.845m + stop-gap + 제동여유)")
    ap.add_argument("--stop-gap", type=float, default=None, metavar="M",
                    help="앞범퍼가 끝점에서 이만큼 앞에 서도록 (기본: runtime.yaml "
                         "collect.stop_gap_m)")
    ap.add_argument("--save-until", type=float, default=None, metavar="M",
                    help="신호등이 이보다 가까우면 저장 중단 (기본: runtime.yaml "
                         "collect.save_until_m). 너무 가까우면 신호등이 화면 위로 "
                         "벗어난다. 0 이면 제한 없음")
    ap.add_argument("--slowdown", type=float, default=18.0, metavar="M",
                    help="끝점 앞 이 거리부터 감속 시작 (0이면 감속 없음)")
    ap.add_argument("--stop-speed", type=float, default=1.5, metavar="MPS",
                    help="끝점에서의 목표 속도(m/s)")
    ap.add_argument("--end-hold", type=float, default=0.5)
    ap.add_argument("--save-from", type=float, default=None, metavar="M",
                    help="신호등까지 이 거리 안에서만 프레임 저장 (기본: runtime.yaml "
                         "collect.save_from_m). 0 이면 전부 저장")
    ap.add_argument("--settle", type=float, default=0.6, help="연출/텔레포트 후 대기(s)")
    ap.add_argument("--env-settle", type=float, default=2.0,
                    help="날씨/시간 변경 후 렌더 안정화 대기(s)")
    ap.add_argument("--strict-verify", action="store_true",
                    help="ego 가 신호등 미감지 시 실패 처리 (라벨 미검증 주행 방지)")
    ap.add_argument("--no-sibling", action="store_true")
    ap.add_argument("--retry", type=int, default=1, help="주행 실패 시 재시도 횟수")
    ap.add_argument("--quiet", dest="verbose", action="store_false", default=True)
    args = ap.parse_args(argv)

    cfg = load_cfg(args.config)
    col = cfg.get("collect", {})

    spots_all = schema.load_all(args.spots_dir, strict=False)
    if not spots_all:
        print("지점이 없다: %s" % args.spots_dir)
        return 1
    if args.spots:
        want = {s.strip() for s in args.spots.split(",") if s.strip()}
        spots = [s for s in spots_all if s.spot_id in want]
        missing = want - {s.spot_id for s in spots}
        if missing:
            print("없는 지점: %s" % ", ".join(sorted(missing)))
            return 1
    else:
        spots = spots_all

    weathers = ([w.strip().upper() for w in args.weather.split(",")]
                if args.weather else list(col.get("weathers", ["SUNNY", "FOGGY"])))
    hours = ([int(h) for h in args.hour.split(",")]
             if args.hour else list(col.get("hours", [11, 13, 15])))
    n_seeds = int(col.get("object_seeds", 3))
    if args.save_from is None:
        args.save_from = col.get("save_from_m", 70.0)
    if not args.save_from:            # 0 또는 null -> 제한 없음
        args.save_from = None
    if args.save_until is None:
        args.save_until = col.get("save_until_m", 0)
    if not args.save_until:
        args.save_until = None
    if args.arrival_radius is None:
        # 끝점에 바짝 세운다(뒷축 기준). 끝점 이미지까지 저장하려면 작아야 한다.
        # 예전엔 차량 기하로 4.55m 를 잡아 앞범퍼를 정지선에 맞췄는데, 그러면
        # 끝점 4m 앞에서 저장이 끊겼다.
        args.arrival_radius = float(cfg.get("collect", {}).get("arrival_radius_m", 0.8))
    seed_filter = ({int(s) for s in args.seeds.split(",")} if args.seeds else None)

    combos = matrix.build(spots, weathers, hours, n_seeds,
                          shuffle_states=bool(col.get("shuffle_states", True)),
                          seed=col.get("random_seed"))
    if seed_filter is not None:
        combos = [c for c in combos if c.object_seed in seed_filter]
    if args.states:
        want_states = {t.strip() for t in args.states.split(",") if t.strip()}
        for st in want_states:
            tl_states.get(st)          # 유효성 검증
        combos = [c for c in combos if c.state in want_states]
        if not combos:
            print("해당 상태의 조합이 없다: %s" % ", ".join(sorted(want_states)))
            return 1

    progress_path = os.path.join(args.dataset_dir, "progress.txt")
    if args.restart and os.path.exists(progress_path):
        os.rename(progress_path, progress_path + ".bak")
        print("진행상황을 %s.bak 로 옮기고 처음부터 시작한다." % progress_path)
        man = os.path.join(args.dataset_dir, "manifest.csv")
        if os.path.exists(man):
            print("⚠ 기존 %s 에 이어쓰므로 같은 조합의 행이 중복된다.\n"
                  "   깨끗하게 다시 모으려면 dataset/ 를 먼저 비울 것." % man)
    prog = matrix.Progress(progress_path)

    todo = [c for c in combos if not prog.is_done(c)]
    if args.limit:
        todo = todo[:args.limit]

    # 예상 소요시간: 경로길이/속도 + 텔레포트·연출·정지 오버헤드
    by_id = {s.spot_id: s for s in spots}
    overhead = args.settle * 2 + args.end_hold + 1.0
    est = sum(by_id[c.spot_id].path_length_m / max(args.speed * 0.75, 1.0) + overhead
              for c in todo)

    print("=" * 74)
    print("수집 계획")
    print("  환경     : %s × %s (%d조합)"
          % ("/".join(weathers), "/".join(str(h) for h in hours),
             len(weathers) * len(hours)))
    print("  지점     : %d개  객체 seed: %d" % (len(spots), n_seeds))
    if args.states:
        print("  신호 상태: %s (지정한 것만)" % args.states)
    print("  전체 조합: %d회" % len(combos))
    print("  완료     : %d회" % (len(combos) - len([c for c in combos
                                                   if not prog.is_done(c)])))
    print("  이번 실행: %d회  (예상 %s)" % (len(todo), fmt_dur(est)))
    print("  정지     : 앞범퍼가 끝점 %.1fm 앞 (도달반경 %.2fm, 뒷축 기준)"
          % (float(args.stop_gap if args.stop_gap is not None
                   else col.get("stop_gap_m", 0.5)), args.arrival_radius))
    print("  저장 범위: 신호등 %s ~ %s"
          % ("%.0fm" % args.save_from if args.save_from else "무제한",
             "%.0fm" % args.save_until if args.save_until else "0m"))
    print("  저장 위치: %s" % args.dataset_dir)
    print("=" * 74)

    if args.plan:
        groups = matrix.group_runs(todo)
        print("\n실행 순서 (앞부분 %d개 그룹):" % min(8, len(groups)))
        for (w, h, sid, sd), cs in groups[:8]:
            print("  %s %s시 / %s / seed%d → %s"
                  % (w, h, sid, sd, ", ".join(c.state for c in cs)))
        if len(groups) > 8:
            print("  ... 외 %d개 그룹" % (len(groups) - 8))
        prog.close()
        return 0

    if not todo:
        print("\n남은 조합이 없다. 수집 완료.")
        prog.close()
        return 0

    signal.signal(signal.SIGINT, _on_sigint)

    # ---- 준비 ----
    mg = mgeo_mod.MGeo(cfg["paths"]["mgeo_dir"])
    tl_pos = {}
    for n in mg.nodes.values():
        t = n.get("traffic_light_id")
        if t:
            tl_pos.setdefault(t, []).append((n["point"][0], n["point"][1]))
    plans = obj_mod.plan_all(spots_all, n_seeds, mg, cfg.get("objects", {}))

    from sim.world import World

    world = World(cfg)
    tlc = TrafficLightController(world, sibling=not args.no_sibling,
                                 settle_sec=args.settle, strict=args.strict_verify)
    params = DriveParams.from_args(args)

    rec = writer = None
    run_id = time.strftime("%Y%m%d_%H%M%S")
    if not args.no_save:
        from collect.recorder import CameraRecorder, DatasetWriter

        rec = CameraRecorder(topic=cfg.get("ros", {}).get(
            "image_topic", "/image_jpeg/compressed"))
        writer = DatasetWriter(args.dataset_dir, run_id)

    spawner = obj_mod.ObjectSpawner(world)
    world.set_manual_control()

    stats = {"ok": 0, "fail": 0, "saved": 0, "unverified": 0}
    failures = []
    cur_env = cur_objects = None
    t_start = time.time()

    try:
        run_groups = matrix.group_runs(todo)
        for gi, ((weather, hour, spot_id, obj_seed), group) in enumerate(
                run_groups, 1):
            if _stop["flag"]:
                break
            spot = by_id[spot_id]

            # ---- 환경 (가장 바깥 루프) ----
            if (weather, hour) != cur_env:
                world.set_weather(weather)
                world.set_time_hour(hour)
                time.sleep(args.env_settle)   # 렌더 안정화
                cur_env = (weather, hour)
                print("\n[환경] %s / %d시" % (weather, hour))

            # ---- 객체 (신호 상태보다 바깥 = 배치 고정) ----
            obj_key = (spot_id, obj_seed)
            if obj_key != cur_objects:
                spawner.clear()
                placements, _ = plans.get(obj_key, ([], {}))
                if placements:
                    n_sp, failed = spawner.spawn(placements)
                    for model, why in failed:
                        print("    ✘ 객체 %s: %s" % (model, why))
                cur_objects = obj_key
            placements, _ = plans.get(obj_key, ([], {}))

            tl_points = [p for t in spot.tl_ids for p in tl_pos.get(t, [])]
            print("\n[%d/%d] %s / seed%d — %s (%s)"
                  % (gi, len(run_groups), spot_id, obj_seed,
                     ", ".join(c.state for c in group),
                     obj_mod.describe(placements)))

            for combo in group:
                if _stop["flag"]:
                    break
                ok, unverified = run_one(world, tlc, spot, combo, params, rec,
                                         writer, tl_points, placements, run_id, args)
                if unverified:
                    stats["unverified"] += 1
                if ok:
                    prog.mark(combo)
                    stats["ok"] += 1
                else:
                    stats["fail"] += 1
                    failures.append(combo.key)

            if writer is not None:
                stats["saved"] = writer.count

    except KeyboardInterrupt:
        print("\n중단됨.")
    finally:
        print("\n정리 중...")
        try:
            world.brake_stop()
        except Exception:
            pass
        spawner.clear()
        if writer is not None:
            writer.close()
        if rec is not None:
            rec.close()
        world.close()
        prog.close()

    elapsed = time.time() - t_start
    remain = len(combos) - len(prog.done)
    print("\n" + "=" * 74)
    print("이번 실행: 성공 %d / 실패 %d  (%s)" % (stats["ok"], stats["fail"],
                                              fmt_dur(elapsed)))
    if writer is not None:
        print("저장한 이미지: %d장 → %s" % (stats["saved"], writer.manifest_path))
    if stats["unverified"]:
        print("⚠ 주행 내내 신호등을 못 봐서 라벨을 검증하지 못한 주행 %d회."
              % stats["unverified"])
    if failures:
        print("실패한 조합 %d개 (재실행하면 다시 시도):" % len(failures))
        for k in failures[:10]:
            print("   %s" % k)
        if len(failures) > 10:
            print("   ... 외 %d개" % (len(failures) - 10))
    print("전체 진행: %d/%d 완료, %d 남음" % (len(prog.done), len(combos), remain))
    print("=" * 74)
    return 0


def run_one(world, tlc, spot, combo, params, rec, writer, tl_points,
            placements, run_id, args):
    """조합 하나: 시작점 이동 → 신호 연출 → 주행."""
    st = spot.start
    for attempt in range(args.retry + 1):
        world.teleport_ego(st["x"], st["y"], st["z"], st["yaw_deg"],
                           settle_sec=args.settle)

        applied_at = time.time()
        expect_color = None
        if spot.tl_ids and combo.state != "unknown":
            expect_color = tl_states.get(combo.state).morai_value
            # 주행 전 확인은 '보이면 확인, 안 보이면 통과'. 진짜 검증은 주행 중에 한다.
            ok, info = tlc.apply(spot.tl_ids, combo.state)
            applied_at = time.time()
            if not ok:
                print("    %-11s ✘ 신호 연출 실패: %s" % (combo.state, info["reason"]))
                if attempt < args.retry:
                    continue
                return False, False

        ctx = {
            "label": combo.state,
            "label_index": tl_states.label_index(combo.state),
            "spot_id": spot.spot_id, "kind": spot.kind,
            "weather": combo.weather, "hour": combo.hour,
            "object_seed": combo.object_seed, "state": combo.state,
            "objects": " ".join(p.model for p in placements) or "none",
            "run_id": run_id,
        }
        ok, info = drive_spot(world, spot, params, rec=rec, writer=writer,
                              ctx=ctx, not_before=applied_at,
                              tl_points=tl_points,
                              should_stop=lambda: _stop["flag"],
                              expect_color=expect_color)
        if params.verbose:
            sys.stdout.write("\r\033[K")
        mark = "✔" if ok else "✘"
        vf = info.get("verified_frames", 0)
        print("    %-11s %s %-22s %5.1fs  cte %.2f  저장 %d  검증 %d틱"
              % (combo.state, mark, info["reason"], info["elapsed"],
                 info["max_cte"], info["saved"], vf))
        if ok and expect_color is not None and vf == 0:
            print("      ⚠ 주행 내내 신호등 미감지 — 라벨을 검증하지 못했다")
        if ok:
            return True, (expect_color is not None and vf == 0)
        if _stop["flag"]:
            return False, False
        if attempt < args.retry:
            print("      재시도 %d/%d" % (attempt + 1, args.retry))
    return False, False


if __name__ == "__main__":
    sys.exit(main())
