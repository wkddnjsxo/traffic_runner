#!/usr/bin/env python3
"""
수집된 프레임 일부를 지우고 번호를 다시 매긴다.

신호등이 화각 밖이거나 뒷면만 보이는 구간은 라벨이 맞아도 학습에 해롭다
(실측: sig_006 30-45m 구간 정확도 46.8% — 찍기 수준).
그런 구간을 걷어낼 때 쓴다.

  # 앞쪽 N 프레임 삭제 (주행마다 적용)
  python3 tools/prune_frames.py --spot sig_006 --drop-first 51

  # 신호등이 이 거리보다 먼 프레임 삭제
  python3 tools/prune_frames.py --spot sig_006 --max-dist 39

  # 미리보기 (실제로 안 지움)
  python3 tools/prune_frames.py --spot sig_006 --drop-first 51 --dry-run

이미지 파일과 manifest.csv 를 함께 정리하고, 남은 프레임을 0 부터 다시 번호매긴다.
"""

import argparse
import csv
import os
import shutil
import sys

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS_ROOT = os.path.dirname(os.path.dirname(PKG_ROOT))

DRIVE_KEYS = ("weather", "hour", "spot_id", "object_seed", "state")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="프레임 삭제 + 번호 재정렬",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--dataset-dir", default=os.path.join(WS_ROOT, "dataset"))
    ap.add_argument("--spot", required=True, help="대상 spot_id (쉼표로 여러개)")
    ap.add_argument("--drop-first", type=int, default=None,
                    help="각 주행에서 앞쪽 N 프레임 삭제 (0..N-1)")
    ap.add_argument("--max-dist", type=float, default=None,
                    help="신호등까지 이 거리보다 먼 프레임 삭제(m)")
    ap.add_argument("--drop-detected", action="store_true",
                    help="시뮬이 신호등을 감지한 프레임 삭제 (unknown 오염 제거용). "
                         "tl_color_observed 가 -2/0 이 아닌 프레임을 지운다")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.drop_first is None and args.max_dist is None and not args.drop_detected:
        print("--drop-first / --max-dist / --drop-detected 중 하나는 줘야 한다.")
        return 1

    root = os.path.abspath(args.dataset_dir)
    man = os.path.join(root, "manifest.csv")
    if not os.path.exists(man):
        print("manifest 없음: %s" % man)
        return 1

    targets = {s.strip() for s in args.spot.split(",") if s.strip()}

    with open(man) as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)
    print("매니페스트 %d행" % len(rows))

    # 주행별로 묶어서 판정
    drives = {}
    for r in rows:
        key = tuple(r[k] for k in DRIVE_KEYS)
        drives.setdefault(key, []).append(r)

    keep, drop = [], []
    for key, rs in drives.items():
        if key[2] not in targets:
            keep.extend(rs)
            continue
        rs.sort(key=lambda r: int(r["frame_idx"]))
        for r in rs:
            bad = False
            if args.drop_first is not None and int(r["frame_idx"]) < args.drop_first:
                bad = True
            if args.max_dist is not None and r.get("dist_to_tl_m"):
                try:
                    if float(r["dist_to_tl_m"]) > args.max_dist:
                        bad = True
                except ValueError:
                    pass
            if args.drop_detected:
                try:
                    c = int(r.get("tl_color_observed", -2))
                    if c not in (-2, 0):
                        bad = True
                except (ValueError, TypeError):
                    pass
            (drop if bad else keep).append(r)

    print("삭제 대상 %d행, 유지 %d행" % (len(drop), len(keep)))
    if drop:
        d = [float(r["dist_to_tl_m"]) for r in drop if r.get("dist_to_tl_m")]
        if d:
            print("  삭제 구간 거리: %.1f ~ %.1fm" % (min(d), max(d)))
    if not drop:
        print("삭제할 게 없다.")
        return 0

    # 유지분을 주행별로 다시 0 부터 번호매김 (파일도 함께 이동)
    keep_by_drive = {}
    for r in keep:
        keep_by_drive.setdefault(tuple(r[k] for k in DRIVE_KEYS), []).append(r)

    renames = []      # (old_abs, new_abs, row, new_idx)
    for key, rs in keep_by_drive.items():
        if key[2] not in targets:
            continue
        rs.sort(key=lambda r: int(r["frame_idx"]))
        for new_idx, r in enumerate(rs):
            old_rel = r["image_path"]
            new_rel = os.path.join(os.path.dirname(old_rel), "%06d.jpg" % new_idx)
            if old_rel != new_rel:
                renames.append((os.path.join(root, old_rel),
                                os.path.join(root, new_rel), r, new_idx))
            else:
                r["frame_idx"] = str(new_idx)

    print("번호 재정렬 대상 %d개 파일" % len(renames))

    if args.dry_run:
        print("\n[dry-run] 아무것도 안 지웠다. 예시:")
        for r in drop[:3]:
            print("  삭제  %s" % r["image_path"])
        for old, new, _, _ in renames[:3]:
            print("  이동  %s → %s"
                  % (os.path.relpath(old, root), os.path.relpath(new, root)))
        return 0

    # 1) 삭제
    n_del = 0
    for r in drop:
        p = os.path.join(root, r["image_path"])
        if os.path.exists(p):
            os.remove(p)
            n_del += 1
    print("이미지 %d개 삭제" % n_del)

    # 2) 번호 재정렬. 겹침을 피하려고 임시 이름을 거친다.
    tmp = []
    for old, new, r, idx in renames:
        if os.path.exists(old):
            t = old + ".tmp_prune"
            os.rename(old, t)
            tmp.append((t, new, r, idx))
    for t, new, r, idx in tmp:
        os.rename(t, new)
        r["image_path"] = os.path.relpath(new, root)
        r["frame_idx"] = str(idx)
    print("이미지 %d개 번호 재정렬" % len(tmp))

    # 3) 매니페스트 다시 쓰기 (백업 남김)
    shutil.copy2(man, man + ".bak")
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in keep:
            w.writerow(r)
    print("매니페스트 갱신: %d행 (백업 %s.bak)" % (len(keep), os.path.basename(man)))

    # 4) progress.txt 는 건드리지 않는다.
    #    재수집하고 싶으면 해당 조합 줄을 지워야 한다고 알려준다.
    print("\n주의: progress.txt 는 그대로다. 이 지점을 다시 수집하려면")
    print("      dataset/progress.txt 에서 해당 줄을 지우거나 --restart 를 쓸 것.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
