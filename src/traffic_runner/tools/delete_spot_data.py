#!/usr/bin/env python3
"""
특정 지점의 수집 데이터를 통째로 지운다 (재수집용).

manifest.csv 에서 그 지점 행을 빼고, 이미지 폴더를 지우고, progress.txt 에서
그 지점 줄을 지운다(→ 재수집 시 다시 수집됨).

  python3 tools/delete_spot_data.py sig_005 sig_006
  python3 tools/delete_spot_data.py sig_006 --dry-run
"""

import argparse
import csv
import os
import shutil
import sys

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS_ROOT = os.path.dirname(os.path.dirname(PKG_ROOT))


def main(argv=None):
    ap = argparse.ArgumentParser(description="지점 수집 데이터 삭제")
    ap.add_argument("spots", nargs="+", help="지울 spot_id 들")
    ap.add_argument("--dataset-dir", default=os.path.join(WS_ROOT, "dataset"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.dataset_dir)
    man = os.path.join(root, "manifest.csv")
    targets = set(args.spots)

    if not os.path.exists(man):
        print("manifest 없음: %s" % man)
        return 1

    with open(man) as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)
    keep = [r for r in rows if r["spot_id"] not in targets]
    removed = len(rows) - len(keep)

    # 이미지 폴더
    img_dirs = []
    images = os.path.join(root, "images")
    if os.path.isdir(images):
        for env in os.listdir(images):
            for sid in targets:
                d = os.path.join(images, env, sid)
                if os.path.isdir(d):
                    img_dirs.append(d)

    # progress
    prog = os.path.join(root, "progress.txt")
    prog_removed = 0
    prog_keep = []
    if os.path.exists(prog):
        for line in open(prog):
            sid = line.split("|")[2] if line.count("|") >= 2 else ""
            if sid in targets:
                prog_removed += 1
            else:
                prog_keep.append(line)

    print("대상 지점: %s" % ", ".join(sorted(targets)))
    print("  매니페스트 삭제 행 : %d / %d" % (removed, len(rows)))
    print("  이미지 폴더        : %d개" % len(img_dirs))
    print("  progress 삭제 줄   : %d" % prog_removed)

    if args.dry_run:
        print("\n[dry-run] 아무것도 안 지웠다.")
        for d in img_dirs[:6]:
            print("  삭제 예정: %s" % os.path.relpath(d, root))
        return 0

    if removed == 0 and not img_dirs and prog_removed == 0:
        print("지울 데이터가 없다.")
        return 0

    shutil.copy2(man, man + ".bak")
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in keep:
            w.writerow(r)
    for d in img_dirs:
        shutil.rmtree(d, ignore_errors=True)
    if os.path.exists(prog):
        with open(prog, "w") as f:
            f.writelines(prog_keep)

    print("\n삭제 완료 (매니페스트 백업: manifest.csv.bak)")
    print("남은 지점: %s" % ", ".join(sorted(set(r["spot_id"] for r in keep))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
