#!/usr/bin/env python3
"""
데이터셋 무결성 검사 — 학습에 넣기 전에 이걸로 확인한다.

  python3 check_dataset.py                 # 기본 dataset/
  python3 check_dataset.py --data ../dataset --deep

검사 항목:
  1. 매니페스트 형식 (컬럼, 행 수, 중복)
  2. 이미지 파일 존재 / 실제로 열리는가 / 크기가 일정한가
  3. 라벨 <-> label_index 일관성, 클래스 이름이 학습 코드와 같은가
  4. 폴더 경로와 메타데이터가 일치하는가 (weather/spot/seed/state)
  5. 라벨 <-> 시뮬 관측색(tl_color_observed) 교차검증
  6. train.py 가 실제로 읽을 수 있는가 (Dataset 통과)
"""

import argparse
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from dataset import CLASS_NAMES, MANIFEST_COLS_REQUIRED  # noqa: E402

#: 라벨 이름 -> MORAI 가 보고하는 tl_color 값 (수집기 tl/states.py 와 같아야 함)
MORAI_VALUE = {"red": 1, "yellow": 4, "green": 16, "red_yellow": 5,
               "red_left": 33, "green_left": 48, "left": 32}


def main(argv=None):
    ap = argparse.ArgumentParser(description="데이터셋 무결성 검사")
    ap.add_argument("--data", default=os.environ.get("TR_DATA", "/workspace/dataset"))
    ap.add_argument("--deep", action="store_true",
                    help="모든 이미지를 실제로 열어본다 (느림). 기본은 표본 300장")
    ap.add_argument("--sample", type=int, default=300)
    args = ap.parse_args(argv)

    import pandas as pd
    from PIL import Image

    root = os.path.abspath(args.data)
    man = os.path.join(root, "manifest.csv")
    problems = []
    warns = []

    print("=" * 72)
    print("데이터셋 검사: %s" % root)
    print("=" * 72)

    if not os.path.exists(man):
        print("✘ manifest.csv 가 없다: %s" % man)
        return 1
    df = pd.read_csv(man)
    print("\n[1] 매니페스트")
    print("    행 수        : %d" % len(df))
    print("    컬럼 %d개    : %s" % (len(df.columns), ", ".join(df.columns[:6]) + " ..."))

    missing = [c for c in MANIFEST_COLS_REQUIRED if c not in df.columns]
    if missing:
        problems.append("필수 컬럼 없음: %s" % ", ".join(missing))
    dup = int(df["image_path"].duplicated().sum())
    if dup:
        warns.append("image_path 중복 %d행 (학습 시 마지막 것만 사용됨)" % dup)
    print("    중복 경로    : %d %s" % (dup, "(재수집분, 자동 처리됨)" if dup else ""))

    # ---- 2. 이미지 ----
    print("\n[2] 이미지 파일")
    paths = df["image_path"].tolist()
    check = paths if args.deep else paths[:: max(1, len(paths) // args.sample)]
    missing_files, unreadable, sizes = [], [], collections.Counter()
    for rel in check:
        full = os.path.join(root, rel)
        if not os.path.exists(full):
            missing_files.append(rel)
            continue
        try:
            with Image.open(full) as im:
                im.verify()
            with Image.open(full) as im:
                sizes[im.size] += 1
        except Exception as exc:
            unreadable.append((rel, str(exc)[:40]))
    print("    검사 %d장 (%s)" % (len(check), "전체" if args.deep else "표본"))
    print("    누락        : %d" % len(missing_files))
    print("    열기 실패   : %d" % len(unreadable))
    print("    해상도      : %s" % dict(sizes))
    if missing_files:
        problems.append("이미지 파일 누락 %d개 (예: %s)" % (len(missing_files), missing_files[0]))
    if unreadable:
        problems.append("깨진 이미지 %d개 (예: %s)" % (len(unreadable), unreadable[0]))
    if len(sizes) > 1:
        warns.append("해상도가 섞여 있다: %s (리사이즈되므로 치명적이진 않음)" % dict(sizes))

    # ---- 3. 라벨 일관성 ----
    print("\n[3] 라벨")
    bad_idx = []
    for lab, grp in df.groupby("label"):
        if lab not in CLASS_NAMES:
            problems.append("학습 코드가 모르는 라벨: %s" % lab)
            continue
        want = CLASS_NAMES.index(lab)
        wrong = grp[grp["label_index"] != want]
        if len(wrong):
            bad_idx.append((lab, want, sorted(set(wrong["label_index"])), len(wrong)))
    if bad_idx:
        for lab, want, got, n in bad_idx:
            problems.append("label_index 불일치: %s 는 %d 여야 하는데 %s (%d행)"
                            % (lab, want, got, n))
    else:
        print("    label ↔ label_index 일치 ✔")
    counts = df["label"].value_counts()
    for c in CLASS_NAMES:
        n = int(counts.get(c, 0))
        flag = "  ← 없음!" if n == 0 else ""
        print("      %-12s %7d%s" % (c, n, flag))
        if n == 0:
            warns.append("클래스 '%s' 데이터가 없다" % c)

    # ---- 4. 경로 ↔ 메타 일치 ----
    print("\n[4] 폴더 경로 ↔ 메타데이터")
    mism = 0
    for _, r in df.head(5000).iterrows():
        parts = str(r["image_path"]).split(os.sep)
        # images/<weather>_<hour>/<spot>/seed<NN>/<state>/x.jpg
        if len(parts) < 6:
            mism += 1
            continue
        env, spot, seed, state = parts[1], parts[2], parts[3], parts[4]
        if (env != "%s_%s" % (r["weather"], r["hour"]) or spot != r["spot_id"]
                or seed != "seed%02d" % int(r["object_seed"]) or state != r["state"]):
            mism += 1
    print("    불일치      : %d / %d" % (mism, min(5000, len(df))))
    if mism:
        problems.append("경로와 메타데이터 불일치 %d행" % mism)

    # ---- 5. 라벨 ↔ 시뮬 관측색 ----
    print("\n[5] 라벨 ↔ 시뮬 관측색 (tl_color_observed)")
    if "tl_color_observed" in df.columns:
        sig = df[df["label"] != "unknown"]
        obs = pd.to_numeric(sig["tl_color_observed"], errors="coerce")
        expect = sig["label"].map(MORAI_VALUE)
        detected = obs.notna() & (obs != -2) & (obs != 0)
        conflict = int((detected & (obs != expect)).sum())
        print("    감지된 프레임 : %d / %d" % (int(detected.sum()), len(sig)))
        print("    라벨과 불일치 : %d" % conflict)
        if conflict:
            problems.append("라벨과 시뮬 관측색이 다른 프레임 %d개 — 라벨 오염" % conflict)
        else:
            print("    → 감지된 프레임은 전부 라벨과 일치 ✔")
    else:
        warns.append("tl_color_observed 컬럼이 없어 교차검증 생략")

    # ---- 6. train.py 파이프라인 ----
    print("\n[6] 학습 파이프라인 통과 시험")
    try:
        from dataset import TrafficLightDataset, load_manifest, split_by_drive
        from train import build_transforms

        d = load_manifest(root)
        tr, va = split_by_drive(d, 0.2, 42)
        ds = TrafficLightDataset(tr.head(8), root, build_transforms(224, True))
        x, y = ds[0]
        print("    Dataset[0] → 텐서 %s, 라벨 %d (%s)"
              % (tuple(x.shape), y, CLASS_NAMES[y]))
        if x.shape[0] != 3:
            problems.append("채널 수가 3이 아니다: %s" % (tuple(x.shape),))
        print("    train %d / val %d 로 분할 ✔" % (len(tr), len(va)))
    except Exception as exc:
        problems.append("학습 파이프라인 실패: %s" % exc)

    # ---- 결과 ----
    print("\n" + "=" * 72)
    if problems:
        print("문제 %d건:" % len(problems))
        for p in problems:
            print("  ✘ %s" % p)
    if warns:
        print("경고 %d건:" % len(warns))
        for w in warns:
            print("  ⚠ %s" % w)
    if not problems:
        print("✔ 학습에 넣어도 되는 상태다.")
    print("=" * 72)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
