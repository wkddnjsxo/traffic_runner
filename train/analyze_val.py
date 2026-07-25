#!/usr/bin/env python3
"""
val 오류를 지점/거리/라벨로 쪼개 본다 — 리포트 이상 원인 규명용.

  python3 analyze_val.py --ckpt runs/<시각>/best.pt
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from dataset import (CLASS_NAMES, TrafficLightDataset,  # noqa: E402
                     load_manifest, split_by_drive)
from infer import resolve_ckpt                          # noqa: E402
from train import build_model, build_transforms         # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=os.environ.get("TR_DATA", "/workspace/dataset"))
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout-spots", default=None,
                    help="이 지점들을 val 로 (학습 때와 같게 줘야 한다)")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(resolve_ckpt(args.ckpt), map_location=device, weights_only=False)
    size = int(ck.get("args", {}).get("size", 224))
    model = build_model(pretrained=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    df = load_manifest(args.data)
    holdout = ([x.strip() for x in args.holdout_spots.split(",")]
               if args.holdout_spots else None)
    _, val_df = split_by_drive(df, args.val_ratio, args.seed, holdout)
    ds = TrafficLightDataset(val_df, args.data, build_transforms(size, False))
    ld = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=device.type == "cuda")

    preds = []
    with torch.no_grad():
        for i, (x, _) in enumerate(ld):
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                preds.append(model(x).argmax(1).cpu())
            if i % 20 == 0:
                sys.stdout.write("\r  추론 %d/%d" % (i * args.batch, len(ds)))
                sys.stdout.flush()
    print("\r\033[K추론 완료")

    v = val_df.copy()
    v["pred"] = torch.cat(preds).numpy()
    v["ok"] = (v["pred"] == v["label_index"]).astype(int)
    v["dist"] = v["dist_to_tl_m"]
    print("\n전체 val 정확도 %.4f" % v["ok"].mean())

    bins = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 999)]

    def binname(d):
        if d != d:
            return "unknown"
        for a, b in bins:
            if a <= d < b:
                return "%d-%d" % (a, b if b < 999 else 99)
        return "?"

    v["bin"] = v["dist"].map(binname)

    print("\n[A] 지점 × 거리구간 정확도")
    order = ["0-15", "15-30", "30-45", "45-60", "60-99", "unknown"]
    print("%-9s" % "spot" + "".join("%12s" % o for o in order))
    for sid, g in v.groupby("spot_id"):
        cells = []
        for o in order:
            m = g[g["bin"] == o]
            cells.append("%5.1f%%(%d)" % (100 * m["ok"].mean(), len(m)) if len(m)
                         else "      -    ")
        print("%-9s" % sid + "".join("%12s" % c for c in cells))

    print("\n[B] 30-45m 구간에서 틀린 것 (지점/라벨/예측)")
    bad = v[(v["bin"] == "30-45") & (v["ok"] == 0)]
    print("  틀린 프레임 %d / %d" % (len(bad), int((v["bin"] == "30-45").sum())))
    cnt = bad.groupby(["spot_id", "label", "pred"]).size().sort_values(ascending=False)
    for (sid, lab, pr), n in cnt.head(15).items():
        print("    %-9s %-11s → %-11s %5d" % (sid, lab, CLASS_NAMES[pr], n))

    print("\n[C] 라벨 × 거리구간 정확도")
    print("%-12s" % "label" + "".join("%12s" % o for o in order))
    for lab, g in v.groupby("label"):
        cells = []
        for o in order:
            m = g[g["bin"] == o]
            cells.append("%5.1f%%(%d)" % (100 * m["ok"].mean(), len(m)) if len(m)
                         else "      -    ")
        print("%-12s" % lab + "".join("%12s" % c for c in cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
