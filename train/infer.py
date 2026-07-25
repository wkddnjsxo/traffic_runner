#!/usr/bin/env python3
"""
학습한 모델로 추론.

  # 이미지 파일 / 폴더
  python3 infer.py --ckpt runs/<시각>/best.pt --images some.jpg
  python3 infer.py --ckpt runs/<시각>/best.pt --images dir/ --topk 3

  # 매니페스트로 정확도 재측정 (수집한 데이터 일부 검증용)
  python3 infer.py --ckpt runs/<시각>/best.pt --manifest ../dataset/manifest.csv --limit 2000

컨테이너 안에서 돌린다 (torch 필요). run.sh 에 infer 명령을 추가해 뒀다.
"""

import argparse
import glob
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from train import build_model, build_transforms   # noqa: E402


def resolve_ckpt(ckpt_path):
    """
    체크포인트 경로를 찾는다.

    컨테이너 작업 디렉터리는 /workspace 인데 체크포인트는 /workspace/train/runs/ 에
    있어서, 'runs/...' 처럼 짧게 줘도 찾아준다. 여러 후보를 순서대로 시도한다.
    """
    if os.path.isfile(ckpt_path):
        return ckpt_path
    candidates = [
        ckpt_path,
        os.path.join(HERE, ckpt_path),                    # train/ 기준
        os.path.join(HERE, "runs", ckpt_path),            # runs/ 안이라고 가정
    ]
    # 'best' 나 실행폴더 이름만 줬을 때 best.pt 로 보정
    if not ckpt_path.endswith(".pt"):
        candidates.append(os.path.join(HERE, "runs", ckpt_path, "best.pt"))
        candidates.append(os.path.join(HERE, ckpt_path, "best.pt"))
    for c in candidates:
        if os.path.isfile(c):
            return c
    runs_dir = os.path.join(HERE, "runs")
    avail = []
    if os.path.isdir(runs_dir):
        for d in sorted(os.listdir(runs_dir)):
            if os.path.isfile(os.path.join(runs_dir, d, "best.pt")):
                avail.append("runs/%s/best.pt" % d)
    raise FileNotFoundError(
        "체크포인트를 못 찾았다: %s\n  있는 것:\n    %s"
        % (ckpt_path, "\n    ".join(avail) or "(runs/ 에 없음)"))


def load_model(ckpt_path, device):
    ckpt_path = resolve_ckpt(ckpt_path)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = ck.get("classes")
    size = int(ck.get("args", {}).get("size", 224))
    model = build_model(pretrained=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    tf = build_transforms(size, train_aug=False)
    print("[infer] %s | 클래스 %d개 | val_acc(저장시) %.3f"
          % (os.path.basename(ckpt_path), len(classes), ck.get("val_acc", -1)))
    return model, tf, classes


@torch.no_grad()
def predict_batch(model, tf, paths, device, classes, topk=1):
    """이미지 경로 리스트 -> [(path, [(라벨, 확률), ...])]."""
    imgs = []
    ok_paths = []
    for p in paths:
        try:
            imgs.append(tf(Image.open(p).convert("RGB")))
            ok_paths.append(p)
        except Exception as exc:
            print("  ✘ %s: %s" % (p, exc))
    if not imgs:
        return []
    x = torch.stack(imgs).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        prob = F.softmax(model(x).float(), dim=1)
    out = []
    for p, pr in zip(ok_paths, prob.cpu()):
        vals, idx = pr.topk(min(topk, len(classes)))
        out.append((p, [(classes[i], float(v)) for v, i in zip(vals, idx)]))
    return out


def gather_images(spec):
    if os.path.isdir(spec):
        files = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            files += glob.glob(os.path.join(spec, "**", ext), recursive=True)
        return sorted(files)
    return [spec]


def run_images(model, tf, classes, args, device):
    files = gather_images(args.images)
    if not files:
        print("이미지를 못 찾았다: %s" % args.images)
        return 1
    print("%d개 이미지 추론\n" % len(files))
    for i in range(0, len(files), args.batch):
        chunk = files[i:i + args.batch]
        for path, preds in predict_batch(model, tf, chunk, device, classes, args.topk):
            top = preds[0]
            extra = ("  " + " ".join("%s:%.2f" % (n, p) for n, p in preds[1:])
                     if args.topk > 1 else "")
            print("  %-55s → %-11s %.3f%s"
                  % (os.path.basename(path), top[0], top[1], extra))
    return 0


def run_manifest(model, tf, classes, args, device):
    import pandas as pd

    df = pd.read_csv(args.manifest)
    root = os.path.dirname(os.path.abspath(args.manifest))
    if args.limit:
        df = df.sample(min(args.limit, len(df)), random_state=0).reset_index(drop=True)
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    correct = total = 0
    conf = {}
    paths = [os.path.join(root, p) for p in df["image_path"]]
    labels = list(df["label"])
    for i in range(0, len(paths), args.batch):
        res = predict_batch(model, tf, paths[i:i + args.batch], device, classes, 1)
        for (path, preds), lab in zip(res, labels[i:i + args.batch]):
            pred = preds[0][0]
            total += 1
            if pred == lab:
                correct += 1
            conf[(lab, pred)] = conf.get((lab, pred), 0) + 1
        sys.stdout.write("\r  %d/%d  acc %.3f" % (total, len(paths), correct / max(total, 1)))
        sys.stdout.flush()
    print("\n\n정확도 %.4f (%d/%d)" % (correct / max(total, 1), correct, total))
    print("\n틀린 것 상위:")
    wrong = sorted(((v, k) for k, v in conf.items() if k[0] != k[1]), reverse=True)
    for v, (lab, pred) in wrong[:12]:
        print("  %-11s → %-11s %d회" % (lab, pred, v))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="학습 모델 추론")
    ap.add_argument("--ckpt", required=True,
                    help="체크포인트. 'runs/<시각>/best.pt' 처럼 짧게 줘도 된다")
    ap.add_argument("--images", default=None, help="이미지 파일 또는 폴더")
    ap.add_argument("--manifest", nargs="?", const=os.environ.get(
        "TR_DATA", "/workspace/dataset") + "/manifest.csv", default=None,
        help="manifest.csv 로 정확도 재측정. 값 없이 --manifest 만 주면 "
             "기본 데이터셋(%s)을 쓴다" % (os.environ.get("TR_DATA", "/workspace/dataset")))
    ap.add_argument("--limit", type=int, default=None, help="매니페스트에서 N개만")
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tf, classes = load_model(args.ckpt, device)

    if args.manifest:
        return run_manifest(model, tf, classes, args, device)
    if args.images:
        return run_images(model, tf, classes, args, device)
    print("--images 또는 --manifest 중 하나를 줄 것.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
