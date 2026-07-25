#!/usr/bin/env python3
"""
신호등 인지 ResNet18 학습.

  python3 train.py                          # 기본 설정
  python3 train.py --epochs 30 --batch 128
  python3 train.py --max-dist 40            # 신호등 40m 이내 프레임만
  python3 train.py --holdout-spots sig_006  # 그 지점을 통째로 val 로

데이터는 수집기가 만든 dataset/manifest.csv 를 그대로 읽는다.
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from dataset import (CLASS_NAMES, NUM_CLASSES, TrafficLightDataset,   # noqa: E402
                     class_counts, class_weights, load_manifest, split_by_drive)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(size, train_aug=True, aspect=4.0 / 3.0):
    """
    증강은 보수적으로 건다.

    ★ 종횡비 보존 ★
    원본은 1280x960 (4:3). 정사각(size,size)으로 리사이즈하면 가로를 1.33배
    찌그러뜨려 안 그래도 작은 신호등이 더 뭉개진다. 그래서 4:3 비율을 유지한
    (H=size, W=size*4/3) 로 리사이즈한다. ResNet 은 정사각이 아니어도 된다
    (adaptive average pool 이 들어 있다).

    ★ 좌우 반전 금지 ★
    RandomHorizontalFlip 을 쓰면 좌회전 화살표가 우회전처럼 보여
    red_left/green_left/left 라벨이 뒤집힌다.
    색조 변경(ColorJitter 의 hue)도 신호등 색 자체를 바꾸므로 금지. 밝기/대비만.
    """
    h = size
    w = int(round(size * aspect))
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if not train_aug:
        return transforms.Compose([
            transforms.Resize((h, w)),
            transforms.ToTensor(), norm])
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1, hue=0.0),
        transforms.ToTensor(),
        norm,
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.08)),
    ])


def build_model(pretrained=True):
    """
    ResNet18. pretrained=True 면 ImageNet 가중치를 받는다.

    가중치는 인터넷에서 받으므로, 오프라인 컨테이너에서 처음 실행하면 실패한다.
    실패 시 랜덤 초기화로 조용히 넘어가지 않고 명확히 알린다 — 랜덤 초기화면
    수렴이 훨씬 느리고 정확도도 낮아 "왜 안 되지" 로 시간을 버리기 때문이다.
    """
    weights = None
    if pretrained:
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        except Exception as exc:
            print("⚠ pretrained 가중치 로드 실패(%s). 랜덤 초기화로 진행한다." % exc)
    try:
        model = models.resnet18(weights=weights)
    except Exception as exc:
        # 다운로드 실패 (오프라인 등)
        print("⚠ ImageNet 가중치 다운로드 실패: %s" % exc)
        print("  → 랜덤 초기화로 진행. 컨테이너에 미리 받아두거나 --no-pretrained 를 쓸 것.")
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def evaluate(model, loader, device, criterion, want_preds=False):
    model.eval()
    loss_sum = n = correct = 0
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    all_pred = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(x)
                loss = criterion(out, y)
            pred = out.argmax(1)
            loss_sum += float(loss) * y.size(0)
            correct += int((pred == y).sum())
            n += y.size(0)
            for t, p in zip(y.cpu(), pred.cpu()):
                conf[t, p] += 1
            if want_preds:
                all_pred.append(pred.cpu())
    preds = torch.cat(all_pred).numpy() if want_preds and all_pred else None
    return loss_sum / max(n, 1), correct / max(n, 1), conf, preds


def report_by_weather_distance(val_df, preds, out_path=None):
    """
    날씨 × 거리 구간별 정확도, 그리고 날씨별 클래스 정확도.

    거리 구간만으로는 오해를 부른다 — 각 구간에 특정 지점/클래스가 몰려서,
    그 지점을 val 로 뽑았는지에 따라 100% 나 급락이 나온다. 그래서 (1) 거리 구간과
    (2) 날씨별 클래스 recall 을 함께 낸다. FOGGY 에서 특정 클래스가 낮으면
    그게 안개 때문에 못 맞히는 것이다.
    """
    df = val_df.copy()
    df["pred"] = preds
    df["correct"] = (df["pred"] == df["label_index"]).astype(int)
    df["dist"] = df["dist_to_tl_m"]

    lines = ["\n[리포트 1] 날씨 × 거리 구간별 정확도 (신호등 있는 프레임만)"]
    bins = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 999)]
    sig = df[df["dist"].notna()]
    lines.append("%-8s" % "weather"
                 + "".join("%13s" % ("%d-%dm" % (a, b if b < 999 else 99))
                           for a, b in bins))
    for w in sorted(sig["weather"].unique()):
        wd = sig[sig["weather"] == w]
        cells = []
        for a, b in bins:
            m = wd[(wd["dist"] >= a) & (wd["dist"] < b)]
            cells.append("%5.1f%%(%d)" % (100 * m["correct"].mean(), len(m))
                         if len(m) else "      -    ")
        lines.append("%-8s" % w + "".join("%13s" % c for c in cells))

    lines.append("\n[리포트 2] 날씨별 클래스 recall (안개가 어느 클래스를 무너뜨리나)")
    classes = sorted(df["label_index"].unique())
    from dataset import CLASS_NAMES as CN
    w_col = max(11, max(len(CN[c]) for c in classes) + 2)
    lines.append("%-8s" % "weather" + "".join(("%%%ds" % w_col) % CN[c] for c in classes))
    for w in sorted(df["weather"].unique()):
        wd = df[df["weather"] == w]
        cells = []
        for c in classes:
            m = wd[wd["label_index"] == c]
            cells.append("%5.1f%%" % (100 * m["correct"].mean()) if len(m) else "   -  ")
        lines.append("%-8s" % w + "".join(("%%%ds" % w_col) % c for c in cells))

    text = "\n".join(lines)
    print(text)
    if out_path:
        with open(out_path, "w") as f:
            f.write(text + "\n")
    return text


def print_confusion(conf):
    print("\n혼동행렬 (행=정답, 열=예측)")
    w = max(8, max(len(c) for c in CLASS_NAMES) + 2)
    fmt = "%%%ds" % w
    print("%-12s" % "" + "".join(fmt % c for c in CLASS_NAMES) + "%9s" % "recall")
    for i, name in enumerate(CLASS_NAMES):
        row = conf[i]
        tot = int(row.sum())
        rec = (float(row[i]) / tot * 100) if tot else 0.0
        print("%-12s" % name + "".join(fmt % int(v) for v in row)
              + ("%8.1f%%" % rec if tot else "       -"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="신호등 인지 ResNet18 학습")
    ap.add_argument("--data", default=os.environ.get("TR_DATA", "/workspace/dataset"),
                    help="dataset 폴더 (manifest.csv 가 있는 곳)")
    ap.add_argument("--out", default=os.environ.get("TR_OUT", "/workspace/train/runs"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-dist", type=float, default=None,
                    help="신호등까지 이 거리 이내 프레임만 사용(m)")
    ap.add_argument("--min-dist", type=float, default=None)
    ap.add_argument("--holdout-spots", default=None,
                    help="이 지점들을 통째로 val 로 (쉼표 구분)")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--compile", action="store_true", help="torch.compile 사용")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("torch %s | CUDA %s" % (torch.__version__, torch.version.cuda))
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability(0)
        print("GPU: %s (sm_%d%d, %.1fGB)"
              % (torch.cuda.get_device_name(0), cap[0], cap[1],
                 torch.cuda.get_device_properties(0).total_memory / 1e9))
        if cap[0] >= 12:
            print("  → Blackwell. cu128 이상 휠이 아니면 여기서 커널 에러가 난다.")
    else:
        print("⚠ GPU 를 못 찾았다. CPU 로 학습하면 매우 느리다.")
        print("  docker run 에 --gpus all 을 줬는지, cu128 휠인지 확인할 것.")
    print("=" * 70)

    # ---- 데이터 ----
    df = load_manifest(args.data, min_dist=args.min_dist, max_dist=args.max_dist)
    if df.empty:
        print("필터 후 데이터가 없다.")
        return 1
    holdout = ([s.strip() for s in args.holdout_spots.split(",")]
               if args.holdout_spots else None)
    train_df, val_df = split_by_drive(df, args.val_ratio, args.seed, holdout)

    print("\n클래스 분포 (train / val)")
    tc, vc = class_counts(train_df), class_counts(val_df)
    for i, name in enumerate(CLASS_NAMES):
        flag = "  ← val 에 없음" if vc[i] == 0 and tc[i] > 0 else ""
        print("  %-12s %7d / %6d%s" % (name, tc[i], vc[i], flag))

    train_ds = TrafficLightDataset(train_df, args.data, build_transforms(args.size, True))
    val_ds = TrafficLightDataset(val_df, args.data, build_transforms(args.size, False))
    pin = device.type == "cuda"
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=pin, drop_last=True,
                          persistent_workers=args.workers > 0)
    val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=pin,
                        persistent_workers=args.workers > 0)

    # ---- 모델 ----
    model = build_model(not args.no_pretrained).to(device)
    if args.compile:
        model = torch.compile(model)

    w = None if args.no_class_weights else class_weights(train_df).to(device)
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    # max_lr = lr * 3. pretrained 백본에는 *10 이 너무 커서 초반에 발산할 수 있다.
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr * 3, total_steps=args.epochs * max(len(train_ld), 1),
        pct_start=0.1)

    os.makedirs(args.out, exist_ok=True)
    run_dir = os.path.join(args.out, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    best = 0.0
    print("\n학습 시작 — %d epoch, batch %d, %d steps/epoch"
          % (args.epochs, args.batch, len(train_ld)))
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum = n = correct = 0
        for bi, (x, y) in enumerate(train_ld, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(x)
                loss = criterion(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

            loss_sum += float(loss) * y.size(0)
            correct += int((out.argmax(1) == y).sum())
            n += y.size(0)
            if bi % 20 == 0 or bi == len(train_ld):
                sys.stdout.write("\r\033[K  ep%02d %4d/%d  loss %.4f  acc %.3f"
                                 % (ep, bi, len(train_ld), loss_sum / n, correct / n))
                sys.stdout.flush()

        vl, va, conf, _ = evaluate(model, val_ld, device, criterion)
        print("\r\033[K  ep%02d  train loss %.4f acc %.3f | val loss %.4f acc %.3f  (%.0fs)"
              % (ep, loss_sum / max(n, 1), correct / max(n, 1), vl, va, time.time() - t0))

        torch.save({"model": model.state_dict(), "classes": CLASS_NAMES,
                    "args": vars(args), "val_acc": va},
                   os.path.join(run_dir, "last.pt"))
        if va > best:
            best = va
            torch.save({"model": model.state_dict(), "classes": CLASS_NAMES,
                        "args": vars(args), "val_acc": va},
                       os.path.join(run_dir, "best.pt"))

    # ★ 리포트는 best.pt(최고 val) 기준으로 낸다 ★
    # 마지막 epoch 모델로 뽑으면, 저장된 best 와 다른 모델의 성적을 보게 되어
    # "best 는 99.9% 인데 혼동행렬은 86%" 같은 엇갈림이 생긴다.
    best_path = os.path.join(run_dir, "best.pt")
    if os.path.exists(best_path):
        ck = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        print("\n[리포트] best.pt (val %.4f) 기준으로 재평가" % ck.get("val_acc", -1))
    _, final_acc, conf, preds = evaluate(model, val_ld, device, criterion,
                                         want_preds=True)
    print_confusion(conf)
    report_by_weather_distance(val_df, preds,
                               os.path.join(run_dir, "report_weather_dist.txt"))

    print("\n최고 val 정확도 %.4f (위 리포트가 이 모델 기준) | 저장: %s"
          % (best, run_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
