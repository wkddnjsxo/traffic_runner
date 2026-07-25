#!/usr/bin/env python3
"""
해상도별 추론 속도 측정.

학습 해상도를 올리면 추론도 그만큼 느려진다. 실주행에 쓸 수 있는지는
카메라 주기(10Hz = 100ms)와 비교해야 판단할 수 있으므로 실제로 재본다.

  python3 bench_infer.py                      # 224/320/384 비교
  python3 bench_infer.py --sizes 384 --batch 1
"""

import argparse
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from train import build_model   # noqa: E402


def bench(model, x, iters, device):
    # 워밍업 (CUDA 커널 컴파일/캐시)
    with torch.no_grad():
        for _ in range(10):
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main(argv=None):
    ap = argparse.ArgumentParser(description="해상도별 추론 속도")
    ap.add_argument("--sizes", default="224,320,384")
    ap.add_argument("--batch", type=int, default=1,
                    help="실주행은 한 장씩 처리하므로 1 이 현실적")
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("장치: %s" % (torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"))
    print("배치 %d, 반복 %d회\n" % (args.batch, args.iters))

    model = build_model(pretrained=False).to(device).eval()

    print("%-8s %-14s %10s %10s %12s" % ("size", "입력(H×W)", "1장(ms)", "FPS", "10Hz 여유"))
    print("-" * 60)
    for s in [int(x) for x in args.sizes.split(",")]:
        h, w = s, int(round(s * 4 / 3))
        x = torch.randn(args.batch, 3, h, w, device=device)
        dt = bench(model, x, args.iters, device)
        per_img = dt / args.batch * 1000
        fps = 1.0 / (dt / args.batch)
        margin = "%.0f배 여유" % (100.0 / per_img) if per_img < 100 else "✘ 못 따라감"
        print("%-8d %-14s %9.2f %10.0f %12s"
              % (s, "%d×%d" % (h, w), per_img, fps, margin))

    if device.type == "cuda":
        print("\nGPU 메모리: %.2fGB 사용 / %.1fGB"
              % (torch.cuda.max_memory_allocated() / 1e9,
                 torch.cuda.get_device_properties(0).total_memory / 1e9))
    return 0


if __name__ == "__main__":
    sys.exit(main())
