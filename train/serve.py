#!/usr/bin/env python3
"""
추론 서버 — 컨테이너 안에서 돈다.

JPEG 바이트를 받아 신호 상태를 돌려준다. 카메라(ROS)는 WSL 에 있고 모델은
컨테이너에 있어서(WSL 은 Python 3.8 이라 cu128 torch 를 못 깐다) TCP 로 잇는다.

프로토콜 (양방향 모두 길이 접두):
    요청 : [4바이트 빅엔디안 길이][JPEG 바이트]
    응답 : [4바이트 빅엔디안 길이][JSON]
    JSON : {"label": "red", "index": 0, "prob": 0.98,
            "top": [["red",0.98],["left",0.01]], "ms": 2.4}

  python3 serve.py --ckpt runs/<시각>/best.pt --port 5555
"""

import argparse
import io
import json
import os
import socket
import struct
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from infer import resolve_ckpt                    # noqa: E402
from train import (IMAGENET_MEAN, IMAGENET_STD,   # noqa: E402
                   build_model, build_transforms)


def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def main(argv=None):
    ap = argparse.ArgumentParser(description="신호등 인지 추론 서버")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = resolve_ckpt(args.ckpt)
    ck = torch.load(path, map_location=device, weights_only=False)
    classes = ck["classes"]
    size = int(ck.get("args", {}).get("size", 224))
    model = build_model(pretrained=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    tf = build_transforms(size, train_aug=False)   # CPU 경로(폴백)용

    # ★ 전처리를 GPU 에서 한다 ★
    # PIL 의 CPU 리사이즈(1280x960 -> 384x512)가 전체 시간의 90% 를 먹는다
    # (실측: CPU 경로 118ms vs GPU 경로 9.9ms — 12배). 결과는 동일하다
    # (예측 확률 차이 0.0005).
    out_h, out_w = size, int(round(size * 4 / 3))
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def preprocess(img):
        x = torch.from_numpy(np.asarray(img, dtype=np.uint8)).to(device)
        x = x.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        x = F.interpolate(x, size=(out_h, out_w), mode="bilinear",
                          align_corners=False)
        return (x - mean) / std

    print("=" * 62)
    print("모델   : %s" % os.path.basename(path))
    print("해상도 : %d (입력 %dx%d)" % (size, size, int(size * 4 / 3)))
    print("장치   : %s" % (torch.cuda.get_device_name(0)
                           if device.type == "cuda" else "CPU"))
    print("전처리 : %s" % ("GPU (리사이즈+정규화)" if device.type == "cuda" else "CPU"))
    print("val_acc: %.4f (저장 시점)" % ck.get("val_acc", -1))
    print("=" * 62)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(4)
    print("대기 중: %s:%d  (Ctrl-C 로 종료)\n" % (args.host, args.port))

    n_total = 0
    try:
        while True:
            conn, addr = srv.accept()
            print("[연결] %s:%d" % addr)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            n_conn = 0
            try:
                while True:
                    head = recv_exact(conn, 4)
                    if head is None:
                        break
                    (length,) = struct.unpack(">I", head)
                    if length == 0 or length > 50 * 1024 * 1024:
                        break
                    data = recv_exact(conn, length)
                    if data is None:
                        break

                    t0 = time.perf_counter()
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                    if device.type == "cuda":
                        x = preprocess(img)
                    else:
                        x = tf(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        with torch.autocast("cuda", dtype=torch.bfloat16,
                                            enabled=device.type == "cuda"):
                            prob = F.softmax(model(x).float(), dim=1)[0].cpu()
                    dt = (time.perf_counter() - t0) * 1000

                    vals, idx = prob.topk(min(args.topk, len(classes)))
                    resp = {
                        "label": classes[int(idx[0])],
                        "index": int(idx[0]),
                        "prob": float(vals[0]),
                        "top": [[classes[int(i)], round(float(v), 4)]
                                for v, i in zip(vals, idx)],
                        "ms": round(dt, 2),
                    }
                    payload = json.dumps(resp).encode()
                    conn.sendall(struct.pack(">I", len(payload)) + payload)
                    n_conn += 1
                    n_total += 1
                    if n_conn % 50 == 0:
                        sys.stdout.write("\r\033[K  처리 %d장 (마지막 %s %.2f, %.1fms)"
                                         % (n_conn, resp["label"], resp["prob"],
                                            resp["ms"]))
                        sys.stdout.flush()
            except ConnectionResetError:
                pass
            finally:
                conn.close()
                print("\r\033[K[종료] %s:%d — %d장 처리" % (addr[0], addr[1], n_conn))
    except KeyboardInterrupt:
        print("\n서버 종료. 총 %d장 처리." % n_total)
    finally:
        srv.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
