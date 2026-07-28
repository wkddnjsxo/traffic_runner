#!/usr/bin/env python3
"""
추론 서버 — GPU 가 있는 프로세스에서 돈다.

카메라 클라이언트(live_infer.py)가 JPEG 를 TCP 로 보내면 신호 상태를 돌려준다.
GPU 전처리(리사이즈+정규화)까지 여기서 하므로 장당 ~3ms.

프로토콜 (양방향 길이 접두):
    요청 : [4바이트 빅엔디안 길이][JPEG 바이트]
    응답 : [4바이트 빅엔디안 길이][JSON]
    JSON : {"label": "red", "prob": 0.98,
            "top": [["red",0.98],["left",0.01]], "ms": 2.4}
    label : unknown 제외 최고 확률이 --min-detect 이하이면 "not_detected" (기권).

  python3 serve.py --ckpt best.pt
  python3 serve.py --ckpt best.pt --min-detect 0.4   # 기권 임계값
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

from model import IMAGENET_MEAN, IMAGENET_STD, decide, load_model   # noqa: E402


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
    ap.add_argument("--ckpt", required=True, help="best.pt 경로 (deploy/ 안이면 파일명만)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--min-detect", type=float, default=0.40,
                    help="unknown 제외 최고 확률이 이 값 이하이면 not_detected 로 기권")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tf, classes, size, val_acc = load_model(args.ckpt, device)

    # GPU 전처리 (PIL CPU 리사이즈가 전체의 90% 를 먹어서 GPU 로 옮김: 118ms→10ms)
    out_h, out_w = size, int(round(size * 4 / 3))
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def preprocess(img):
        x = torch.from_numpy(np.asarray(img, dtype=np.uint8)).to(device)
        x = x.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        x = F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)
        return (x - mean) / std

    print("=" * 62)
    print("모델   : %s | 클래스 %d개" % (os.path.basename(args.ckpt), len(classes)))
    print("해상도 : %d (입력 %dx%d)" % (size, size, out_w))
    print("장치   : %s" % (torch.cuda.get_device_name(0)
                           if device.type == "cuda" else "CPU"))
    print("전처리 : %s" % ("GPU" if device.type == "cuda" else "CPU"))
    print("val_acc: %.4f (저장 시점)" % val_acc)
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
                            probs = F.softmax(model(x).float(), dim=1)[0].cpu()
                    dt = (time.perf_counter() - t0) * 1000

                    label, prob = decide(probs, classes, args.min_detect)
                    vals, idx = probs.topk(min(args.topk, len(classes)))
                    resp = {
                        "label": label,          # 신호색 문자열 또는 "not_detected"
                        "prob": round(float(prob), 4),
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
