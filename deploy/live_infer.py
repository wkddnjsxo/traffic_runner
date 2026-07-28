#!/usr/bin/env python3
"""
실시간 신호등 인지 (대회용) — 카메라 UDP 를 받아 추론 서버로 넘기고 결과를 출력한다.

카메라(MORAI UDP) → 이 스크립트 → TCP → serve.py(GPU 추론) → 예측.
이 파일은 **표준 라이브러리만** 쓴다 (torch 불필요). 추론은 서버가 한다.

  # 다른 터미널에서 서버 먼저:  python3 serve.py --ckpt best.pt
  python3 live_infer.py                       # UDP 9090, 예측 출력
  python3 live_infer.py --udp-port 9090        # 포트 명시
  python3 live_infer.py --min-prob 0.7 --topk  # 애매한 예측 표시

MORAI Sensor Setting 의 Destination Port 를 --udp-port 와 맞출 것.
"""

import argparse
import json
import os
import socket
import struct
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from camera_udp import CameraUDP   # noqa: E402


class InferClient(object):
    """추론 서버에 JPEG 를 보내고 결과 JSON 을 받는다."""

    def __init__(self, host, port, timeout=5.0):
        self.addr = (host, port)
        self.timeout = timeout
        self.sock = None
        self.connect()

    def connect(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect(self.addr)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock = s
        print("[client] 추론 서버 연결 %s:%d" % self.addr)

    def predict(self, jpeg):
        self.sock.sendall(struct.pack(">I", len(jpeg)) + jpeg)
        head = self._recv(4)
        (n,) = struct.unpack(">I", head)
        return json.loads(self._recv(n).decode())

    def _recv(self, n):
        buf = b""
        while len(buf) < n:
            c = self.sock.recv(n - len(buf))
            if not c:
                raise ConnectionError("서버 연결 끊김")
            buf += c
        return buf

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="실시간 신호등 인지 (대회용, UDP)",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--udp-ip", default="0.0.0.0", help="카메라 UDP 수신 IP")
    ap.add_argument("--udp-port", type=int, default=9090,
                    help="카메라 UDP 포트 (MORAI Sensor Setting 의 Destination Port)")
    ap.add_argument("--host", default="127.0.0.1", help="추론 서버 주소")
    ap.add_argument("--port", type=int, default=5555, help="추론 서버 포트")
    ap.add_argument("--hz", type=float, default=10.0, help="초당 추론 횟수 상한")
    ap.add_argument("--min-prob", type=float, default=0.0,
                    help="확률이 이보다 낮으면 UNCERTAIN 으로 표시 (0=끄기)")
    ap.add_argument("--topk", action="store_true", help="상위 후보도 표시")
    args = ap.parse_args(argv)

    # ---- 카메라 UDP ----
    udp = CameraUDP(ip=args.udp_ip, port=args.udp_port)
    print("[udp] %s:%d 수신 대기..." % (args.udp_ip, args.udp_port))
    first = udp.wait_first(timeout=10.0)
    if first is None:
        print("UDP 프레임을 못 받았다. MORAI 카메라 UDP 설정과 포트(%d)를 확인할 것."
              % args.udp_port)
        return 1
    print("[udp] 수신 시작 (%d bytes)" % first)

    client = InferClient(args.host, args.port)

    period = 1.0 / max(args.hz, 0.5)
    n = 0
    last_seq = -1
    pred_counts = {}
    print("\nCtrl-C 로 종료\n")
    try:
        while True:
            loop = time.time()
            jpeg, seq = udp.latest()
            if jpeg is None or seq == last_seq:
                time.sleep(0.005)
                continue
            last_seq = seq

            try:
                r = client.predict(jpeg)
            except (ConnectionError, socket.timeout) as exc:
                print("\n서버 연결 문제: %s — 재연결" % exc)
                time.sleep(1.0)
                try:
                    client.connect()
                except Exception:
                    pass
                continue
            n += 1
            pred_counts[r["label"]] = pred_counts.get(r["label"], 0) + 1

            label = r["label"].upper()
            if args.min_prob and r["prob"] < args.min_prob:
                label = "UNCERTAIN(%s)" % r["label"]
            line = "  ▶ %-16s  %3.0f%%   (%.1fms)" % (label, r["prob"] * 100, r["ms"])
            if args.topk:
                line += "   [" + " ".join("%s %.0f%%" % (a, b * 100)
                                          for a, b in r["top"][1:]) + "]"
            sys.stdout.write("\r\033[K" + line)
            sys.stdout.flush()

            sleep = period - (time.time() - loop)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\n\n=== 요약 ===")
        print("추론 %d회" % n)
        if pred_counts:
            print("예측 분포:")
            for lab, c in sorted(pred_counts.items(), key=lambda x: -x[1]):
                print("  %-12s %6d회 (%.1f%%)" % (lab, c, 100.0 * c / n))
    finally:
        client.close()
        udp.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
