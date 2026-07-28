#!/usr/bin/env python3
"""
신호등 인지 결과를 **import 해서** 받는 API (판단·제어용).

토픽 구독/프린트 대신, 판제 코드가 이 모듈을 import 해서 최신 신호 문자열을
바로 읽는다. 내부적으로 카메라 UDP 수신 + 추론 서버(serve.py) TCP 요청을
백그라운드 스레드로 돌리고, state() 는 항상 **가장 최근** 결과를 즉시 돌려준다
(호출이 블로킹되지 않으므로 제어 루프 주기에 영향 없음).

    from traffic_light import TrafficLightClient

    tl = TrafficLightClient(host="127.0.0.1", port=5555, udp_port=9090)
    tl.start()                       # 카메라·추론 백그라운드 시작
    ...
    while control_loop:
        light = tl.state()           # "green" / "red" / ... / "not_detected"
        if light == "red":
            brake()
        elif light == "not_detected":
            keep_previous()          # 지킬 신호 없음
    tl.stop()

state() 반환 문자열 8종+1:
    red yellow green red_yellow red_left green_left left  (신호색)
    not_detected                                          (기권/신호 없음)

이 파일은 표준 라이브러리만 쓴다 (torch 불필요). 추론은 serve.py(GPU)가 한다.
"""

import socket
import struct
import threading
import time

from camera_udp import CameraUDP
from live_infer import InferClient

NOT_DETECTED = "not_detected"


class TrafficLightClient(object):
    """
    카메라 UDP → 추론 서버 → 최신 신호 문자열. 백그라운드로 갱신된다.

    host/port   : 추론 서버(serve.py) 주소. 다른 PC 면 서버 IP.
    udp_ip/port : MORAI 카메라 UDP 수신 주소·포트 (Sensor Setting 의 Destination Port).
    hz          : 초당 추론 상한.
    stale_sec   : 이 시간 이상 새 결과가 없으면 state() 가 not_detected 를 준다
                  (카메라·서버가 끊겨도 옛 값을 붙들지 않도록).
    """

    def __init__(self, host="127.0.0.1", port=5555,
                 udp_ip="0.0.0.0", udp_port=9090, hz=10.0, stale_sec=1.0):
        self._host, self._port = host, port
        self._udp_ip, self._udp_port = udp_ip, udp_port
        self._period = 1.0 / max(hz, 0.5)
        self._stale = stale_sec

        self._lock = threading.Lock()
        self._label = NOT_DETECTED
        self._prob = 0.0
        self._last_ok = 0.0          # 마지막으로 결과를 받은 시각(perf_counter)
        self._udp = None
        self._client = None
        self._thread = None
        self._running = False

    # ---- 수명 주기 ----
    def start(self, wait_first=10.0):
        """카메라·추론 백그라운드 시작. 첫 프레임을 wait_first 초 기다린다(0=안 기다림)."""
        if self._running:
            return self
        self._udp = CameraUDP(ip=self._udp_ip, port=self._udp_port)
        if wait_first:
            self._udp.wait_first(timeout=wait_first)
        self._client = InferClient(self._host, self._port)
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="tl-infer")
        self._thread.daemon = True
        self._thread.start()
        return self

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._client is not None:
            self._client.close()
        if self._udp is not None:
            self._udp.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ---- 판제가 읽는 부분 ----
    def state(self):
        """최신 신호 문자열. 결과가 stale_sec 보다 오래됐으면 not_detected."""
        with self._lock:
            if time.perf_counter() - self._last_ok > self._stale:
                return NOT_DETECTED
            return self._label

    def detail(self):
        """(label, prob) — 확률까지 필요할 때."""
        with self._lock:
            if time.perf_counter() - self._last_ok > self._stale:
                return NOT_DETECTED, 0.0
            return self._label, self._prob

    # ---- 내부 루프 ----
    def _loop(self):
        last_seq = -1
        while self._running:
            loop = time.perf_counter()
            jpeg, seq = self._udp.latest()
            if jpeg is None or seq == last_seq:
                time.sleep(0.005)
                continue
            last_seq = seq
            try:
                r = self._client.predict(jpeg)
            except (ConnectionError, socket.timeout, struct.error):
                time.sleep(0.5)
                try:
                    self._client.connect()
                except Exception:
                    pass
                continue
            with self._lock:
                self._label = r["label"]
                self._prob = r["prob"]
                self._last_ok = time.perf_counter()
            sleep = self._period - (time.perf_counter() - loop)
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    # 데모: import 없이 이 파일만 실행하면 state() 를 주기적으로 찍는다.
    import argparse

    ap = argparse.ArgumentParser(description="TrafficLightClient 데모")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--udp-port", type=int, default=9090)
    a = ap.parse_args()

    tl = TrafficLightClient(host=a.host, port=a.port, udp_port=a.udp_port).start()
    print("state() 폴링 (Ctrl-C 종료)\n")
    try:
        while True:
            print("\r\033[K  신호 = %-14s" % tl.state().upper(), end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        tl.stop()
