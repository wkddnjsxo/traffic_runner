"""
MORAI 카메라 UDP 수신 → JPEG 프레임 조립.

대회 환경은 카메라를 ROS 가 아니라 UDP 로 보낸다. MORAI 공식 파서
(MORAI-NetworkModule/lib/define/Camera.py, network/UDP.py)의 프로토콜을 그대로
따르되, 바운딩박스는 빼고 이미지만 조립한다.

프로토콜 (패킷 65000바이트 고정, little-endian, 패딩 없음):
    header : char[3]   'MOR'(이미지 청크) 또는 'BOX'(바운딩박스, 무시)
    ---- header == 'MOR' 이면 이어지는 64997바이트가 IMAGE ----
    sec       : int32
    nsec      : int32
    index     : int32          # 청크 순번 (0,1,2... 프레임 안에서). 프레임번호 아님.
    size      : int32          # 이 청크의 유효 JPEG 바이트 수 (<= 64979)
    jpeg_data : byte[64979]     # 앞 size 바이트만 유효
    tail      : char[2]         # 'AI'(더 있음) / 'EI'(End Image, 끝)

한 프레임의 JPEG 가 64979 보다 크면 여러 'MOR' 청크로 쪼개져 온다. size 만큼씩
이어붙이다가 tail=='EI' 를 만나면 한 장 완성.

★ 실측 확인 (K-city, 1280x960) ★
  index 0 size 64979 tail 'AI' → index 1 ... 'AI' → index 2 size 53534 'EI'
  한 프레임 ≈ 183KB, 3청크. tail 은 'AI'/'EI' 다 (공식 예제의 'MI' 가 아님).

공식 코드가 multiprocessing.Process + ctypes.memmove 로 무겁게 짠 걸,
struct 로 가볍게 대체했다 (추론 파이프라인엔 이미지만 필요).
"""

import socket
import struct
import threading


PACKET_SIZE = 65000
HEADER_SIZE = 3
IMG_HEADER_FMT = "<4i"          # sec, nsec, index, size
IMG_HEADER_SIZE = struct.calcsize(IMG_HEADER_FMT)   # 16
JPEG_MAX = 64979
TAIL_SIZE = 2


class CameraUDP(object):
    """
    카메라 UDP 를 백그라운드에서 받아 최신 완성 프레임을 들고 있는다.

    latest() 로 (jpeg_bytes, seq) 를 가져간다. ROS 의 CameraRecorder 와 같은
    인터페이스라, live_infer 에서 소스만 바꿔 끼우면 된다.
    """

    def __init__(self, ip="0.0.0.0", port=1111, rcvbuf=1 << 20):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
        self.sock.bind((ip, port))

        self._lock = threading.Lock()
        self._jpeg = None            # 최신 완성 프레임
        self._seq = 0
        self._buffer = b""           # 조립 중인 프레임
        self._running = True
        self.frames = 0
        self.addr = (ip, port)

        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def _recv_loop(self):
        while self._running:
            try:
                raw, _ = self.sock.recvfrom(PACKET_SIZE)
            except OSError:
                break
            if len(raw) < HEADER_SIZE:
                continue
            header = raw[:HEADER_SIZE]
            if header != b"MOR":         # BOX 등은 무시
                continue

            body = raw[HEADER_SIZE:]
            if len(body) < IMG_HEADER_SIZE:
                continue
            sec, nsec, index, size = struct.unpack_from(IMG_HEADER_FMT, body, 0)
            if size < 0 or size > JPEG_MAX:
                continue
            jpeg_start = IMG_HEADER_SIZE
            chunk = body[jpeg_start:jpeg_start + size]
            # tail 은 jpeg_data(64979) 영역 뒤에 온다
            tail = body[jpeg_start + JPEG_MAX: jpeg_start + JPEG_MAX + TAIL_SIZE]

            # 프레임 경계: index(청크 순번)가 0 이면 새 프레임 시작.
            # 앞 프레임이 EI 를 못 받고 끊겼어도 index 0 을 만나면 버리고 새로 시작한다.
            if index == 0:
                self._buffer = b""

            self._buffer += chunk
            if tail == b"EI":            # 프레임 완성
                jpeg = self._buffer
                self._buffer = b""
                # JPEG 유효성 최소 확인 (SOI 0xFFD8 ~ EOI 0xFFD9)
                if jpeg[:2] == b"\xff\xd8" and jpeg[-2:] == b"\xff\xd9":
                    with self._lock:
                        self._jpeg = jpeg
                        self._seq += 1
                        self.frames += 1
            elif tail != b"AI":
                # 'AI'/'EI' 외의 tail → 프레임 경계 유실. 버퍼 초기화로 복구.
                self._buffer = b""

    def latest(self):
        """(jpeg_bytes, seq) 또는 (None, 0)."""
        with self._lock:
            return self._jpeg, self._seq

    def wait_first(self, timeout=10.0):
        import time
        t0 = time.time()
        while time.time() - t0 < timeout:
            j, _ = self.latest()
            if j is not None:
                return len(j)
            time.sleep(0.02)
        return None

    def close(self):
        self._running = False
        try:
            self.sock.close()
        except Exception:
            pass
