# 신호등 인지 — 추론 배포판 (deploy)

MORAI 카메라(UDP)로 실시간 신호등 상태를 인지한다. **이 폴더만 있으면 추론이 된다**
(학습·수집 코드 불필요).

출력 8종: `red` `yellow` `green` `red_yellow` `red_left` `green_left` `left` `unknown`

## 구성

| 파일 | 역할 | 의존성 |
|---|---|---|
| `serve.py` | 추론 서버 (GPU) | torch, torchvision, pillow, numpy |
| `model.py` | ResNet18 정의 + 체크포인트 로드 | torch |
| `camera_udp.py` | MORAI 카메라 UDP 파서 | **표준 라이브러리만** |
| `live_infer.py` | 카메라 수신 → 예측을 터미널에 출력 (데모/확인용) | **표준 라이브러리만** |
| `traffic_light.py` | **판제가 import 해서 최신 신호를 받는 API** | **표준 라이브러리만** |
| `best.pt` | 학습된 모델 (size 384, val 99.8%) | — |

구조: `카메라(UDP) → live_infer.py / traffic_light.py → TCP → serve.py(GPU 추론) → 예측`.
추론은 서버가 하므로 카메라 쪽은 torch 가 필요 없다.
- 터미널로 눈으로 볼 때는 `live_infer.py`,
- 판단·제어 코드에서 값으로 받을 때는 `traffic_light.py` 를 import.

## 설치 (GPU 있는 PC)

```bash
# GPU 에 맞는 torch 설치
#   RTX 40 (Ada):      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#   RTX 50 (Blackwell): pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install --user pillow numpy

# GPU 확인 (capability 나오고 에러 없으면 OK)
python3 -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

## 실행 — 터미널 2개

**1) 추론 서버** (GPU)
```bash
python3 serve.py --ckpt best.pt                 # 기권 임계값 기본 0.40
python3 serve.py --ckpt best.pt --min-detect 0.5 # 더 보수적으로
```
```
장치 : NVIDIA GeForce RTX 4060 Ti | 대기 중: 0.0.0.0:5555
```

**2) 카메라 → 추론**
```bash
python3 live_infer.py                    # UDP 9090, 예측 출력
python3 live_infer.py --udp-port 9090     # 포트 명시
python3 live_infer.py --min-prob 0.7 --topk   # 애매한 예측 표시
```
```
  ▶ GREEN_LEFT      94%   (3.1ms)
  ▶ NOT_DETECTED    31%   (3.0ms)
```

## 기권(not_detected)

`unknown` 을 제외한 신호 클래스 중 최고 확률이 **`--min-detect`(기본 0.40) 이하**이면
색을 억지로 찍지 않고 `not_detected` 로 기권한다. 터미널에도 `NOT_DETECTED` 로 나온다.

출력 문자열은 **신호색 8종 중 하나 또는 `not_detected`** — 전부 문자열이다.

| label | 뜻 |
|---|---|
| `red` `yellow` `green` `red_yellow` `red_left` `green_left` `left` | 인식된 신호색 |
| `not_detected` | 어떤 신호색도 확신 못 함(≤ min-detect) → 기권. 지킬 신호 없음 |

```
서버 응답 JSON: {"label":"not_detected", "prob":0.31,
                 "top":[["green",0.31],["unknown",0.28],...], "ms":3.0}
```

임계값을 더 보수적으로 하려면 서버에 `--min-detect 0.5` 처럼 준다.

## 판단·제어에서 import 로 받기

토픽 구독이나 프린트 파싱 없이, 판제 코드가 `traffic_light.py` 를 import 해서
최신 신호 문자열을 바로 읽는다. 카메라 수신·추론은 백그라운드에서 돌고,
`state()` 는 항상 **가장 최근** 값을 즉시 돌려준다(제어 루프를 막지 않음).

```python
from traffic_light import TrafficLightClient

tl = TrafficLightClient(host="127.0.0.1", port=5555, udp_port=9090)
tl.start()                       # 카메라·추론 백그라운드 시작

while control_loop:
    light = tl.state()           # "green" / "red" / ... / "not_detected"
    if light == "red":
        brake()
    elif light == "not_detected":
        keep_previous()          # 지킬 신호 없음
tl.stop()
```

- `state()` → 신호 문자열 하나. `detail()` → `(label, prob)`.
- 서버(`serve.py`)와 다른 PC 면 `host` 를 서버 IP 로.
- 카메라·서버가 끊겨 `stale_sec`(기본 1s) 이상 새 값이 없으면 자동으로 `not_detected`
  를 준다(옛 신호를 붙들지 않음).
- `python3 traffic_light.py` 로 실행하면 `state()` 를 주기적으로 찍는 데모가 돈다.

Ctrl-C 로 끝내면 예측 분포 요약이 나온다.

## MORAI 카메라 설정 (Sensor Setting)

| 항목 | 값 | 비고 |
|---|---|---|
| Sensor Network | **UDP** | |
| Destination IP | 추론 PC 의 IP | 카메라가 여기로 전송 |
| Destination Port | **9090** | `--udp-port` 와 일치 |
| Horizontal FOV | **65** | 학습과 동일해야 함 |
| Rotation Pitch | **340** (위로 20°) | 학습과 동일해야 함 |

> 해상도(Width×Height)는 서버가 384 로 리사이즈하므로 640×480/1280×960 어느 쪽이든 동작.
> 단 **FOV 65 와 Pitch 340 은 학습과 같아야** 신호등의 화면상 위치·크기가 맞다.

## 옵션

| `live_infer.py` 옵션 | 뜻 |
|---|---|
| `--udp-port N` | 카메라 UDP 포트 (기본 9090) |
| `--host / --port` | 추론 서버 주소 (기본 127.0.0.1:5555). 서버·카메라가 다른 PC 면 서버 IP |
| `--min-prob 0.7` | 확률 낮으면 `UNCERTAIN` 표시 |
| `--topk` | 2·3순위 후보도 표시 |
| `--hz N` | 초당 추론 횟수 상한 |

## 다른 모델로 교체

`best.pt` 를 새 체크포인트로 바꾸거나 `--ckpt <경로>` 로 지정하면 된다.
체크포인트에 학습 해상도(size)가 들어있어 서버가 자동으로 맞춘다.
