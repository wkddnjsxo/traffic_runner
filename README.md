# traffic_runner

MORAI 시뮬레이터에서 **신호등 인지 AI(ResNet18) 학습 데이터셋을 자동 수집**하고,
학습·실시간 추론까지 하는 파이프라인.

출력 클래스 8종: `red` `yellow` `green` `red_yellow` `red_left` `green_left` `left` `unknown`

- **수집**: gRPC 로 신호등 연출 + 날씨/시간 + 객체 배치 → pure pursuit 주행 → 카메라 프레임 저장
- **학습**: Docker(cu128) 컨테이너에서 ResNet18 학습 (RTX 5070/4060 Blackwell·Ada)
- **추론**: 카메라(ROS 또는 UDP) → TCP → 컨테이너 추론 서버 → 실시간 신호 인지

---

## 0. 환경 / 전제

| 항목 | 값 |
|---|---|
| MORAI 맵 이름 | **`R_KR_PR_K-city_2025`** (infor 폴더명은 PG 지만 서버 맵명은 **PR**) |
| MORAI ego 차량 | `2023_Hyundai_Ioniq5` |
| 실행 환경 | MORAI=Windows, 이 코드=WSL Ubuntu (gRPC/UDP 로 통신) |
| Python (수집·추론 클라) | 3.8 (WSL) — ROS noetic |
| Python (학습·추론 서버) | Docker 컨테이너, cu128 torch |
| gRPC host | WSL 에서 본 Windows IP. `ip route \| grep default` 로 확인 |

`config/` 에서 `grpc.host`, `morai.map_name` 을 본인 환경에 맞게 확인:
[src/traffic_runner/config/runtime.yaml](src/traffic_runner/config/runtime.yaml)

---

## 1. 카메라 센서 설정 (현재값)

MORAI **Sensor Setting** 에서 아래대로 맞춘다.

| 항목 | 값 |
|---|---|
| Camera | Camera-1 / Ego-0 |
| Position (X, Y, Z) | 0.00, 0.00, **1.40** |
| Rotation (Roll, Pitch, Yaw) | 0, **340** (= 위로 20°), 0 |
| Frame Rate | 10 Hz |
| Compression Ratio | 90 |
| **Width × Height** | **640 × 480** |
| **Horizontal FOV** | **65** |
| Sensor Network | **UDP** |
| Host Sensor IP | 192.168.80.1 (Windows) |
| Destination IP | 192.168.88.11 (WSL — 추론 돌리는 PC) |
| **Destination Port** | **9090** |

> **해상도**: 학습 데이터는 1280×960 으로 수집했지만, 추론 파이프라인이 입력을
> 384×512 로 리사이즈하므로 **위 640×480 설정에서도 실시간 추론이 정상 동작함(검증됨)**.
> 서버가 어떤 해상도로 들어와도 384 로 맞춰 추론하기 때문. HFOV 65 와
> Pitch 340(위로 20°)은 학습과 동일해야 하므로 그대로 둘 것 — 이 둘이 바뀌면
> 신호등의 화면상 위치·크기가 달라져 성능에 직접 영향.
> Destination IP/Port 는 추론 PC 환경에 맞게, 코드에선 `--udp-port` 로 지정.

---

## 2. 데이터 수집

### 준비
1. MORAI 실행 → 맵 `R_KR_PR_K-city_2025` 로드 → **gRPC 서버 ON**
2. `config/runtime.yaml` 의 `grpc.host` 확인

### 지점(spot) 만들기 — 3가지 방법

**(A) 좌표 직접 지정** — 시작/끝점을 도로망으로 이어줌 (권장)
```bash
cd src/traffic_runner
python3 tools/where.py                     # 지금 ego 위치·링크·신호등 확인
python3 tools/make_spot.py --start X Y --end X Y --kind signal --dry-run
python3 tools/make_spot.py --start X Y --end X Y --kind signal --id sig_010
```

**(B) MGeo 자동 생성** — 신호등 접근로를 통째로
```bash
python3 tools/gen_spots_from_mgeo.py --unknown 8
```

**(C) 텔레옵 캡처** — 손으로 몰며 궤적 기록
```bash
../../capture.sh
```

### 지점 검수 → 수집
```bash
python3 tools/spot_report.py           # 지점별 상태·클래스 커버리지
python3 tools/check_visibility.py      # 신호등이 화각 안에 잡히는지 (오염 예방)

python3 tools/collect.py --plan        # 수집 계획 미리보기 (시뮬 불필요)
python3 tools/collect.py               # 전체 수집 (Ctrl-C 로 중단해도 이어서 재개)
python3 tools/collect.py --spots sig_010 --seeds 0   # 일부만
```

수집 루프: **환경(날씨×시간) → 지점 → 객체seed → 신호상태(랜덤순서)**.
객체가 신호상태보다 바깥이라 "객체는 신호와 독립". 신호 연출 후 `ActorState.tl_color`
로 되읽어 검증하고, 라벨 불일치 주행은 자동으로 버린다.

### 산출물
```
dataset/
├── manifest.csv     # image_path,label,label_index,spot_id,weather,hour,object_seed,
│                    #   state,frame_idx,dist_to_tl_m,tl_color_observed, ... (21컬럼)
├── progress.txt     # 완료 조합 (재개용)
└── images/<weather>_<hour>/<spot>/seed<NN>/<state>/000000.jpg
```
경로가 `dataset/` 기준 상대경로라 폴더째 마운트해도 동작.

---

## 3. 학습 (Docker, RTX 50/40 시리즈)

> RTX 5070(Blackwell sm_120)은 **cu128 이상 torch 필수**. 일반 휠은 커널 에러가 난다.
> Dockerfile 이 cu128 로 빌드하고 ImageNet 가중치도 미리 받아둔다.

```bash
cd train
./run.sh build            # 이미지 빌드 (torch 다운로드, 최초 1회 ~10분)
./run.sh check            # ★ GPU 확인 먼저 — sm_120/sm_89 + matmul 커널 정상 확인
./run.sh checkdata        # 데이터셋 무결성 (라벨·이미지·경로·오염 검사)

# 최종 학습 (대회용, 전체 데이터)
./run.sh train --epochs 3 --size 384 --batch 48 --lr 2e-4
```

`--size 384` 가 핵심 — 좌회전 화살표(green_left)가 224 에선 뭉개진다(76%→100%).
val 은 2~3 epoch 에 수렴하므로 epoch 를 늘려도 과적합만 된다.

### 검증 옵션
```bash
# 지점 홀드아웃: 그 지점을 통째로 val 로 → "처음 보는 교차로" 성능
./run.sh train --epochs 3 --size 384 --batch 48 --lr 2e-4 --holdout-spots sig_005

# 거리 필터: 신호등 N m 이내만 학습
./run.sh train --max-dist 40

# 학습 후 오류 분석 (지점×거리×클래스별)
./run.sh analyze --ckpt runs/<시각>/best.pt --holdout-spots sig_005
```

학습 결과는 `train/runs/<시각>/best.pt` (44MB). 리포트는 best 기준으로 나온다.

---

## 4. 실시간 추론

카메라(WSL)와 모델(컨테이너)이 분리돼 있어 **TCP 로 잇는다**.
컨테이너에서 서버를 띄우고, WSL 에서 카메라를 받아 넘긴다.

### 4-1. 추론 서버 (컨테이너) — 항상 먼저
```bash
cd train
./run.sh serve --ckpt runs/20260725_061944/best.pt
```
GPU 전처리(리사이즈+정규화)까지 서버가 하므로 장당 ~3ms. 10Hz 에 여유 충분.

### 4-2. 카메라 → 추론 (WSL)

**개발용 (ROS 토픽)** — K-city 개발 중, 시뮬 정답과 비교
```bash
source /opt/ros/noetic/setup.bash
cd src/traffic_runner
python3 tools/live_infer.py                 # 예측 + gRPC 정답 비교
python3 tools/live_infer.py --cycle green,green_left --spot sig_006   # 특정 쌍 혼동 측정
```

**대회용 (UDP)** — 신호 토픽 없는 맵, 예측만 출력
```bash
cd src/traffic_runner
python3 tools/live_infer.py --compete                   # UDP 9090, 예측만 크게 출력
python3 tools/live_infer.py --compete --udp-port 9090   # 포트 명시
python3 tools/live_infer.py --compete --min-prob 0.7 --topk   # 애매한 예측 표시
```

`--compete` = UDP 소스 + gRPC 없음 + 대회용 큰 출력. 출력 예:
```
  ▶ GREEN_LEFT      94%   (3.1ms)
```

MORAI 카메라 UDP 프로토콜(`MOR` 헤더, 청크 tail `AI`/`EI`, 65000바이트 패킷)은
[sim/camera_udp.py](src/traffic_runner/sim/camera_udp.py) 가 파싱한다.
UDP 설정은 MORAI **Sensor Setting** 의 Destination IP/Port 와 맞출 것(위 1절).

---

## 5. 도구 목록

### 수집 (`src/traffic_runner/tools/`)
| 도구 | 용도 |
|---|---|
| `where.py` | 현재 ego 위치·링크·신호등 조회 (좌표 찍기) |
| `make_spot.py` | 시작/끝점 좌표로 지점 생성 (도로망 라우팅) |
| `gen_spots_from_mgeo.py` | MGeo 에서 신호등 접근로 자동 생성 |
| `capture_spot.py` | 텔레옵으로 궤적 캡처 (`capture.sh`) |
| `edit_spot.py` | 지점의 신호등 ID·has_left 등 수정 |
| `spot_report.py` | 지점 검수 + 클래스 커버리지 |
| `check_visibility.py` | 신호등이 카메라 화각에 잡히는지 검사 |
| `collect.py` | **전체 매트릭스 수집** (재개 가능) |
| `test_run.py` | 지점 하나로 수집 시나리오 시험 |
| `test_objects.py` | 객체 배치 미리보기 |
| `prune_frames.py` | 수집분 일부 삭제 + 번호 재정렬 (오염/화각밖 제거) |
| `delete_spot_data.py` | 지점 데이터 통째 삭제 (재수집용) |
| `live_infer.py` | **실시간 추론** (ROS/UDP) |

### 학습·추론 (`train/`)
| 도구 | 용도 |
|---|---|
| `run.sh` | 컨테이너 헬퍼: `build/check/checkdata/train/serve/infer/analyze/shell` |
| `train.py` | ResNet18 학습 |
| `dataset.py` | manifest → Dataset (주행단위 계층분할) |
| `check_dataset.py` | 데이터셋 무결성 검사 |
| `infer.py` | 이미지/매니페스트 배치 추론 |
| `serve.py` | TCP 추론 서버 (실시간용) |
| `analyze_val.py` | val 오류를 지점×거리×클래스로 분해 |
| `bench_infer.py` | 해상도별 추론 속도 측정 |
| `Dockerfile` | cu128 학습 환경 |

---

## 6. 대회 PC 이관 체크리스트

1. `train/`, `src/traffic_runner/`, `train/runs/<모델>/best.pt` 복사
2. `cd train && ./run.sh build && ./run.sh check` — GPU(4060=sm_89) 정상 확인
3. MORAI **Sensor Setting**: 해상도·FOV·UDP Destination IP/Port 확인 (1절)
4. `./run.sh serve --ckpt runs/<모델>/best.pt` (컨테이너)
5. `python3 tools/live_infer.py --compete --udp-port <포트>` (WSL)

---

## 구조
```
traffic_runner/               catkin 워크스페이스 루트
├── capture.sh                텔레옵 캡처 실행
├── spots/                    지점 정의 (sig_*, unk_*)
├── dataset/                  수집 결과 (images/ + manifest.csv)
├── train/                    학습·추론 컨테이너
└── src/
    ├── morai_msgs →          auto_ws/src/morai_msgs (심볼릭 링크)
    └── traffic_runner/
        ├── config/runtime.yaml
        ├── tl/               신호 상태 정의 + 연출/검증
        ├── sim/              gRPC world, 카메라(ROS/UDP)
        ├── spot/             지점 스키마 + MGeo 라우팅
        ├── drive/            pure pursuit
        ├── collect/          수집 매트릭스·객체·레코더
        └── tools/            위 도구들
```

MORAI gRPC SDK 는 `auto_ws/src/auto_scenario_runner/grpc_inha_univ` 를 경로 참조로 재사용.
설계 상세: [docs/DESIGN.md](docs/DESIGN.md)
