# traffic_runner 설계

MORAI 시뮬레이터에서 **신호등 인지 AI(ResNet18) 학습 데이터셋을 자동 수집**하는
시나리오 러너. 학습 코드는 이 저장소에 없다 — 여기는 수집 전용이다.

---

## 1. 출력 클래스

| idx | name | MORAI 값 | gRPC enum | 좌회전 필요 |
|----:|------|---------:|-----------|:---:|
| 0 | `red` | 1 | `TL_COLOR_R` | |
| 1 | `yellow` | 4 | `TL_COLOR_Y` | |
| 2 | `green` | 16 | `TL_COLOR_SG` | |
| 3 | `red_yellow` | 5 | `TL_COLOR_R_WITH_Y` | |
| 4 | `red_left` | 33 | `TL_COLOR_R_WITH_GLEFT` | ✔ |
| 5 | `green_left` | 48 | `TL_COLOR_G_WITH_GLEFT` | ✔ |
| 6 | `left` | 32 | `TL_COLOR_LG` | ✔ |
| 7 | `unknown` | — | — | |

**검증된 사실:** MORAI 의 gRPC `TrafficLightColor` enum 값과 ROS
`morai_msgs/SetTrafficLight.trafficLightStatus` 정수값이 **완전히 동일**하다
(실제 proto 모듈에서 확인). 따라서 제어 경로를 gRPC 로 하든 ROS 로 하든
`tl/states.py` 의 테이블 하나로 둘 다 커버된다. 클래스 정의를 바꿀 일이 생기면
**오직 `tl/states.py` 만** 고치면 된다.

`unknown` 은 연출값이 아니다. 신호등이 아예 없는 구간을 따로 캡처해서 얻는 라벨이다.

---

## 2. 수집 루프 구조

사용자 요구: *"객체는 신호 상태와 독립이어야 한다. 객체 설정 하나를 만든 뒤,
그 설정에서 모든 가능한 신호 상태를 반복한다."* → 루프 중첩 순서가 이 요구를 그대로 인코딩한다.

```
for env in [(SUNNY,11) (SUNNY,13) (SUNNY,15) (FOGGY,11) (FOGGY,13) (FOGGY,15)]:   # 6
  set_weather(env.weather); set_time(env.hour)

  for spot in spots:                        # 캡처한 지점들
    for object_seed in 0..N-1:              # 객체 설정 (0 = 객체 없음)
      spawn_objects(spot, object_seed)      # ← 신호 상태보다 바깥. 한 번 깔면 유지.

      for state in shuffle(spot.states()):  # 화살표 있으면 7종, 없으면 4종
        set_traffic_light(spot.tl_ids, state)
        teleport_ego(spot.start)
        drive_pure_pursuit(spot.path) → 매 프레임 이미지 저장 (라벨 = state)
        # 끝점 도달 시 종료 → 다음 상태로 (시작점 복귀)

      despawn_objects()
```

- **신호 상태 순서는 매 반복마다 랜덤 셔플** (`collect.shuffle_states: true`).
  고정 순서면 "빨강 다음엔 항상 노랑" 같은 환경 편향이 프레임 순서에 남는다.
- **객체는 신호 상태 루프 밖**에서 스폰하고 유지한다. 같은 객체 배치 위에서
  7가지 신호를 모두 찍으므로, 객체와 신호 라벨 사이의 상관이 데이터에 생기지 않는다.
- `unknown` 지점은 신호 연출 없이 seed 당 1회만 주행한다.

### 조합 수

`총 주행 횟수 = Σ_지점 ( 6 × object_seeds × len(states) )`

화살표 있는 지점 1개 + 객체 seed 3 → `6 × 3 × 7 = 126` 회.
`tools/spot_report.py` 가 이 값과 예상 소요시간, 클래스별 커버리지를 계산해 준다.

### 신호 전환 데이터셋

고정 상태 주행과 별개로, **주행 중 신호가 바뀌는** 시퀀스도 수집한다.
같은 지점/객체 설정에서 전환 시나리오를 하나의 "상태"처럼 취급한다.

```
GREEN → YELLOW → RED     (딜레마존)
RED → RED_YELLOW → GREEN (출발)
GREEN → GREEN_LEFT       (좌회전 개시)
```

전환 순간 프레임은 라벨이 모호하므로 **전환 명령 전후 `guard_frames` 만큼은
`transition` 태그를 붙여 매니페스트에 남기고 학습에서 제외하거나 별도 취급**한다.
전환 트리거는 "시작점으로부터의 진행거리"로 잡는다(시간 기준은 속도에 따라 흔들린다).

---

## 3. 지점(spot) — 이번에 구현한 부분

한 spot = 데이터 수집의 최소 주행 단위. **시작점 / 끝점 / 그 사이 경로 / 신호등 정보.**

### 왜 경로까지 기록하는가

주행을 **pure pursuit** 로 하므로 추종할 기준 경로가 필요하다.
시작점·끝점 두 좌표만으로 직선 보간하면 교차로 접근로 곡선에서 차선을 벗어난다.

### 생성 방법 두 가지

**(A) MGeo 자동 생성 — 주력** (`tools/gen_spots_from_mgeo.py`)

실측으로 확인한 사실:

| 검증 항목 | 결과 |
|---|---|
| MGeo 링크와 시뮬 좌표 일치 | 주행 중 ego 가 링크 폴리라인에서 **0.15m** 이내 |
| yaw 규약 | MORAI `rotation.z` == mgeo `atan2(dy,dx)` (정상 주행 시 차이 3~5°) |
| 생성 경로 heading 연속성 | 점프 중앙값 **3.6°**, 최대 11.4° |
| 끝점 ~ 정지선 거리 | 목표 5.0m, 오차 중앙값 **0.00m** |
| 링크 이탈 | **0건** |

MGeo 구조 (실측):
- `link_type '6'` = 일반 도로, `'1'` = 교차로 내부 링크
- 교차로 내부 링크의 `related_signal` 이 movement 를 표시:
  `straight` / `left` / `left_unprotected` / `right_unprotected` / `uturn_normal`
- `node.traffic_light_id` = 정지선을 관장하는 신호등
- `link.points` 는 항상 `from_node → to_node` 순서

**`has_left` 판정**: 정지선 노드에서 나가는 movement 에 `left`(보호좌회전)가 있으면
화살표 램프가 있다고 본다. `left_unprotected`(비보호)는 화살표가 없으므로 제외한다.
이 구분을 놓치면 화살표 없는 신호등에 `red_left` 를 연출하려다 라벨이 오염된다.

경로는 정지선에서 **뒤로** 거슬러 올라가며 만든다. 분기점에서는 진행방향이 가장
자연스럽게 이어지는 선행 링크를 고르고(교차로 내부 링크는 후순위), 90° 이상 꺾이면
이어붙이지 않는다 — 아무거나 고르면 옆길로 새서 신호등이 화면에서 사라진다.

**(B) 시작점/끝점 직접 지정** (`tools/make_spot.py`)

좌표 두 개를 주면 링크 그래프 위에서 Dijkstra 로 최단 경로를 찾아 잇는다.
각 점은 가장 가까운 링크에 수직 투영해 스냅한다(교차로 내부 링크는 후순위 —
시작점을 교차로 한복판에 잡는 건 대개 실수다).

kind(`signal`/`unknown`)는 명시할 수도 있고 `auto` 로 두면 경로가 신호등 정지선을
지나는지로 판정한다. `unknown` 으로 지정했는데 경로가 정지선을 지나면 경고한다 —
신호등이 화면 구석에라도 보이면 unknown 라벨이 오염되기 때문이다.

**일방통행 함정**: MORAI 도로망은 방향이 있다. 시작/끝을 거꾸로 주면 경로 탐색이
실패하는 게 아니라 **맵을 한 바퀴 돌아** 성립한다(실측: 직선 156m 짜리가 739m 경로로
신호등 8개를 지남). 그래서 경로/직선거리 비가 2.5배를 넘으면 저장을 막는다.

**(C) 텔레옵 캡처 — 보조** (`tools/capture_spot.py`)

자동 생성이 안 맞는 곳만 손으로 찍는다. **시작점 마킹 → 실제로 몰고 간 궤적 →
끝점 마킹**을 하나의 spot 으로 저장한다. 두 방식의 산출물은 스키마가 같아서
같은 디렉터리에 섞여도 되고(`gen_` vs `sig_`/`unk_` 접두사), 수집기는 구분하지 않는다.

### 파일 형식 (`spots/<spot_id>.yaml`)

```yaml
spot_id: sig_001
map: R_KR_PR_K-city_2025
kind: signal              # signal | unknown
note: 4거리 남측 진입
traffic_light:
  ids: [C1256W000003]     # ★ 사용자가 수집해서 채우는 값
  link_id: A2256W000532   # 캡처 시 ego 가 있던 링크 (참고용, 자동 기록)
  has_left: true          # 좌회전 화살표 유무 → 수집 상태가 7종/4종으로 갈린다
start: {x: .., y: .., z: .., yaw_deg: ..}
end:   {x: .., y: .., z: .., yaw_deg: ..}
arrival_radius_m: 3.0     # 끝점 도달 판정 반경
path_length_m: 57.6
capture: {source: grpc, interval_m: 0.5, created_at: '...'}
path:
  - [x, y, z, yaw_deg]    # 0.5m 간격 (커브에서는 5° 마다 추가)
  - ...
```

`traffic_light.states` 를 명시하면 자동 산출(7종/4종)을 덮어쓸 수 있다.
특정 지점만 일부 상태를 빼고 싶을 때 쓴다.

### 캡처 툴 동작

`tools/capture_spot.py` — 실행 중인 시뮬에 **재시작 없이 attach** 한다
(`start()` 를 호출하면 시뮬/UDP 네트워크가 초기화되어 운전 세션이 끊긴다).
ego pose 를 20Hz 로 폴링하면서 터미널 raw mode 로 단일키를 받는다.

경로점은 **0.5m 이동** 또는 **5° 방향변화** 중 먼저 오는 조건에서 기록한다.
거리 기준만 쓰면 급커브에서 점이 성겨져 pure pursuit 이 코너를 자른다.

> 폴링 주기 때문에 실제 간격은 양자화된다 (20Hz·8m/s → 0.4m/틱 → 실효 0.8m).
> 주행 시 `utils/geometry.resample()` 로 등간격 재샘플링해서 쓴다.

---

## 4. 워크스페이스 구조

`traffic_runner` 는 **독립 catkin 워크스페이스**로 만들었다.

- 카메라 이미지를 ROS 토픽(`/image_jpeg/compressed`)으로 받기로 했으므로 ROS 패키지여야 한다.
- 대회용 `auto_ws`(주행/채점)와 목적이 달라 분리하는 편이 낫다. 서로의 빌드를 깨지 않는다.
- 공통 자산은 **심볼릭 링크로 재사용**한다 — 복사본을 만들면 반드시 갈라진다.

```
traffic_runner/                          ← catkin 워크스페이스 루트
├── capture.sh                           지점 캡처 실행
├── docs/DESIGN.md
├── spots/                               ★ 캡처 산출물 (지점 YAML)
├── dataset/                             ★ 수집 산출물 (이미지 + 라벨)
└── src/
    ├── CMakeLists.txt → catkin toplevel.cmake   (심볼릭 링크)
    ├── morai_msgs    → auto_ws/src/morai_msgs   (심볼릭 링크)
    └── traffic_runner/
        ├── config/runtime.yaml
        ├── tl/states.py          신호 상태 ↔ MORAI 값 (단일 진실 원천)
        ├── sim/pose_source.py    ego pose (gRPC attach / ROS)
        ├── spot/schema.py        지점 YAML 로드·저장·검증
        ├── utils/                keyboard(raw mode), geometry
        └── tools/
            ├── capture_spot.py   ★ 지점 캡처 툴
            └── spot_report.py    지점 검수 + 조합 수 계산
```

MORAI gRPC SDK(`grpc_inha_univ`)는 `auto_ws` 것을 **경로 참조**로 쓴다
(`config/runtime.yaml` 의 `paths.grpc_src`). `auto_scenario_runner` 의
파이썬 모듈(`utils/`, `core/`)에는 의존하지 않는다 — 같은 이름의 패키지가 있어
import 가 섞이면 곤란하기 때문이다. 제조사 SDK 에만 의존한다.

---

## 5. 앞으로 구현할 모듈

| 모듈 | 역할 | 상태 |
|------|------|:---:|
| `tl/states.py` | 상태 ↔ MORAI 값 매핑 | ✅ |
| `sim/pose_source.py` | ego pose 읽기 | ✅ |
| `spot/schema.py` | 지점 정의 | ✅ |
| `spot/mgeo.py` + `tools/gen_spots_from_mgeo.py` | MGeo 지점 자동 생성 | ✅ |
| `tools/make_spot.py` | 시작점/끝점 직접 지정 (도로망 라우팅) | ✅ |
| `tools/where.py` | 현재 ego 위치·링크·신호등 조회 | ✅ |
| `tools/capture_spot.py` | 텔레옵 지점 캡처 | ✅ |
| `tools/spot_report.py` | 지점 검수 | ✅ |
| `tl/controller.py` | 신호등 연출 (gRPC/ROS) + 적용 검증 | ⬜ |
| `sim/world.py` | attach/날씨/시간/텔레포트/스폰 통합 브리지 | ⬜ |
| `drive/pure_pursuit.py` | 경로 추종 (`/ctrl_cmd` 발행) | ⬜ |
| `collect/objects.py` | 객체 seed → 스폰 배치 | ⬜ |
| `collect/recorder.py` | 이미지 저장 + 매니페스트 CSV | ⬜ |
| `collect/matrix.py` | 조합 생성 + 진행상황 저장(재개 가능) | ⬜ |
| `collect/runner_node.py` | 메인 오케스트레이터 | ⬜ |
| `collect/transitions.py` | 신호 전환 시퀀스 | ⬜ |

### 해결됨: 신호등 ID 형식 = MGeo ID

**시뮬레이터가 쓰는 신호등 ID 는 MGeo 의 `traffic_light_id` 와 같다.** 실측:

ego 가 링크 `A2256W000219` 위에 있을 때
`ActorState.vehicle_state.tl_id` 가 **`C1256W000077`** 을 반환했고,
MGeo 에서 그 링크의 `to_node.traffic_light_id` 도 **`C1256W000077`** 로 일치했다.

따라서 `spots/*.yaml` 의 `traffic_light.ids` 에 들어 있는 MGeo ID 를
`set_traffic_light_info()` 에 그대로 넘기면 된다. 매핑 테이블이 필요 없다.

참고로 `GetTrafficLightInfo` / `GetIntersectionTLInfo` 조회는 attach 상태에서
전부 빈 응답(`phase: -1`)이었다. 이 API 들은 `start()` 로 시작된 세션을 요구하는 것으로
보인다. 하지만 ID 확인에는 `ActorState.tl_id` 로 충분했다.
`tools/where.py` 와 `tools/capture_spot.py` 가 이 값을 실시간으로 보여준다.

### 설계 시 주의할 점

- **신호등 연출 후 반드시 검증한다.** `set_traffic_light_info()` 의 반환값만
  믿지 말고 `get_traffic_light_info(GET_TL_INFO_BY_TL_ID)` 로 되읽어 실제 색을
  확인한 뒤 주행을 시작한다. 연출이 안 먹은 채로 주행하면 **라벨이 틀린 데이터**가
  쌓이는데, 이건 조용히 학습을 망가뜨리는 최악의 실패다.
- **`set_sibling`**: `set_traffic_light_info(..., sibling=True)` 는 같은 진입로의
  형제 신호등을 함께 바꾼다. 한 진입로에 신호등이 여러 개면 이게 편하지만,
  의도치 않은 신호등까지 바뀔 수 있으므로 지점별로 실측해서 정할 것.
- **`is_impulse=False`**(영구)로 걸어야 주행 내내 상태가 유지된다.
  `True` 는 일시적이라 시뮬 자체 신호 스케줄에 곧 덮어쓰인다.
- **이미지-라벨 동기화**: 이미지 타임스탬프가 신호 연출 완료 시각 이후인 것만 저장한다.
  연출 직후 몇 프레임은 이전 상태가 찍혀 있을 수 있다.
- **날씨/시간 변경 후 안정화 대기**가 필요하다 (렌더링이 즉시 반영되지 않는다).
- **재개 가능하게** 만든다. 수백~수천 회 주행이라 중간에 반드시 끊긴다.
  완료한 조합을 진행상황 파일에 남기고 재실행 시 건너뛴다.

---

## 6. 데이터셋 출력 형식 (예정)

```
dataset/
├── manifest.csv
└── images/SUNNY_11/sig_001/seed01/red/000000.jpg
```

`manifest.csv` 컬럼:
`image_path, label, label_index, spot_id, weather, hour, object_seed, state,
frame_idx, dist_to_end_m, ego_x, ego_y, ego_yaw, is_transition, run_id, timestamp`

ResNet18 학습은 별도 저장소에서 이 manifest 를 읽어 쓴다.
`dist_to_end_m`(신호등까지 거리)이 있으면 "너무 멀어서 안 보이는 프레임" 을
학습 시 걸러내거나 `unknown` 으로 재라벨링할 수 있다.
