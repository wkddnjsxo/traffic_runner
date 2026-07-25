"""
'지점(spot)' 정의 — 시작점 / 끝점 / 그 사이 주행 경로 / 신호등 정보.

한 spot = 데이터 수집의 최소 주행 단위다.
  시작점에 ego 를 놓고 -> path 를 pure pursuit 로 추종 -> 끝점 도달 시 1회 주행 종료.

주행을 pure pursuit 로 하기 때문에 시작점·끝점 두 좌표만으로는 부족하다
(교차로 접근로는 대개 곡선이라 직선 보간이 차선을 벗어난다).
그래서 캡처 툴은 시작점 마킹 후 사람이 실제로 몰고 간 궤적을 그대로 path 에 기록하고,
그 궤적이 pure pursuit 의 기준 경로가 된다.

YAML 스키마 (spots/<spot_id>.yaml):

    spot_id: sig_001
    map: R_KR_PR_K-city_2025
    kind: signal            # signal | unknown
    note: "K-city 4거리 남측 진입"
    traffic_light:
      ids: ["C1256W000003"] # ★ 사용자가 수집해서 채우는 값
      link_id: "A2256W000532"
      has_left: true        # 좌회전 화살표 램프(4구)가 있는가
                            #   true  -> 7종 수집 (red_yellow/red_left/green_left/left 포함)
                            #   false -> 3종 수집 (red/yellow/green)
    start: {x: .., y: .., z: .., yaw_deg: ..}
    end:   {x: .., y: .., z: .., yaw_deg: ..}
    arrival_radius_m: 3.0
    path_length_m: 87.3
    path:
      - [x, y, z, yaw_deg]
      - ...
    capture: {source: grpc, interval_m: 0.5, created_at: "..."}
"""

import os
import re

import yaml

from tl import states as tl_states


SPOT_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

KIND_SIGNAL = "signal"
KIND_UNKNOWN = "unknown"
KINDS = (KIND_SIGNAL, KIND_UNKNOWN)


class SpotError(ValueError):
    pass


class Spot(object):
    def __init__(self, data, source_path=None):
        self.data = data
        self.source_path = source_path

    # ------------------------------------------------------------- accessors
    @property
    def spot_id(self):
        return self.data["spot_id"]

    @property
    def kind(self):
        return self.data.get("kind", KIND_SIGNAL)

    @property
    def is_unknown(self):
        return self.kind == KIND_UNKNOWN

    @property
    def tl(self):
        return self.data.get("traffic_light") or {}

    @property
    def tl_ids(self):
        return list(self.tl.get("ids") or [])

    @property
    def verified_no_tl(self):
        """사람이 실제 화면으로 '신호등 안 보임' 을 확인했는가 (unknown 지점)."""
        return bool(self.data.get("verified_no_tl", False))

    @property
    def has_left(self):
        return bool(self.tl.get("has_left", False))

    @property
    def start(self):
        return self.data["start"]

    @property
    def end(self):
        return self.data["end"]

    @property
    def path(self):
        return self.data.get("path") or []

    @property
    def path_length_m(self):
        return float(self.data.get("path_length_m", 0.0))

    @property
    def arrival_radius_m(self):
        return float(self.data.get("arrival_radius_m", 3.0))

    def states(self):
        """
        이 지점에서 수집할 신호 상태 목록.

        unknown 지점은 신호등이 없으므로 연출 없이 1회만 주행하고 라벨은 unknown.
        signal 지점은 좌회전 화살표 유무에 따라 7종 또는 3종.
        """
        if self.is_unknown:
            return ["unknown"]
        override = self.tl.get("states")
        if override:
            for name in override:
                tl_states.get(name)  # 유효성 검증
            return list(override)
        return tl_states.states_for(self.has_left)

    def __repr__(self):
        return "<Spot %s kind=%s len=%.1fm states=%d>" % (
            self.spot_id, self.kind, self.path_length_m, len(self.states())
        )


# --------------------------------------------------------------------- io
def validate(data, strict=True):
    """
    spot dict 검증. 문제가 있으면 SpotError.

    strict=False 면 '아직 사용자가 안 채운 값'(신호등 ID 등)은 경고 목록으로만 돌려주고
    통과시킨다. 캡처 직후 저장 시점에는 strict=False, 수집 실행 직전에는 strict=True.
    """
    warnings = []

    spot_id = data.get("spot_id")
    if not spot_id or not SPOT_ID_RE.match(str(spot_id)):
        raise SpotError("spot_id 가 없거나 형식이 잘못됨(영숫자/_/- 만): %r" % spot_id)

    kind = data.get("kind", KIND_SIGNAL)
    if kind not in KINDS:
        raise SpotError("%s: kind 는 %s 중 하나여야 함 (현재 %r)" % (spot_id, KINDS, kind))

    for key in ("start", "end"):
        pose = data.get(key)
        if not isinstance(pose, dict):
            raise SpotError("%s: '%s' pose 가 없음" % (spot_id, key))
        for f in ("x", "y", "z", "yaw_deg"):
            if f not in pose:
                raise SpotError("%s: '%s' pose 에 '%s' 없음" % (spot_id, key, f))

    path = data.get("path") or []
    if len(path) < 2:
        raise SpotError(
            "%s: path 점이 %d개뿐. pure pursuit 추종에 쓸 궤적이 필요하다 "
            "(캡처 시 시작점 마킹 후 실제로 몰고 가야 기록된다)." % (spot_id, len(path))
        )
    for i, p in enumerate(path):
        if len(p) < 4:
            raise SpotError("%s: path[%d] 형식은 [x, y, z, yaw_deg] 이어야 함" % (spot_id, i))

    tl = data.get("traffic_light") or {}
    if kind == KIND_SIGNAL:
        ids = tl.get("ids") or []
        if not ids:
            msg = "%s: traffic_light.ids 가 비어 있음 (제어할 신호등 ID를 채워야 수집 가능)" % spot_id
            if strict:
                raise SpotError(msg)
            warnings.append(msg)
        if "has_left" not in tl:
            msg = "%s: traffic_light.has_left 미지정 (좌회전 화살표 유무)" % spot_id
            if strict:
                raise SpotError(msg)
            warnings.append(msg)
    else:
        if tl.get("ids"):
            warnings.append("%s: kind=unknown 인데 traffic_light.ids 가 있음 (무시됨)" % spot_id)

    return warnings


def load(path, strict=False):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SpotError("%s: YAML 최상위가 dict 가 아님" % path)
    validate(data, strict=strict)
    return Spot(data, source_path=path)


def load_all(spots_dir, strict=False):
    """spots 디렉터리의 모든 *.yaml 을 spot_id 순으로 읽는다."""
    spots = []
    if not os.path.isdir(spots_dir):
        return spots
    for name in sorted(os.listdir(spots_dir)):
        if not name.endswith((".yaml", ".yml")):
            continue
        if name.startswith("_"):  # _index.yaml 등 메타파일 제외
            continue
        spots.append(load(os.path.join(spots_dir, name), strict=strict))
    return spots


def save(data, spots_dir):
    """
    spot 을 spots_dir/<spot_id>.yaml 로 저장하고 경로를 돌려준다.

    path 는 줄당 한 점씩 flow style 로 써서 사람이 diff 로 읽을 수 있게 한다.
    """
    validate(data, strict=False)
    if not os.path.isdir(spots_dir):
        os.makedirs(spots_dir)
    out_path = os.path.join(spots_dir, "%s.yaml" % data["spot_id"])

    head_keys = ("spot_id", "map", "kind", "note", "verified_no_tl", "traffic_light",
                 "start", "end", "arrival_radius_m", "path_length_m", "capture")
    head = {k: data[k] for k in head_keys if k in data}  # py3.7+ 삽입순서 유지

    with open(out_path, "w") as f:
        f.write(yaml.safe_dump(head, allow_unicode=True, default_flow_style=False,
                               sort_keys=False))
        f.write("\npath:\n")
        for p in data["path"]:
            f.write("  - [%.4f, %.4f, %.4f, %.3f]\n" % (p[0], p[1], p[2], p[3]))
    return out_path


# ------------------------------------------------------------------ reporting
def next_spot_id(spots_dir, kind):
    """sig_001 / unk_001 형태로 다음 번호를 만든다."""
    prefix = "unk" if kind == KIND_UNKNOWN else "sig"
    used = set()
    if os.path.isdir(spots_dir):
        for name in os.listdir(spots_dir):
            m = re.match(r"^%s_(\d+)\.ya?ml$" % prefix, name)
            if m:
                used.add(int(m.group(1)))
    n = 1
    while n in used:
        n += 1
    return "%s_%03d" % (prefix, n)


def combination_count(spots, n_envs, n_object_seeds):
    """
    수집 조합 총 개수 = sum_over_spots( n_envs * n_object_seeds * len(states) ).

    사용자 설계상 객체 설정(seed)이 바깥 루프, 신호 상태가 안쪽 루프다.
    """
    total = 0
    per_spot = []
    for s in spots:
        n = n_envs * n_object_seeds * len(s.states())
        per_spot.append((s.spot_id, len(s.states()), n))
        total += n
    return total, per_spot
