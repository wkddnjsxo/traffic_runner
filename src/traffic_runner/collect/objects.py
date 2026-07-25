"""
객체(NPC 차량 / 장애물) 배치.

설계 원칙 두 가지.

1. **객체는 신호 상태와 완전히 독립이다.**
   배치는 (지점, object_seed) 만으로 결정된다. 날씨·시간·신호 상태는 시드에
   들어가지 않는다. 그래서 같은 seed 로 만든 배치 위에서 모든 신호 상태를
   반복해도 배치가 그대로 유지되고, 객체와 신호 라벨 사이에 상관이 생기지 않는다.

2. **객체는 끝점 '너머' 에만 놓는다.**
   ego 는 시작점→끝점 구간만 주행하므로 그 구간에 객체를 놓으면 들이받는다.
   끝점 이후 도로를 mgeo 로 연장해서 그 위에 배치하면, ego 와 충돌하지 않으면서
   카메라 정면(신호등 주변)에는 잡힌다.

seed 0 은 항상 '객체 없음' 이다 (깨끗한 기준 세트).
"""

import hashlib
import math
import random


# ---------------------------------------------------------------- 모델 카탈로그
# MORAI 시뮬레이터 GetAvailableObject 로 실측 검증된 이름들.
#   kind: 'vehicle' -> spawn_vehicle, 'obstacle' -> spawn_obstacle
CATALOG = {
    "sedan": {
        "kind": "vehicle",
        "models": ["2014_Kia_K7", "2015_Kia_K5", "2020_Kia_Stinger"],
    },
    "suv": {
        "kind": "vehicle",
        "models": ["2021_Volkswagen_Golf_GTI"],
    },
    "standard": {
        "kind": "obstacle",
        "models": ["OBJ_Kia_Sorento", "WoodBox"],
    },
    "custom": {
        "kind": "obstacle",
        "models": ["obj_hdmighty_2022_aerialworkplatform",
                   "obj_rnpremium_2014_garbage"],
    },
}

ALL_MODELS = [(cat, m) for cat, spec in CATALOG.items() for m in spec["models"]]


class Placement(object):
    """객체 하나의 배치."""

    __slots__ = ("category", "model", "kind", "x", "y", "z", "yaw_deg",
                 "label", "ahead_m", "lateral_m", "lane")

    def __init__(self, category, model, kind, x, y, z, yaw_deg, label,
                 ahead_m, lateral_m, lane="same"):
        self.category = category
        self.model = model
        self.kind = kind
        self.x, self.y, self.z = x, y, z
        self.yaw_deg = yaw_deg
        self.label = label
        self.ahead_m = ahead_m       # 끝점 진행방향 기준 전방 거리
        self.lateral_m = lateral_m   # 끝점 진행방향 기준 횡방향(+좌 / -우)
        self.lane = lane             # 'same' | 'opposite'

    def __repr__(self):
        return ("<%s %s %s @+%.0fm lat%+.1fm>"
                % (self.lane, self.category, self.model, self.ahead_m, self.lateral_m))


def assign_slots(spots, n_seeds):
    """
    (spot_id, seed) -> 라운드로빈 슬롯 번호. 모델을 균등하게 돌리는 데 쓴다.

    ★ 왜 그냥 순서대로 매기지 않는가 ★

    "빨간불엔 트럭이 있더라" 를 학습하는 것을 막는 1차 방어는 루프 구조다:
    객체는 (지점, seed) 로 고정되고 그 위에서 모든 신호 상태가 돌므로,
    한 배치 안에서 객체와 신호는 완전히 독립이다. 라운드로빈은 이 성질을
    건드리지 않는다.

    그런데 2차 상관이 생길 수 있다. 신호 상태 집합이 지점마다 다르기 때문이다
    (3구 신호등 지점은 red/yellow/green 만, 4구는 7종). 모델을 전체 조합에
    통짜로 돌리면 어떤 모델이 3구 지점에만 걸리고 4구 지점에는 한 번도 안 걸릴 수
    있다. 그러면 그 모델은 left 계열 신호와 한 번도 같이 안 나타나고,
    결국 "이 차가 보이면 좌회전 신호는 아니다" 라는 상관이 데이터에 남는다.

    그래서 **같은 상태 집합을 가진 지점끼리 묶어서 그룹 안에서 따로 라운드로빈**
    한다. 이러면 모든 모델이 모든 상태 집합에 고르게 등장한다.
    """
    groups = {}
    for spot in sorted(spots, key=lambda s: s.spot_id):
        if spot.is_unknown:
            continue
        key = tuple(spot.states())
        groups.setdefault(key, []).append(spot.spot_id)

    slots = {}
    for key, spot_ids in sorted(groups.items()):
        i = 0
        for spot_id in spot_ids:
            for seed in range(1, n_seeds):   # seed 0 = 객체 없음
                slots[(spot_id, seed)] = i
                i += 1
    return slots


def models_for_slot(slot, count):
    """슬롯 번호에서 모델 count 개를 라운드로빈으로 뽑는다."""
    n = len(ALL_MODELS)
    return [ALL_MODELS[(slot * count + k) % n] for k in range(count)]


def _seed_for(spot_id, object_seed):
    """
    (지점, seed) 로 결정론적 시드를 만든다.

    hash() 는 파이썬 실행마다 값이 달라지므로(PYTHONHASHSEED) 쓸 수 없다.
    재실행·재개 시에도 같은 배치가 나와야 하므로 md5 를 쓴다.
    """
    key = "%s|%d" % (spot_id, int(object_seed))
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _polyline_pose_at(pts, s):
    """폴리라인에서 시작부터 s(m) 지점의 (x, y, z, heading_rad)."""
    acc = 0.0
    for i in range(len(pts) - 1):
        ax, ay, az = pts[i][0], pts[i][1], pts[i][2]
        bx, by, bz = pts[i + 1][0], pts[i + 1][1], pts[i + 1][2]
        seg = math.hypot(bx - ax, by - ay)
        if seg < 1e-9:
            continue
        if acc + seg >= s:
            t = (s - acc) / seg
            return (ax + (bx - ax) * t, ay + (by - ay) * t, az + (bz - az) * t,
                    math.atan2(by - ay, bx - ax))
        acc += seg
    ax, ay, az = pts[-2][0], pts[-2][1], pts[-2][2]
    bx, by, bz = pts[-1][0], pts[-1][1], pts[-1][2]
    return bx, by, bz, math.atan2(by - ay, bx - ax)


def _candidates_in_view(pts, end, min_ahead, max_ahead, max_lateral):
    """
    폴리라인 위에서 '끝점 정면에 보이는' 자리만 골라낸다.

    도로가 끝점 직후 꺾이는 지점이 있다. 그런 곳은 차선을 따라가도 카메라 화각
    밖으로 벗어난다(실측: 전방 11m 인데 옆으로 16.7m). 그래서 폴리라인 진행거리가
    아니라 끝점 진행방향 기준 전방/횡 성분으로 판정한다.

    반환 [(x, y, z, heading_rad, fwd, lat), ...]
    """
    end_yaw = math.radians(float(end.get("yaw_deg", 0.0)))
    cos_e, sin_e = math.cos(end_yaw), math.sin(end_yaw)
    total = _polyline_length(pts)

    out = []
    s = 0.0
    while s <= total:
        x, y, z, heading = _polyline_pose_at(pts, s)
        dx, dy = x - end["x"], y - end["y"]
        fwd = cos_e * dx + sin_e * dy
        lat = -sin_e * dx + cos_e * dy
        if min_ahead <= fwd <= max_ahead and abs(lat) <= max_lateral:
            out.append((x, y, z, heading, fwd, lat))
        s += 0.5
    return out


def _polyline_length(pts):
    return sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
               for i in range(len(pts) - 1))


def plan_positions(spot, object_seed, mgeo, cfg=None):
    """
    모델을 빼고 '자리' 만 계산한다. 같은 입력이면 항상 같은 결과.

    모델 배정을 뒤로 미루는 이유: 배치는 실패할 수 있다(끝점 직후 도로가 꺾이거나
    반대차선이 화각 밖이면 자리가 안 나온다). 자리를 잡기 전에 모델을 배정하면
    실패한 슬롯에 배정된 모델이 조용히 사라져서, 그 모델만 특정 신호 상태와
    한 번도 같이 안 나타나는 상관이 생긴다(실측으로 잡힌 버그).

    반환 (positions, meta). positions = [{'x','y','z','yaw_deg','lane','ahead_m','lateral_m'}]
    """
    cfg = cfg or {}
    # 끝점 바로 너머의 좁은 띠에만 놓는다. 너무 멀면 화면에서 작아져 신호등 주변
    # 장면을 바꾸는 효과가 없고, 너무 가까우면 ego 가 끝점에서 들이받는다.
    min_ahead = float(cfg.get("min_ahead_m", 5.0))
    max_ahead = float(cfg.get("max_ahead_m", 25.0))
    # 같은 차선: 차선 안에서의 미세 흔들림만 (차선 폭만큼 밀면 옆 차선이 된다)
    jitters = list(cfg.get("lateral_offsets_m", [-0.6, 0.0, 0.6]))
    max_lateral = float(cfg.get("max_lateral_m", 2.5))
    # 반대차선은 애초에 옆으로 떨어져 있으므로 허용 폭이 따로 필요하다
    max_lateral_opp = float(cfg.get("max_lateral_opposite_m", 20.0))
    min_gap = float(cfg.get("min_gap_m", 5.0))
    n_same = int(cfg.get("same_lane_objects", 1))
    n_opp = int(cfg.get("opposite_lane_objects", 1))

    if int(object_seed) <= 0:
        return [], {"reason": "seed 0 = 객체 없음", "notes": []}

    rng = random.Random(_seed_for(spot.spot_id, object_seed))
    end = spot.end
    positions = []
    notes = []

    # ---- 같은 차선 (끝점 너머) ----
    links = spot.data.get("capture", {}).get("links") or []
    last_link = links[-1] if links else spot.tl.get("link_id", "")
    pts, ext = mgeo.extend_beyond(last_link, (end["x"], end["y"]), max_ahead + 10.0)
    same_cands = (_candidates_in_view(pts, end, min_ahead, max_ahead, max_lateral)
                  if len(pts) >= 2 else [])
    if not same_cands:
        notes.append("같은 차선에 자리 없음 (끝점 직후 도로가 꺾임, 가용 %.0fm)"
                     % ext.get("available_m", 0.0))

    used = []
    for k in range(n_same):
        c = _pick_spaced(rng, same_cands, used, min_gap) if same_cands else None
        if c is None:
            break
        x, y, z, heading, fwd, lat = c
        used.append(fwd)
        j = rng.choice(jitters)
        positions.append({
            "x": x - math.sin(heading) * j, "y": y + math.cos(heading) * j,
            "z": z + 0.2, "yaw_deg": math.degrees(heading),
            "lane": "same", "ahead_m": fwd, "lateral_m": lat + j,
            "label": "tr_%s_s%d_same%d" % (spot.spot_id, object_seed, k)})

    # ---- 반대차선 (마주오는 방향) ----
    opp_pts, opp_meta = mgeo.opposite_polyline(
        (end["x"], end["y"]), float(end.get("yaw_deg", 0.0)),
        probe_ahead_m=(min_ahead + max_ahead) / 2.0,
        length_m=max_ahead + 30.0)
    opp_cands = (_candidates_in_view(opp_pts, end, min_ahead, max_ahead, max_lateral_opp)
                 if len(opp_pts) >= 2 else [])
    if not opp_cands:
        notes.append("반대차선에 자리 없음 (%s)" % opp_meta.get("error", "화각 밖"))

    used_opp = []
    for k in range(n_opp):
        c = _pick_spaced(rng, opp_cands, used_opp, min_gap) if opp_cands else None
        if c is None:
            break
        x, y, z, heading, fwd, lat = c
        used_opp.append(fwd)
        j = rng.choice(jitters)
        positions.append({
            "x": x - math.sin(heading) * j, "y": y + math.cos(heading) * j,
            "z": z + 0.2, "yaw_deg": math.degrees(heading),
            "lane": "opposite", "ahead_m": fwd, "lateral_m": lat + j,
            "label": "tr_%s_s%d_opp%d" % (spot.spot_id, object_seed, k)})

    positions.sort(key=lambda p: p["ahead_m"])
    return positions, {"reason": "ok" if positions else ("; ".join(notes) or "배치 실패"),
                       "notes": notes,
                       "opposite_lateral_m": opp_meta.get("lateral_m")}


def plan_all(spots, n_seeds, mgeo, cfg=None):
    """
    모든 (지점, seed) 조합의 배치를 한 번에 만든다. 모델은 여기서 배정한다.

    ★ 모델을 여기서 배정하는 이유 ★

    "빨간불엔 트럭이 있더라" 를 막는 1차 방어는 루프 구조다 — 객체는 (지점, seed)
    로 고정되고 그 위에서 모든 신호 상태가 돌므로, 한 배치 안에서 객체와 신호는
    독립이다.

    2차 상관은 지점마다 신호 상태 집합이 다른 데서 온다(3구 지점은 3종, 4구는 7종).
    어떤 모델이 3구 지점에만 걸리면 left 계열 신호와 한 번도 같이 안 나타나고,
    "이 차가 보이면 좌회전이 아니다" 라는 상관이 남는다.

    그래서 **같은 상태 집합을 가진 지점끼리 묶어, 실제로 놓인 자리들에 대해서만**
    모델을 라운드로빈한다. '실제로 놓인' 이 중요하다 — 자리를 못 잡아 실패한
    슬롯까지 세면 그 모델이 통째로 누락된다.

    반환 dict[(spot_id, seed)] -> (placements, meta)
    """
    cfg = cfg or {}
    raw = {}
    groups = {}
    for spot in sorted(spots, key=lambda s: s.spot_id):
        if spot.is_unknown:
            continue
        key = tuple(spot.states())
        for seed in range(n_seeds):
            pos, meta = plan_positions(spot, seed, mgeo, cfg)
            raw[(spot.spot_id, seed)] = (spot, pos, meta)
            if pos:
                groups.setdefault(key, []).append((spot.spot_id, seed))

    out = {}
    n_models = len(ALL_MODELS)
    for key in sorted(groups):
        i = 0
        for combo in groups[key]:
            spot, pos, meta = raw[combo]
            placements = []
            for p in pos:
                cat, model = ALL_MODELS[i % n_models]
                i += 1
                placements.append(Placement(
                    category=cat, model=model, kind=CATALOG[cat]["kind"],
                    x=p["x"], y=p["y"], z=p["z"], yaw_deg=p["yaw_deg"],
                    label=p["label"], ahead_m=p["ahead_m"],
                    lateral_m=p["lateral_m"], lane=p["lane"]))
            out[combo] = (placements, meta)

    # 객체가 하나도 안 놓인 조합(seed 0 포함)도 채워 넣는다
    for combo, (spot, pos, meta) in raw.items():
        out.setdefault(combo, ([], meta))
    return out


def _pick_spaced(rng, candidates, used, min_gap):
    """이미 쓴 위치들과 min_gap 이상 떨어진 후보를 고른다."""
    for _ in range(40):
        c = rng.choice(candidates)
        if all(abs(c[4] - u) >= min_gap for u in used):
            return c
    return None


def describe(placements):
    """사람이 읽는 한 줄 요약."""
    if not placements:
        return "객체 없음"
    return ", ".join("%s[%s](+%.0fm,%+.1f)"
                     % (p.model, "동일" if p.lane == "same" else "반대",
                        p.ahead_m, p.lateral_m)
                     for p in placements)


class ObjectSpawner(object):
    """배치 계획을 시뮬에 올리고 내린다."""

    def __init__(self, world):
        self.world = world
        self.spawned = []

    def spawn(self, placements):
        """반환 (성공 수, 실패 목록)."""
        failed = []
        for p in placements:
            tf = self.world.make_transform(p.x, p.y, p.z, p.yaw_deg)
            actor = None
            try:
                if p.kind == "vehicle":
                    actor = self.world.world.spawn_vehicle(
                        transform=tf, model_name=p.model, label=p.label,
                        velocity=0.0)
                else:
                    actor = self.world.world.spawn_obstacle(
                        transform=tf, model_name=p.model, label=p.label)
            except Exception as exc:
                failed.append((p.model, str(exc)[:60]))
                continue
            if actor is None:
                failed.append((p.model, "spawn 실패 (모델명 확인)"))
            else:
                self.spawned.append(actor)
        return len(self.spawned), failed

    def clear(self):
        """이 클라이언트가 스폰한 액터를 전부 지운다."""
        for actor in self.spawned:
            try:
                actor.destroy()
            except Exception:
                pass
        self.spawned = []
        # 누락분까지 확실히 정리 (client_key 로 필터되어 사용자 액터는 안전)
        try:
            self.world.world.destroy_all_actors()
        except Exception:
            pass
