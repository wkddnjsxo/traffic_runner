"""
수집 매트릭스 — 조합 열거 + 진행상황 저장(재개).

루프 중첩 순서가 사용자 요구를 그대로 인코딩한다:

  환경(날씨×시간)              ← 가장 바깥. 설정 비용이 크고 렌더 안정화가 필요하다.
    지점
      객체 seed                ← 신호 상태보다 바깥. 배치를 깔고 유지한다.
        신호 상태 (랜덤 순서)   ← 가장 안쪽. 매번 시작점으로 돌아간다.

객체가 신호 상태보다 바깥이어야 "객체는 신호 상태와 독립" 이 성립한다.
같은 배치 위에서 모든 신호를 찍으므로 객체와 라벨 사이에 상관이 생기지 않는다.

수백 회 주행이라 중간에 반드시 끊긴다. 완료한 조합을 한 줄씩 파일에 남기고,
재실행 시 건너뛴다.
"""

import os
import random


class Combo(object):
    __slots__ = ("weather", "hour", "spot_id", "object_seed", "state")

    def __init__(self, weather, hour, spot_id, object_seed, state):
        self.weather = weather
        self.hour = hour
        self.spot_id = spot_id
        self.object_seed = object_seed
        self.state = state

    @property
    def key(self):
        return "%s|%s|%s|%d|%s" % (self.weather, self.hour, self.spot_id,
                                   self.object_seed, self.state)

    def __repr__(self):
        return "<%s>" % self.key


def build(spots, weathers, hours, n_seeds, shuffle_states=True, seed=None):
    """
    조합 전체를 루프 순서대로 열거한다.

    신호 상태 순서는 (환경, 지점, 객체seed) 마다 새로 섞는다. 고정 순서면
    "빨강 다음엔 항상 노랑" 같은 순서 편향이 프레임 시퀀스에 남는다.
    seed 를 주면 셔플이 재현 가능해진다(재개 시 같은 순서).
    """
    combos = []
    for weather in weathers:
        for hour in hours:
            for spot in spots:
                states_all = spot.states()
                for object_seed in range(n_seeds):
                    states = list(states_all)
                    if shuffle_states and len(states) > 1:
                        # 조합마다 결정론적으로 다른 순서
                        rng = random.Random(
                            "%s|%s|%s|%d|%s" % (weather, hour, spot.spot_id,
                                                object_seed, seed))
                        rng.shuffle(states)
                    for state in states:
                        combos.append(Combo(weather, hour, spot.spot_id,
                                            object_seed, state))
    return combos


class Progress(object):
    """
    완료한 조합을 기록해 재개를 가능하게 한다.

    한 줄에 조합 키 하나. append-only 라 중간에 죽어도 앞부분은 살아남는다.
    """

    def __init__(self, path):
        self.path = path
        self.done = set()
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        self.done.add(line)
        d = os.path.dirname(os.path.abspath(path))
        if d and not os.path.isdir(d):
            os.makedirs(d)
        self._fp = open(path, "a", buffering=1)

    def is_done(self, combo):
        return combo.key in self.done

    def mark(self, combo):
        if combo.key in self.done:
            return
        self.done.add(combo.key)
        self._fp.write(combo.key + "\n")

    def close(self):
        try:
            self._fp.close()
        except Exception:
            pass


def group_runs(combos):
    """
    조합을 (환경, 지점, 객체seed) 단위로 묶는다.

    이 단위 안에서는 환경 설정과 객체 스폰을 한 번만 하면 되므로,
    묶어서 처리하면 gRPC 호출과 렌더 안정화 대기가 크게 줄어든다.
    """
    groups = []
    cur_key = None
    cur = []
    for c in combos:
        key = (c.weather, c.hour, c.spot_id, c.object_seed)
        if key != cur_key:
            if cur:
                groups.append((cur_key, cur))
            cur_key, cur = key, []
        cur.append(c)
    if cur:
        groups.append((cur_key, cur))
    return groups
