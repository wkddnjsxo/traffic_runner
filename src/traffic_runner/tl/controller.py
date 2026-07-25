"""
신호등 연출 + 검증.

연출이 조용히 실패하면 **라벨이 틀린 데이터**가 쌓인다. 이건 학습을 망가뜨리는
가장 나쁜 실패 방식이라, 설정 후 반드시 되읽어 확인한다.

검증 채널: ego 가 신호등 앞에 있을 때 `ActorState.vehicle_state.tl_color` 가
시뮬레이터가 실제로 표시 중인 색을 알려준다. 이 값은 카메라가 보는 것과 같은
소스이므로 가장 믿을 만한 확인 수단이다.
(`GetTrafficLightInfo` RPC 는 attach 모드에서 빈 응답이라 쓸 수 없다.)
"""

import time

from tl import states as tl_states


class TrafficLightController(object):
    def __init__(self, world, sibling=True, settle_sec=0.4, verify_timeout=2.0,
                 strict=False):
        self.world = world
        self.sibling = sibling
        self.settle_sec = settle_sec
        self.verify_timeout = verify_timeout
        # strict=True 면 'ego 가 신호등을 감지 못함' 도 실패로 친다.
        # ego 가 신호등에 연결되는 자리(정지선 앞 링크)에서만 켤 것.
        self.strict = strict
        self.last_error = None
        self.skipped_verifications = 0

    def apply(self, tl_ids, state_name, verify=True):
        """
        신호등들을 state_name 상태로 만든다.

        반환: (ok, info)
          ok   : 전송 성공 여부 (verify=True 면 확인까지 통과해야 True)
          info : {'sent': [...], 'failed': [...], 'observed': int|None,
                  'verified': bool|None, 'reason': str}
        """
        st = tl_states.get(state_name)
        if st.morai_value is None:
            return True, {"sent": [], "failed": [], "observed": None,
                          "verified": None, "reason": "unknown 은 연출 대상이 아님"}

        color = st.morai_value
        sent, failed = [], []
        for tl_id in tl_ids:
            try:
                ok = self.world.set_traffic_light(tl_id, color, impulse=False,
                                                 sibling=self.sibling)
            except Exception as exc:
                ok = False
                self.last_error = str(exc)
            (sent if ok else failed).append(tl_id)

        info = {"sent": sent, "failed": failed, "observed": None,
                "verified": None, "reason": ""}

        if failed:
            info["reason"] = "set_traffic_light 실패: %s" % ", ".join(failed)
            return False, info

        if self.settle_sec > 0:
            time.sleep(self.settle_sec)

        if not verify:
            return True, info

        ok, observed, reason = self.verify(state_name)
        if observed is None or observed == -2:
            self.skipped_verifications += 1
        info["observed"] = observed
        info["verified"] = ok
        info["reason"] = reason
        return ok, info

    def verify(self, state_name):
        """
        ego 가 보고하는 tl_color 가 기대한 상태와 맞는지 확인한다.

        반환 (ok, observed_value, reason).
        ego 가 신호등을 감지하지 못하는 위치면 확인 불가 -> (True, None, '감지 안 됨')
        으로 통과시킨다. 신호등이 안 보이는 곳에서 못 봤다고 실패로 칠 수는 없다.
        """
        expect = tl_states.get(state_name).morai_value
        deadline = time.time() + self.verify_timeout
        observed = None
        while time.time() < deadline:
            state = self.world.ego_state()
            if state is None:
                time.sleep(0.05)
                continue
            observed = state["tl_color"]
            if observed == expect:
                return True, observed, "일치"
            # -2 = NOT_DETECTED : ego 가 아직 신호등을 못 잡은 상태
            if observed != -2:
                # 다른 색이 보이면 잠깐 더 기다려본다 (반영 지연)
                time.sleep(0.1)
                continue
            time.sleep(0.1)

        if observed is None or observed == -2:
            if self.strict:
                return False, observed, (
                    "ego 가 신호등을 감지 못함 — 연출이 먹었는지 확인할 수 없다. "
                    "시작점이 신호등 링크에서 너무 멀거나 신호등 ID 가 틀렸을 수 있다.")
            return True, observed, "ego 가 신호등을 감지 못함 (확인 생략 — 라벨 미검증)"
        return False, observed, ("기대 %s(%d) != 실제 %s(%d)"
                                 % (state_name, expect,
                                    _name_of(observed), observed))

    def turn_off(self, tl_ids):
        """신호등을 소등한다 (-1)."""
        for tl_id in tl_ids:
            try:
                self.world.set_traffic_light(tl_id, tl_states.MORAI_OFF,
                                             impulse=False, sibling=self.sibling)
            except Exception:
                pass


def _name_of(value):
    for st in tl_states.STATES.values():
        if st.morai_value == value:
            return st.name
    return {0: "unspecified", -1: "off", -2: "not_detected"}.get(value, "?")
