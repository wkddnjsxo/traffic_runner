"""
신호 상태 정의 — 이 프로젝트의 단일 진실 원천(single source of truth).

ResNet18 출력 클래스 8종과 MORAI 신호등 제어값을 여기 한 곳에서만 매핑한다.
라벨링/제어/설정검증이 전부 이 테이블을 참조하므로, 클래스를 늘리거나 값을
바꿀 일이 생기면 여기만 고치면 된다.

중요: MORAI 의 gRPC TrafficLightColor enum 값과 ROS morai_msgs/SetTrafficLight 의
trafficLightStatus 정수값은 **동일**하다 (실측 확인:
TL_COLOR_R=1, TL_COLOR_Y=4, TL_COLOR_SG=16, TL_COLOR_LG=32,
TL_COLOR_R_WITH_Y=5, TL_COLOR_R_WITH_GLEFT=33, TL_COLOR_G_WITH_GLEFT=48).
따라서 전송 방식(gRPC / ROS)이 달라도 아래 morai_value 하나로 둘 다 제어된다.
"""

from collections import OrderedDict


# 소등(default). 신호등을 끌 때 쓰는 값이며 클래스가 아니다.
MORAI_OFF = -1


#: 상태를 연출하려면 신호등에 어떤 램프가 있어야 하는지
REQ_NONE = None
REQ_LEFT = "left"      # 좌회전 화살표 램프 (4구 신호등)


class TLState(object):
    """하나의 신호 상태 정의."""

    __slots__ = ("name", "morai_value", "grpc_enum", "requires", "label_index", "desc")

    def __init__(self, name, morai_value, grpc_enum, requires, label_index, desc):
        self.name = name                    # 설정파일/파일명/라벨에 쓰는 표준 이름
        self.morai_value = morai_value      # ROS SetTrafficLight.trafficLightStatus == gRPC enum 값
        self.grpc_enum = grpc_enum          # gRPC TrafficLightColor enum 심볼명
        self.requires = requires            # 연출에 필요한 램프 (REQ_LEFT 또는 None)
        self.label_index = label_index      # ResNet18 출력 인덱스
        self.desc = desc

    @property
    def needs_left(self):
        return self.requires == REQ_LEFT

    def __repr__(self):
        return "TLState(%s, morai=%s, idx=%d)" % (self.name, self.morai_value,
                                                  self.label_index)


# ---------------------------------------------------------------------------
# 클래스 정의. label_index 순서 = 학습 시 클래스 인덱스 순서.
# 이 순서는 한 번 정하면 바꾸지 말 것 (바꾸면 기존 수집분과 라벨이 어긋난다).
#
# requires 는 "이 상태를 연출하려면 신호등에 그 램프가 있어야 한다" 는 뜻이다.
# 램프가 없는 신호등에 연출하면 화면에 안 나타나거나 엉뚱하게 표시되어
# 라벨이 오염된다.
# ---------------------------------------------------------------------------
_DEFS = [
    #  name           morai  grpc enum                 requires   idx  desc
    ("red",             1,   "TL_COLOR_R",             REQ_NONE,  0, "적색"),
    ("yellow",          4,   "TL_COLOR_Y",             REQ_NONE,  1, "황색"),
    ("green",          16,   "TL_COLOR_SG",            REQ_NONE,  2, "녹색(직진)"),
    ("red_yellow",      5,   "TL_COLOR_R_WITH_Y",      REQ_LEFT,  3, "적색+황색"),
    ("red_left",       33,   "TL_COLOR_R_WITH_GLEFT",  REQ_LEFT,  4, "적색+좌회전"),
    ("green_left",     48,   "TL_COLOR_G_WITH_GLEFT",  REQ_LEFT,  5, "녹색+좌회전"),
    ("left",           32,   "TL_COLOR_LG",            REQ_LEFT,  6, "좌회전 단독"),
    ("unknown",      None,   None,                     REQ_NONE,  7, "신호등 없음/판별불가"),
]

STATES = OrderedDict()
for _n, _m, _g, _r, _i, _d in _DEFS:
    STATES[_n] = TLState(_n, _m, _g, _r, _i, _d)

#: 학습 클래스 이름 (label_index 순)
CLASS_NAMES = [s.name for s in sorted(STATES.values(), key=lambda s: s.label_index)]
NUM_CLASSES = len(CLASS_NAMES)

#: 실제로 신호등에 '연출'할 수 있는 상태 (unknown 제외)
CONTROLLABLE = [n for n, s in STATES.items() if s.morai_value is not None]

#: 추가 램프 없이(적/황/녹 3구만으로) 수집 가능한 상태
STATES_BASIC = [n for n in CONTROLLABLE if STATES[n].requires is None]

#: 좌회전 램프(4구 신호등)가 있어야 수집 가능한 상태
STATES_LEFT_ONLY = [n for n in CONTROLLABLE if STATES[n].needs_left]


def get(name):
    """이름으로 TLState 조회. 대소문자/공백 무시."""
    key = str(name).strip().lower()
    if key not in STATES:
        raise KeyError(
            "알 수 없는 신호 상태 '%s'. 사용 가능: %s" % (name, ", ".join(STATES.keys()))
        )
    return STATES[key]


def morai_value(name):
    """상태 이름 -> MORAI 제어 정수값. unknown 은 제어 대상이 아니므로 에러."""
    st = get(name)
    if st.morai_value is None:
        raise ValueError("'%s' 는 연출 대상이 아니다 (신호등 없는 지점의 라벨)." % name)
    return st.morai_value


def label_index(name):
    """상태 이름 -> 학습 클래스 인덱스."""
    return get(name).label_index


def states_for(has_left):
    """
    지점의 신호등이 가진 램프에 따라 수집할 상태 목록을 준다.

      3구 신호등 (적/황/녹)        : red, yellow, green                        → 3종
      4구 신호등 (+좌회전 화살표)   : + red_yellow, red_left, green_left, left  → 7종

    red_yellow 가 좌회전 그룹에 있는 이유: 적+황 동시 점등은 좌회전 화살표가 있는
    신호등의 페이즈 전환에서만 나타난다. 3구 신호등에 연출하면 실제로는 볼 수 없는
    조합이라 라벨이 오염된다.
    """
    out = list(STATES_BASIC)
    if has_left:
        out += STATES_LEFT_ONLY
    # label_index 순으로 정렬해 출력 순서를 안정시킨다
    return sorted(out, key=lambda n: STATES[n].label_index)


def grpc_color(name):
    """
    상태 이름 -> gRPC TrafficLightColor enum 값.

    grpc_inha_univ 의 proto 를 import 할 수 있는 sys.path 가 잡혀 있어야 한다.
    실패하면 morai_value 로 폴백한다 (값이 동일하므로 안전하다).
    """
    st = get(name)
    if st.morai_value is None:
        raise ValueError("'%s' 는 연출 대상이 아니다." % name)
    try:
        from proto.morai.infrastructure import infrastructure_enum_pb2 as ie

        value = getattr(ie, st.grpc_enum)
        if value != st.morai_value:
            raise RuntimeError(
                "MORAI enum 값이 이 테이블과 다르다: %s=%d (테이블 %d). "
                "tl/states.py 를 시뮬레이터 버전에 맞게 갱신할 것."
                % (st.grpc_enum, value, st.morai_value)
            )
        return value
    except ImportError:
        return st.morai_value


def summary_table():
    """사람이 읽는 요약표 문자열."""
    lines = ["idx  name         morai  필요램프  desc"]
    for name in CLASS_NAMES:
        s = STATES[name]
        mv = "-" if s.morai_value is None else str(s.morai_value)
        lines.append(
            "%3d  %-12s %5s  %-8s  %s"
            % (s.label_index, s.name, mv, s.requires or "-", s.desc)
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary_table())
    print()
    for hl in (False, True):
        st = states_for(hl)
        print("has_left=%-5s → %d종: %s" % (hl, len(st), ", ".join(st)))
