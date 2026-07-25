"""경로/각도 관련 작은 계산들."""

import math


def dist2d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def path_length(points):
    """points: [(x,y,z,yaw), ...] 의 2D 누적 길이(m)."""
    total = 0.0
    for i in range(len(points) - 1):
        total += dist2d(points[i], points[i + 1])
    return total


def normalize_deg(a):
    """각도를 (-180, 180] 으로."""
    a = math.fmod(a + 180.0, 360.0)
    if a <= 0:
        a += 360.0
    return a - 180.0


def angle_diff_deg(a, b):
    """a - b 의 최소 각도차(도)."""
    return normalize_deg(a - b)


def resample(points, interval_m):
    """
    폴리라인을 등간격으로 재샘플링한다. yaw 는 최근접 원본 점의 값을 쓴다.

    캡처 궤적은 사람이 몬 것이라 속도에 따라 점 간격이 들쭉날쭉하다.
    pure pursuit 의 lookahead 탐색을 안정시키려면 등간격이 낫다.
    """
    if len(points) < 2 or interval_m <= 0:
        return list(points)

    # (x,y,z) 3-tuple 과 (x,y,z,yaw) 4-tuple 을 모두 받는다.
    has_yaw = len(points[0]) >= 4

    out = [tuple(points[0])]
    carry = 0.0
    for i in range(len(points) - 1):
        p, q = points[i], points[i + 1]
        seg = dist2d(p, q)
        if seg <= 1e-9:
            continue
        t = interval_m - carry
        while t <= seg:
            r = t / seg
            pt = (
                p[0] + (q[0] - p[0]) * r,
                p[1] + (q[1] - p[1]) * r,
                p[2] + (q[2] - p[2]) * r,
            )
            if has_yaw:
                pt += (q[3] if r > 0.5 else p[3],)
            out.append(pt)
            t += interval_m
        carry = seg - (t - interval_m)
    last = tuple(points[-1])
    if dist2d(out[-1], last) > interval_m * 0.25:
        out.append(last)
    return out
