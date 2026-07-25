"""
Pure pursuit 경로 추종.

지점(spot)의 path 를 따라 시작점에서 끝점까지 주행한다.

lookahead 거리는 속도에 비례해서 잡는다(Ld = k*v + Ld0). 고정값을 쓰면
저속에서 흔들리고 고속에서 코너를 자른다.

경로 인덱스는 앞으로만 진행시킨다. 전체를 매번 재탐색하면 U 자 경로나 자기교차
구간에서 엉뚱한 점으로 튄다.
"""

import math


class PurePursuit(object):
    def __init__(self, path, wheelbase=2.7, lookahead_k=0.6, lookahead_min=4.0,
                 lookahead_max=15.0, max_steer_deg=35.0):
        if len(path) < 2:
            raise ValueError("path 가 너무 짧다 (%d 점)" % len(path))
        self.path = [(float(p[0]), float(p[1])) for p in path]
        self.wheelbase = float(wheelbase)
        self.k = float(lookahead_k)
        self.ld_min = float(lookahead_min)
        self.ld_max = float(lookahead_max)
        self.max_steer = math.radians(max_steer_deg)

        self.idx = 0                 # 앞으로만 전진하는 경로 인덱스
        self.cum = _cumulative(self.path)
        self.total_length = self.cum[-1]

    # ------------------------------------------------------------------ 조회
    def nearest_index(self, x, y, window=200):
        """현재 인덱스 이후 window 개 안에서 가장 가까운 점을 찾는다 (전진만)."""
        best_i, best_d = self.idx, float("inf")
        end = min(len(self.path), self.idx + window)
        for i in range(self.idx, end):
            px, py = self.path[i]
            d = (px - x) ** 2 + (py - y) ** 2
            if d < best_d:
                best_d, best_i = d, i
        self.idx = best_i
        return best_i, math.sqrt(best_d)

    def lookahead_distance(self, speed_mps):
        return max(self.ld_min, min(self.ld_max, self.k * speed_mps + self.ld_min))

    def target_point(self, x, y, speed_mps):
        """lookahead 거리만큼 앞선 경로점."""
        i, _ = self.nearest_index(x, y)
        ld = self.lookahead_distance(speed_mps)
        target_s = self.cum[i] + ld
        j = i
        while j < len(self.path) - 1 and self.cum[j] < target_s:
            j += 1
        return j, self.path[j]

    # ------------------------------------------------------------------ 제어
    def step(self, x, y, yaw_deg, speed_mps):
        """
        반환 (steer_rad, info).

        info: {'target_idx', 'cross_track_m', 'remain_m', 'progress'}
        """
        i, cte = self.nearest_index(x, y)
        j, (tx, ty) = self.target_point(x, y, speed_mps)

        yaw = math.radians(yaw_deg)
        # 차량 좌표계로 변환 (전방 +x, 좌측 +y)
        dx, dy = tx - x, ty - y
        local_x = math.cos(-yaw) * dx - math.sin(-yaw) * dy
        local_y = math.sin(-yaw) * dx + math.cos(-yaw) * dy

        ld = math.hypot(local_x, local_y)
        if ld < 1e-3:
            steer = 0.0
        else:
            # pure pursuit: delta = atan(2 L sin(alpha) / Ld), sin(alpha) = local_y/Ld
            steer = math.atan2(2.0 * self.wheelbase * local_y, ld * ld)
        steer = max(-self.max_steer, min(self.max_steer, steer))

        remain = max(0.0, self.total_length - self.cum[i])
        return steer, {
            "target_idx": j,
            "nearest_idx": i,
            "cross_track_m": cte,
            "remain_m": remain,
            "progress": (self.cum[i] / self.total_length) if self.total_length else 1.0,
        }

    def speed_for(self, steer_rad, base_speed_mps, min_speed_mps=2.0):
        """
        조향이 클수록 감속한다. 코너에서 전속으로 밀면 경로를 벗어난다.
        """
        ratio = abs(steer_rad) / self.max_steer if self.max_steer else 0.0
        return max(min_speed_mps, base_speed_mps * (1.0 - 0.7 * ratio))

    def distance_to_end(self, x, y):
        return math.hypot(self.path[-1][0] - x, self.path[-1][1] - y)


def _cumulative(pts):
    cum = [0.0]
    for i in range(len(pts) - 1):
        cum.append(cum[-1] + math.hypot(pts[i + 1][0] - pts[i][0],
                                        pts[i + 1][1] - pts[i][1]))
    return cum
