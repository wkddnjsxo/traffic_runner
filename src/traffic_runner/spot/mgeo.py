"""
MGeo(link_set/node_set) 그래프 → 지점(spot) 자동 생성.

MGeo 는 시뮬레이터와 정확히 일치한다 (실측: 주행 중 ego 가 링크 폴리라인에서 0.15m 이내,
ego rotation.z 와 링크 진행방향 atan2(dy,dx) 의 차이가 정상 주행 시 3~5°).
따라서 링크 폴리라인을 그대로 시작점·끝점·주행경로로 쓸 수 있다.

구조 (실측 확인):
  - link_type '6' : 일반 도로 링크. 교차로 진입로가 여기 해당.
  - link_type '1' : 교차로 내부 링크. related_signal 로 movement 종류를 표시한다
                    ('straight' / 'left' / 'left_unprotected' / 'right_unprotected' / 'uturn_normal').
  - node.traffic_light_id : 그 노드(정지선)를 관장하는 신호등 ID.
  - link.points 는 항상 from_node -> to_node 순서다.

신호등 접근로 = to_node 에 traffic_light_id 가 있는 링크. K-city 2025 기준 75개
(고유 신호등 54개).

좌회전 화살표 유무(has_left)는 정지선 노드에서 나가는 movement 로 판정한다.
  'left'             = 보호 좌회전 -> 좌회전 화살표 신호등이 있다  -> has_left=True
  'left_unprotected' = 비보호 좌회전 -> 화살표 램프가 없다        -> has_left=False
"""

import collections
import json
import math
import os


ROAD_LINK = "6"
INTERSECTION_LINK = "1"


class MGeo(object):
    def __init__(self, mgeo_dir):
        self.dir = mgeo_dir
        with open(os.path.join(mgeo_dir, "link_set.json")) as f:
            links = json.load(f)
        with open(os.path.join(mgeo_dir, "node_set.json")) as f:
            nodes = json.load(f)

        self.links = {l["idx"]: l for l in links}
        self.nodes = {n["idx"]: n for n in nodes}

        self.out_links = collections.defaultdict(list)   # node -> 나가는 링크들
        self.in_links = collections.defaultdict(list)    # node -> 들어오는 링크들
        for l in links:
            self.out_links[l["from_node_idx"]].append(l)
            self.in_links[l["to_node_idx"]].append(l)

    # ------------------------------------------------------------- 조회
    def approach_links(self):
        """신호등 정지선으로 끝나는 링크(=신호교차로 진입로) 목록."""
        out = []
        for l in self.links.values():
            node = self.nodes.get(l["to_node_idx"])
            if node and node.get("traffic_light_id"):
                out.append(l)
        return out

    def movements_at(self, node_idx):
        """정지선 노드에서 나가는 movement 종류 집합."""
        sigs = set()
        for l in self.out_links.get(node_idx, []):
            s = l.get("related_signal")
            if s:
                sigs.add(s)
        return sigs

    def has_left_arrow(self, node_idx):
        """보호 좌회전(=화살표 램프)이 있는 교차로 진입로인가."""
        return "left" in self.movements_at(node_idx)

    # ------------------------------------------------------- 경로 만들기
    def _heading_at_end(self, link):
        """링크 끝부분 진행방향(rad)."""
        pts = link["points"]
        if len(pts) < 2:
            return 0.0
        a, b = pts[-2], pts[-1]
        return math.atan2(b[1] - a[1], b[0] - a[0])

    def _heading_at_start(self, link):
        pts = link["points"]
        if len(pts) < 2:
            return 0.0
        a, b = pts[0], pts[1]
        return math.atan2(b[1] - a[1], b[0] - a[0])

    def _best_predecessor(self, link, visited):
        """
        진행방향이 가장 자연스럽게 이어지는 선행 링크를 고른다.

        분기점에서 아무거나 고르면 경로가 옆길로 새서 신호등이 화면에서 사라진다.
        그래서 '들어오는 링크의 끝 방향' 과 '현재 링크의 시작 방향' 차이가 최소인 것을 쓴다.
        교차로 내부 링크(type 1)는 감점해서, 가능하면 직선 도로를 따라 뒤로 뻗게 한다.
        """
        cands = [p for p in self.in_links.get(link["from_node_idx"], [])
                 if p["idx"] not in visited]
        if not cands:
            return None

        target = self._heading_at_start(link)

        def cost(p):
            d = abs(_norm_rad(self._heading_at_end(p) - target))
            if p["link_type"] == INTERSECTION_LINK:
                d += math.radians(45)  # 교차로 내부 링크는 후순위
            return d

        best = min(cands, key=cost)
        # 90° 이상 꺾이면 이어붙이지 않는다 (다른 도로로 새는 것)
        if abs(_norm_rad(self._heading_at_end(best) - target)) > math.radians(90):
            return None
        return best

    def build_approach_path(self, approach_link, approach_len_m, end_offset_m):
        """
        정지선에서 뒤로 approach_len_m 만큼 거슬러 올라간 경로를 만든다.

        반환: (points, meta)
          points : [(x, y, z), ...] 시작점 -> 끝점 순서
          meta   : {'links': [...], 'truncated': bool}

        end_offset_m 만큼 정지선 앞에서 끊는다 (교차로 안으로 진입하지 않게).
        """
        chain = [approach_link]
        visited = {approach_link["idx"]}
        total = _polyline_len(approach_link["points"])

        while total < approach_len_m + end_offset_m:
            prev = self._best_predecessor(chain[0], visited)
            if prev is None:
                break
            chain.insert(0, prev)
            visited.add(prev["idx"])
            total += _polyline_len(prev["points"])

        # 링크 폴리라인 이어붙이기 (이음매 중복점 제거)
        pts = []
        for l in chain:
            lp = l["points"]
            if pts and _dist2(pts[-1], lp[0]) < 0.01:
                lp = lp[1:]
            pts.extend([(p[0], p[1], p[2]) for p in lp])

        # 뒤에서 end_offset_m 잘라내기 (정지선 앞에서 멈춤)
        if end_offset_m > 0:
            pts = _trim_end(pts, end_offset_m)
        # 앞에서 잘라 총 길이를 approach_len_m 로 맞추기
        truncated = _polyline_len(pts) < approach_len_m - 1.0
        pts = _trim_front_to_length(pts, approach_len_m)

        return pts, {"links": [l["idx"] for l in chain], "truncated": truncated}

    def _best_successor(self, link, visited):
        """진행방향이 가장 자연스럽게 이어지는 후속 링크 (직진 우선)."""
        cands = [n for n in self.out_links.get(link["to_node_idx"], [])
                 if n["idx"] not in visited]
        if not cands:
            return None
        target = self._heading_at_end(link)
        lane = link.get("ego_lane")

        def cost(n):
            d = abs(_norm_rad(self._heading_at_start(n) - target))
            if n.get("related_signal") in ("left", "left_unprotected", "uturn_normal"):
                d += math.radians(60)   # 좌회전/유턴으로 새지 않게
            # 같은 차선을 유지한다. 객체를 '끝점과 같은 차선' 에 놓으려면
            # 연장 폴리라인이 그 차선 중심선이어야 한다.
            if lane is not None and n.get("ego_lane") != lane:
                d += math.radians(30)
            return d

        best = min(cands, key=cost)
        if abs(_norm_rad(self._heading_at_start(best) - target)) > math.radians(70):
            return None
        return best

    def extend_beyond(self, from_link_id, from_xy, length_m):
        """
        끝점 이후로 도로를 계속 따라간 폴리라인을 만든다.

        객체를 '지점 구간 밖, 끝점 너머'에 놓기 위해 쓴다. ego 는 여기까지 오지
        않으므로 충돌하지 않지만, 카메라에는 정면으로 잡힌다.

        반환 (points, meta). points[0] 은 from_xy 에 가장 가까운 지점.
        """
        link = self.links.get(from_link_id)
        if link is None:
            link, _ = self.nearest_link(from_xy[0], from_xy[1])
            if link is None:
                return [], {"links": [], "error": "시작 링크를 못 찾음"}

        pr = self.project_on_link(link, from_xy[0], from_xy[1])
        pts = self._sub_polyline(link, pr["s"] if pr else 0.0, None)
        chain = [link["idx"]]
        visited = {link["idx"]}

        while _polyline_len(pts) < length_m:
            nxt = self._best_successor(self.links[chain[-1]], visited)
            if nxt is None:
                break
            pts = _append_polyline(pts, [(p[0], p[1], p[2]) for p in nxt["points"]])
            chain.append(nxt["idx"])
            visited.add(nxt["idx"])

        truncated = _polyline_len(pts) < length_m - 1.0
        return pts, {"links": chain, "truncated": truncated,
                     "available_m": _polyline_len(pts)}

    def find_opposite_link(self, x, y, heading_deg, max_dist=25.0,
                           min_angle_deg=140.0):
        """
        (x,y) 근처에서 진행방향이 반대인 도로 링크를 찾는다 (= 반대차선).

        MGeo 의 opp_traffic / driving_direction 필드는 이 맵에서 전부 비어 있어
        쓸 수 없다(실측). 그래서 기하학적으로 찾는다: 근처 도로 링크 중
        진행방향이 min_angle_deg 이상 어긋난 것.

        반환 (link, projection) 또는 (None, None).
        """
        best_l, best_p = None, None
        for l in self.links.values():
            if l["link_type"] != ROAD_LINK:
                continue
            pr = self.project_on_link(l, x, y)
            if pr is None or pr["dist"] > max_dist:
                continue
            h = self.link_heading_at(l, pr["seg"])
            if abs(_norm_deg(h - heading_deg)) < min_angle_deg:
                continue
            if best_p is None or pr["dist"] < best_p["dist"]:
                pr["heading_deg"] = h
                best_l, best_p = l, pr
        return best_l, best_p

    def opposite_polyline(self, from_xy, heading_deg, probe_ahead_m=15.0,
                          length_m=60.0, max_dist=25.0):
        """
        끝점 정면 부근의 반대차선 폴리라인을 준다.

        반대차선은 ego 쪽으로 흘러오므로, 링크 진행방향을 그대로 쓰면 폴리라인이
        ego 를 향해 온다. 객체는 그 위에 놓고 링크 방향을 향하게 하면 마주오는
        차가 된다.
        """
        yaw = math.radians(heading_deg)
        px = from_xy[0] + math.cos(yaw) * probe_ahead_m
        py = from_xy[1] + math.sin(yaw) * probe_ahead_m

        link, pr = self.find_opposite_link(px, py, heading_deg, max_dist=max_dist)
        if link is None:
            return [], {"error": "반대차선을 못 찾음"}

        pts = [(p[0], p[1], p[2]) for p in link["points"]]
        chain = [link["idx"]]
        # 짧으면 선행 링크를 붙여 앞쪽(ego 에서 먼 쪽)을 늘린다
        visited = {link["idx"]}
        while _polyline_len(pts) < length_m:
            prev = self._best_predecessor(self.links[chain[0]], visited)
            if prev is None:
                break
            pts = _append_polyline([(p[0], p[1], p[2]) for p in prev["points"]], pts)
            chain.insert(0, prev["idx"])
            visited.add(prev["idx"])

        return pts, {"links": chain, "lateral_m": pr["dist"]}

    def link_of_point(self, x, y):
        """(x,y) 에 가장 가까운 링크와 거리."""
        best, bestd = None, float("inf")
        for l in self.links.values():
            for p in l["points"]:
                d = math.hypot(p[0] - x, p[1] - y)
                if d < bestd:
                    bestd, best = d, l
        return best, bestd

    # ------------------------------------------------------- 두 점 라우팅
    def project_on_link(self, link, x, y):
        """
        점을 링크 폴리라인에 수직 투영한다.

        반환: {'seg': 세그먼트 인덱스, 't': 세그먼트 내 비율, 'dist': 수직거리,
               'point': 투영점, 's': 링크 시작부터의 거리}
        """
        pts = link["points"]
        best = None
        s_acc = 0.0
        for i in range(len(pts) - 1):
            ax, ay = pts[i][0], pts[i][1]
            bx, by = pts[i + 1][0], pts[i + 1][1]
            dx, dy = bx - ax, by - ay
            seg2 = dx * dx + dy * dy
            if seg2 < 1e-12:
                continue
            t = ((x - ax) * dx + (y - ay) * dy) / seg2
            t = max(0.0, min(1.0, t))
            px, py = ax + dx * t, ay + dy * t
            d = math.hypot(x - px, y - py)
            if best is None or d < best["dist"]:
                seg_len = math.sqrt(seg2)
                pz = pts[i][2] + (pts[i + 1][2] - pts[i][2]) * t
                best = {"seg": i, "t": t, "dist": d, "point": (px, py, pz),
                        "s": s_acc + seg_len * t}
            s_acc += math.hypot(dx, dy)
        return best

    def link_heading_at(self, link, seg_idx):
        """링크의 특정 세그먼트 진행방향(도)."""
        pts = link["points"]
        i = max(0, min(seg_idx, len(pts) - 2))
        return math.degrees(math.atan2(pts[i + 1][1] - pts[i][1],
                                       pts[i + 1][0] - pts[i][0]))

    def nearest_link(self, x, y, max_dist=15.0, prefer_road=True, road_tolerance_m=3.0,
                     heading_hint_deg=None, heading_tol_deg=100.0):
        """
        (x,y) 를 도로망에 스냅한다. 반환 (link, projection) 또는 (None, None).

        두 가지 선호를 건다.

        1) 진행방향(heading_hint_deg): 교차로 안에서는 여러 링크가 겹쳐 지나가므로
           '가장 가까운' 링크가 엉뚱하게 교차하는 방향의 링크일 수 있다. 가려는
           방향과 heading_tol_deg 이상 어긋난 링크는 먼저 걸러낸다.
        2) 일반 도로 링크(type 6) > 교차로 내부 링크(type 1). 시작점을 교차로
           한복판에 잡는 건 대개 실수이기 때문이다.

        단, 선호 때문에 점을 엉뚱하게 멀리 끌고 가면 안 되므로, 선호 대상이
        최근접 링크보다 road_tolerance_m 이상 멀면 최근접 쪽을 그대로 쓴다.
        """
        cands = []
        for l in self.links.values():
            pr = self.project_on_link(l, x, y)
            if pr is None or pr["dist"] > max_dist:
                continue
            pr["heading_deg"] = self.link_heading_at(l, pr["seg"])
            cands.append((l, pr))
        if not cands:
            return None, None

        def pick(pool):
            if not pool:
                return None, None
            nearest = min(pool, key=lambda t: t[1]["dist"])
            if prefer_road:
                roads = [t for t in pool if t[0]["link_type"] != INTERSECTION_LINK]
                if roads:
                    best_road = min(roads, key=lambda t: t[1]["dist"])
                    if best_road[1]["dist"] <= nearest[1]["dist"] + road_tolerance_m:
                        return best_road
            return nearest

        pool = cands
        if heading_hint_deg is not None:
            aligned = [t for t in cands
                       if abs(_norm_deg(t[1]["heading_deg"] - heading_hint_deg))
                       <= heading_tol_deg]
            if aligned:
                pool = aligned

        chosen_l, chosen = pick(pool)
        if chosen is None:
            return None, None

        chosen["link_type"] = chosen_l["link_type"]
        chosen["snapped_to_intersection"] = (chosen_l["link_type"] == INTERSECTION_LINK)
        return chosen_l, chosen

    def _sub_polyline(self, link, s_from=None, s_to=None):
        """링크 폴리라인에서 [s_from, s_to] 구간만 잘라낸다 (링크 시작 기준 거리)."""
        pts = [(p[0], p[1], p[2]) for p in link["points"]]
        if s_from is not None and s_from > 0:
            pts = _trim_front(pts, s_from)
        if s_to is not None:
            total = _polyline_len([(p[0], p[1], p[2]) for p in link["points"]])
            cut_from_end = total - s_to - (s_from or 0.0) if s_from else total - s_to
            if cut_from_end > 0:
                pts = _trim_end(pts, cut_from_end)
        return pts

    def route_between(self, start_xy, end_xy, max_snap_dist=15.0):
        """
        두 점 사이를 도로망을 따라 잇는 경로를 만든다 (Dijkstra).

        시작점/끝점을 각각 가장 가까운 링크에 스냅하고, 그 사이를 링크 그래프로
        최단 경로 탐색한다. 직선으로 잇지 않는 이유는 명백하다 — 곡선 도로에서
        차선을 벗어나고, pure pursuit 이 따라갈 수 없다.

        반환: (points, meta). 실패 시 (None, {'error': ...}).
        """
        import heapq

        # 시작→끝 방위를 진행방향 힌트로 준다. 교차로처럼 여러 링크가 겹치는 곳에서
        # 엉뚱하게 교차하는 방향의 링크로 스냅되는 것을 막는다.
        bearing = math.degrees(math.atan2(end_xy[1] - start_xy[1],
                                          end_xy[0] - start_xy[0]))
        sl, sp = self.nearest_link(start_xy[0], start_xy[1], max_snap_dist,
                                   heading_hint_deg=bearing)
        el, ep = self.nearest_link(end_xy[0], end_xy[1], max_snap_dist,
                                   heading_hint_deg=bearing)
        if sl is None:
            return None, {"error": "시작점이 도로에서 %.0fm 이상 떨어져 있다" % max_snap_dist}
        if el is None:
            return None, {"error": "끝점이 도로에서 %.0fm 이상 떨어져 있다" % max_snap_dist}

        # 같은 링크 위의 두 점
        if sl["idx"] == el["idx"]:
            if ep["s"] < sp["s"]:
                return None, {"error": "끝점이 시작점보다 링크 진행방향에서 뒤에 있다 "
                                       "(링크 %s). 진행방향을 확인할 것." % sl["idx"]}
            pts = self._sub_polyline(sl, sp["s"], ep["s"])
            return pts, {"links": [sl["idx"]], "snap_start_m": sp["dist"],
                         "snap_end_m": ep["dist"]}

        # 시작 링크의 to_node 에서 끝 링크의 from_node 까지 Dijkstra
        src, dst = sl["to_node_idx"], el["from_node_idx"]
        dist = {src: 0.0}
        prev = {}
        pq = [(0.0, src)]
        seen = set()
        while pq:
            d, node = heapq.heappop(pq)
            if node in seen:
                continue
            seen.add(node)
            if node == dst:
                break
            for l in self.out_links.get(node, []):
                nxt = l["to_node_idx"]
                nd = d + float(l.get("link_length", 0.0) or _polyline_len(
                    [(p[0], p[1], p[2]) for p in l["points"]]))
                if nd < dist.get(nxt, float("inf")):
                    dist[nxt] = nd
                    prev[nxt] = (node, l)
                    heapq.heappush(pq, (nd, nxt))

        if dst not in dist:
            return None, {"error": "두 점을 잇는 경로를 못 찾았다 "
                                   "(일방통행 방향이 반대이거나 도로망이 끊겨 있다). "
                                   "시작 링크=%s 끝 링크=%s" % (sl["idx"], el["idx"])}

        # 경로 복원
        chain = []
        node = dst
        while node != src:
            pnode, plink = prev[node]
            chain.append(plink)
            node = pnode
        chain.reverse()

        # 폴리라인 조립: 시작링크 후반부 + 중간링크 전체 + 끝링크 전반부
        pts = list(self._sub_polyline(sl, sp["s"], None))
        for l in chain:
            pts = _append_polyline(pts, [(p[0], p[1], p[2]) for p in l["points"]])
        pts = _append_polyline(pts, self._sub_polyline(el, None, ep["s"]))

        return pts, {
            "links": [sl["idx"]] + [l["idx"] for l in chain] + [el["idx"]],
            "snap_start_m": sp["dist"],
            "snap_end_m": ep["dist"],
        }

    def traffic_lights_along(self, link_ids):
        """경로가 지나는 링크들의 정지선 신호등 ID 목록 (경로 순서대로, 중복 제거)."""
        out = []
        for lid in link_ids:
            link = self.links.get(lid)
            if not link:
                continue
            node = self.nodes.get(link["to_node_idx"])
            tl = node.get("traffic_light_id") if node else None
            if tl and tl not in out:
                out.append(tl)
        return out

    def distance_to_nearest_tl(self, x, y):
        """가장 가까운 신호등 노드까지의 거리."""
        best = float("inf")
        for n in self.nodes.values():
            if not n.get("traffic_light_id"):
                continue
            p = n["point"]
            best = min(best, math.hypot(p[0] - x, p[1] - y))
        return best


# --------------------------------------------------------------- 기하 helper
def _norm_deg(a):
    """각도를 (-180, 180] 으로."""
    a = math.fmod(a + 180.0, 360.0)
    if a <= 0:
        a += 360.0
    return a - 180.0


def _norm_rad(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def _dist2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _polyline_len(pts):
    return sum(_dist2(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def _trim_end(pts, cut_m):
    """끝에서 cut_m 만큼 잘라낸다."""
    if cut_m <= 0 or len(pts) < 2:
        return pts
    acc = 0.0
    for i in range(len(pts) - 1, 0, -1):
        seg = _dist2(pts[i - 1], pts[i])
        if acc + seg >= cut_m:
            r = (cut_m - acc) / seg
            x = pts[i][0] + (pts[i - 1][0] - pts[i][0]) * r
            y = pts[i][1] + (pts[i - 1][1] - pts[i][1]) * r
            z = pts[i][2] + (pts[i - 1][2] - pts[i][2]) * r
            return pts[:i] + [(x, y, z)]
        acc += seg
    return pts[:2]


def _trim_front(pts, cut_m):
    """앞에서 cut_m 만큼 잘라낸다."""
    if cut_m <= 0 or len(pts) < 2:
        return pts
    acc = 0.0
    for i in range(len(pts) - 1):
        seg = _dist2(pts[i], pts[i + 1])
        if acc + seg >= cut_m:
            r = (cut_m - acc) / seg
            x = pts[i][0] + (pts[i + 1][0] - pts[i][0]) * r
            y = pts[i][1] + (pts[i + 1][1] - pts[i][1]) * r
            z = pts[i][2] + (pts[i + 1][2] - pts[i][2]) * r
            return [(x, y, z)] + pts[i + 1:]
        acc += seg
    return pts[-2:]


def _append_polyline(base, extra):
    """폴리라인 이어붙이기. 이음매 중복점은 버린다."""
    if not base:
        return list(extra)
    if extra and _dist2(base[-1], extra[0]) < 0.01:
        extra = extra[1:]
    return base + list(extra)


def _trim_front_to_length(pts, length_m):
    """뒤(끝점)를 기준으로 length_m 만큼만 남기고 앞을 잘라낸다."""
    total = _polyline_len(pts)
    if total <= length_m or len(pts) < 2:
        return pts
    cut = total - length_m
    acc = 0.0
    for i in range(len(pts) - 1):
        seg = _dist2(pts[i], pts[i + 1])
        if acc + seg >= cut:
            r = (cut - acc) / seg
            x = pts[i][0] + (pts[i + 1][0] - pts[i][0]) * r
            y = pts[i][1] + (pts[i + 1][1] - pts[i][1]) * r
            z = pts[i][2] + (pts[i + 1][2] - pts[i][2]) * r
            return [(x, y, z)] + pts[i + 1:]
        acc += seg
    return pts[-2:]


def with_yaw(pts):
    """
    [(x,y,z)] -> [(x,y,z,yaw_deg)].

    yaw 는 다음 점을 향하는 방향. MORAI rotation.z 와 같은 규약임을 실측 확인했다.
    """
    out = []
    for i, p in enumerate(pts):
        # 각 점의 yaw = 다음 점을 향하는 방향. 인접 점이 겹쳐 dx=dy=0 이면
        # 더 멀리 있는 점까지 훑어 실제 진행방향을 찾는다. 안 그러면 시작점
        # yaw 가 0 으로 떨어져(겹친 점) 주행 시작 순간 차가 90° 꺾인다.
        yaw = None
        for j in range(i + 1, len(pts)):
            dx = pts[j][0] - pts[i][0]
            dy = pts[j][1] - pts[i][1]
            if dx or dy:
                yaw = math.degrees(math.atan2(dy, dx))
                break
        if yaw is None:  # 마지막 점: 이전 점에서의 방향을 물려받는다
            yaw = out[-1][3] if out else 0.0
        out.append((p[0], p[1], p[2], yaw))
    return out
