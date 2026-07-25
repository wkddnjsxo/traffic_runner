#!/usr/bin/env bash
# 지점(시작점/끝점/경로) 캡처 툴 실행.
#
#   ./capture.sh                    # 기본 (gRPC attach)
#   ./capture.sh --pose-source ros  # ROS /Ego_topic 구독
#   ./capture.sh --interval 0.3     # 경로점 간격 0.3m
#
# 사전 준비:
#   - MORAI 실행 + 맵 로드 + gRPC 서버 ON
#   - config/runtime.yaml 의 grpc.host / morai.map_name 확인
#   - (--pose-source ros 인 경우에만) catkin_make + MORAI ROS 브리지 연결
set -e

WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$WS_ROOT/src/traffic_runner"

[ -f /opt/ros/noetic/setup.bash ] && source /opt/ros/noetic/setup.bash
[ -f "$WS_ROOT/devel/setup.bash" ] && source "$WS_ROOT/devel/setup.bash"

cd "$PKG"
exec python3 tools/capture_spot.py "$@"
