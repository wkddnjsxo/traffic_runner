#!/usr/bin/env bash
# 좌회전(left) 계열 보강 수집.
#
#   ./collect_left.sh                    # 기본 지점 (sig_007,sig_008,sig_009)
#   ./collect_left.sh sig_007,sig_010    # 지점 직접 지정
#
# 두 단계를 순서대로 돌린다.
#   1단계: left / red_left / green_left  ← 본 목적. left 계열 데이터를 늘린다.
#   2단계: red / yellow / green / red_yellow 를 seed 0 에서만
#          ← 1단계만 하면 "이 교차로 배경 = left 계열" 이라는 지름길이 생긴다.
#            모델이 신호등 대신 배경으로 답을 좁혀버리므로, 같은 지점에서 다른
#            신호도 조금 섞어 그 지름길을 막는다. seed 0 만 쓰므로 비용은 1/3.
#
# 중단(Ctrl-C)해도 dataset/progress.txt 에 남아, 다시 실행하면 이어서 한다.
set -e

WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$WS_ROOT/src/traffic_runner"
SPOTS="${1:-sig_007,sig_008,sig_009}"

[ -f /opt/ros/noetic/setup.bash ] && source /opt/ros/noetic/setup.bash
[ -f "$WS_ROOT/devel/setup.bash" ] && source "$WS_ROOT/devel/setup.bash"
cd "$PKG"

echo "=========================================================="
echo " 좌회전 보강 수집 — 지점: $SPOTS"
echo "=========================================================="

echo
echo "### 1/2 단계: left 계열 3종 ###"
python3 tools/collect.py --spots "$SPOTS" \
    --states left,red_left,green_left "${@:2}"

echo
echo "### 2/2 단계: 나머지 4종 (seed 0 만, 지름길 차단용) ###"
python3 tools/collect.py --spots "$SPOTS" --seeds 0 \
    --states red,yellow,green,red_yellow "${@:2}"

echo
echo "=========================================================="
echo " 완료. 학습:  cd $WS_ROOT/train && ./run.sh train --epochs 5"
echo "=========================================================="
