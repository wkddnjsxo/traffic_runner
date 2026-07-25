#!/usr/bin/env bash
# 학습 컨테이너 빌드 & 실행 헬퍼.
#
#   ./run.sh build            이미지 빌드
#   ./run.sh check            GPU 가 제대로 잡히는지 확인 (제일 먼저 이것부터)
#   ./run.sh train [옵션...]  학습 실행
#   ./run.sh checkdata        데이터셋 무결성 검사 (학습 전에)
#   ./run.sh serve --ckpt X   추론 서버 (실시간 인지용, 포트 5555)
#   ./run.sh infer [옵션...]  추론 (--ckpt 필수)
#   ./run.sh shell            컨테이너 안으로 들어가기
#
# 워크스페이스 전체(~/traffic_runner)를 /workspace 로 마운트한다.
set -e

IMAGE=tl-train
WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMD="${1:-train}"; shift || true

if ! command -v docker >/dev/null 2>&1; then
  echo "docker 가 없다. Docker Desktop 설정에서 이 WSL 배포판의"
  echo "Resources > WSL Integration 을 켤 것."
  exit 1
fi

DOCKER_RUN=(docker run --rm --gpus all
  --shm-size=8g                       # DataLoader 워커가 공유메모리를 쓴다
  -v "$WS_ROOT:/workspace"
  -w /workspace)

# 터미널에서 실행할 때만 -it 를 붙인다. 파이프/heredoc 으로 stdin 을 주면서
# -t 를 붙이면 "cannot attach stdin to a TTY-enabled container" 로 죽는다.
TTY=()
[ -t 0 ] && [ -t 1 ] && TTY=(-it)

case "$CMD" in
  build)
    docker build -t "$IMAGE" "$WS_ROOT/train"
    ;;
  check)
    # stdin 으로 스크립트를 넣으므로 -t 를 쓰면 안 된다
    # ("cannot attach stdin to a TTY-enabled container")
    "${DOCKER_RUN[@]}" -i "$IMAGE" python - <<'PY'
import torch
print("torch      :", torch.__version__)
print("torch cuda :", torch.version.cuda)
print("available  :", torch.cuda.is_available())
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print("device     :", torch.cuda.get_device_name(0))
    print("capability : sm_%d%d" % cap, "(RTX 5070 은 sm_120)")
    x = torch.randn(4096, 4096, device="cuda")
    print("matmul     :", float((x @ x).sum()), "→ 커널 정상")
else:
    print("GPU 미검출: --gpus all, nvidia-container-toolkit, 드라이버를 확인할 것")
PY
    ;;
  train)
    "${DOCKER_RUN[@]}" "${TTY[@]}" "$IMAGE" python train/train.py "$@"
    ;;
  checkdata)
    "${DOCKER_RUN[@]}" "${TTY[@]}" "$IMAGE" python train/check_dataset.py "$@"
    ;;
  analyze)
    "${DOCKER_RUN[@]}" "${TTY[@]}" "$IMAGE" python train/analyze_val.py "$@"
    ;;
  serve)
    # 추론 서버. 포트를 호스트로 열어 WSL 의 live_infer.py 가 붙는다.
    "${DOCKER_RUN[@]}" "${TTY[@]}" -p "${TR_PORT:-5555}:5555" "$IMAGE" \
        python train/serve.py --port 5555 "$@"
    ;;
  infer)
    "${DOCKER_RUN[@]}" "${TTY[@]}" "$IMAGE" python train/infer.py "$@"
    ;;
  shell)
    "${DOCKER_RUN[@]}" -it "$IMAGE" bash
    ;;
  *)
    echo "사용: $0 {build|check|checkdata|train|serve|infer|analyze|shell}"; exit 1;;
esac
