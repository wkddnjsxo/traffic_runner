# 신호등 인지 ResNet18 학습 컨테이너 (RTX 5070 / Blackwell)
#
# ★ RTX 5070 은 Blackwell(sm_120) 이라 CUDA 12.8 이상이 필요하다.
#   `pip install torch` 로 받는 기본 휠은 cu126/cu124 라 sm_120 커널이 없어서
#   "no kernel image is available for execution on the device" 로 죽거나
#   조용히 CPU 로 떨어진다. 반드시 아래 cu128 인덱스를 써야 한다.

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/local/bin/python

# ★ cu128 인덱스 — 이 줄이 핵심이다
RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# ImageNet 사전학습 가중치를 이미지에 미리 받아둔다.
# 안 그러면 오프라인 컨테이너 첫 실행 때 다운로드가 실패해 랜덤 초기화로 떨어진다.
RUN python - <<'PY'
from torchvision import models
models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
print("resnet18 ImageNet 가중치 캐시 완료")
PY

WORKDIR /workspace

# 데이터/코드는 마운트해서 쓴다 (이미지에 굽지 않는다).
# 워크스페이스 전체를 /workspace 로 마운트하면
#   /workspace/dataset      수집 결과
#   /workspace/train        학습 코드
# 가 그대로 보인다.
ENV TR_DATA=/workspace/dataset \
    TR_OUT=/workspace/train/runs

CMD ["python", "train/train.py"]
