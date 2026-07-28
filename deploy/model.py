"""
신호등 인지 ResNet18 — 모델 정의 (추론 전용).

학습 코드(train/)와 이 파일이 유일하게 공유하는 부분만 담았다. pandas·학습 루프
같은 무거운 의존성 없이, 추론에 필요한 것만: 클래스 정의 / 전처리 / 모델 / 체크포인트 로드.
"""

import os

import torch
import torch.nn as nn
from torchvision import models, transforms


#: 학습 시 클래스 인덱스 순서. 체크포인트의 label 순서와 반드시 일치.
CLASS_NAMES = ["red", "yellow", "green", "red_yellow",
               "red_left", "green_left", "left", "unknown"]
NUM_CLASSES = len(CLASS_NAMES)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

#: 확신 없는 예측을 걸러낼 때 쓰는 기권 라벨 (학습 클래스가 아니다).
NOT_DETECTED = "not_detected"


def decide(probs, classes, min_detect=0.40):
    """
    확신 없는 예측을 not_detected 로 기권시킨다.

    unknown 을 제외한 신호 클래스 중 최고 확률이 min_detect **이하**이면
    (= 어떤 신호색도 확실히 못 잡았으면) not_detected 로 판정한다. 제어 입장에서
    "빈 장면"이든 "애매함"이든 지킬 신호가 없다는 점은 같으므로 하나로 묶는다.

    반환: (label, prob)  — label 은 신호색 문자열 또는 "not_detected".
    """
    best_i, best_p = -1, -1.0
    for i, c in enumerate(classes):
        if c == "unknown":
            continue
        p = float(probs[i])
        if p > best_p:
            best_p, best_i = p, i
    if best_i < 0 or best_p <= min_detect:
        return NOT_DETECTED, best_p
    return classes[best_i], best_p


def build_transforms(size, aspect=4.0 / 3.0):
    """
    추론 전처리 — 학습(train_aug=False)과 동일해야 한다.

    ★ 종횡비 보존 ★ 원본 4:3 을 정사각으로 찌그러뜨리면 작은 신호등이 더 뭉개진다.
    (H=size, W=size*4/3) 로 리사이즈. ResNet 은 정사각이 아니어도 된다(adaptive pool).
    """
    h = size
    w = int(round(size * aspect))
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_model():
    """ResNet18 (fc 를 NUM_CLASSES 로 교체). 가중치는 체크포인트에서 로드하므로 pretrained 불필요."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def resolve_ckpt(ckpt_path):
    """
    체크포인트 경로를 찾는다. 'best.pt' 나 'runs/<시각>/best.pt' 처럼 짧게 줘도
    이 파일 기준으로 찾아준다.
    """
    if os.path.isfile(ckpt_path):
        return ckpt_path
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        ckpt_path,
        os.path.join(here, ckpt_path),
        os.path.join(here, "runs", ckpt_path),
    ]
    if not ckpt_path.endswith(".pt"):
        candidates.append(os.path.join(here, "runs", ckpt_path, "best.pt"))
        candidates.append(os.path.join(here, ckpt_path, "best.pt"))
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "체크포인트를 못 찾았다: %s\n"
        "  best.pt 를 deploy/ 에 두거나 --ckpt 로 절대경로를 줄 것." % ckpt_path)


def load_model(ckpt_path, device):
    """
    체크포인트 로드 → (model, transform, classes, size).

    체크포인트에 저장된 학습 해상도(args.size)로 전처리를 맞춘다.
    """
    path = resolve_ckpt(ckpt_path)
    ck = torch.load(path, map_location=device, weights_only=False)
    classes = ck.get("classes", CLASS_NAMES)
    size = int(ck.get("args", {}).get("size", 224))
    model = build_model()
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    tf = build_transforms(size)
    return model, tf, classes, size, ck.get("val_acc", -1)
