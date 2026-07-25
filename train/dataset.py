"""
manifest.csv -> PyTorch Dataset.

수집기(tools/collect.py)가 남긴 매니페스트를 그대로 읽는다. 라벨 인덱스는
수집 쪽 tl/states.py 와 같은 순서여야 하므로 여기에 다시 정의하지 않고
매니페스트의 label_index 를 그대로 쓴다.
"""

import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


#: 수집기 tl/states.py 의 CLASS_NAMES 와 반드시 같은 순서
CLASS_NAMES = ["red", "yellow", "green", "red_yellow",
               "red_left", "green_left", "left", "unknown"]
NUM_CLASSES = len(CLASS_NAMES)

#: 한 번의 주행을 식별하는 키. 이 단위로 train/val 을 가른다.
DRIVE_KEYS = ["spot_id", "weather", "hour", "object_seed", "state"]

#: 학습에 반드시 있어야 하는 매니페스트 컬럼
MANIFEST_COLS_REQUIRED = ["image_path", "label", "label_index", "spot_id",
                          "weather", "hour", "object_seed", "state",
                          "dist_to_tl_m"]


def load_manifest(root, min_dist=None, max_dist=None, spots=None,
                  weathers=None, hours=None):
    """
    매니페스트를 읽고 필터링한다.

    max_dist : 신호등까지 이 거리 이내만 사용. 수집 때 이미 70m 로 잘랐지만,
               학습 시 더 좁히고 싶을 때 쓴다(멀수록 신호등이 몇 픽셀 안 된다).
               unknown 지점은 신호등이 없어 dist_to_tl_m 이 비어 있으므로 항상 남긴다.
    """
    path = os.path.join(root, "manifest.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "매니페스트가 없다: %s\n"
            "  수집을 먼저 돌릴 것: python3 tools/collect.py" % path)

    df = pd.read_csv(path)
    n0 = len(df)

    # 같은 조합을 재수집하면 행이 중복될 수 있다 (--restart 후 이어쓰기)
    df = df.drop_duplicates(subset=["image_path"], keep="last")

    df["dist_to_tl_m"] = pd.to_numeric(df["dist_to_tl_m"], errors="coerce")
    has_tl = df["dist_to_tl_m"].notna()

    if max_dist is not None:
        df = df[~has_tl | (df["dist_to_tl_m"] <= max_dist)]
    if min_dist is not None:
        df = df[~has_tl | (df["dist_to_tl_m"] >= min_dist)]
    if spots:
        df = df[df["spot_id"].isin(spots)]
    if weathers:
        df = df[df["weather"].isin(weathers)]
    if hours:
        df = df[df["hour"].astype(str).isin([str(h) for h in hours])]

    df = df.reset_index(drop=True)
    print("[data] %d행 로드 (원본 %d, 필터 후 %d)" % (len(df), n0, len(df)))
    return df


def drive_groups(df):
    """행마다 '어느 주행에서 나왔는지' 그룹 키를 만든다."""
    return df[DRIVE_KEYS].astype(str).agg("|".join, axis=1)


def split_by_drive(df, val_ratio=0.2, seed=42, holdout_spots=None):
    """
    train/val 분할.

    ★ 프레임 단위 랜덤 분할을 쓰면 안 된다 ★
    한 주행에서 나온 연속 프레임은 서로 거의 같은 그림이다. 무작위로 섞으면
    같은 주행의 프레임이 train 과 val 에 나뉘어 들어가 val 정확도가 부풀려진다.
    그래서 **주행 단위**로 통째로 가른다.

    holdout_spots 를 주면 그 지점 전체를 val 로 뺀다. 새로운 교차로에 대한
    일반화를 보려면 이쪽이 더 엄격하다.
    """
    if holdout_spots:
        val_mask = df["spot_id"].isin(holdout_spots)
        print("[split] 지점 홀드아웃: %s" % ", ".join(sorted(holdout_spots)))
        return df[~val_mask].reset_index(drop=True), df[val_mask].reset_index(drop=True)

    # 클래스별로 따로 나눈다(계층 분할). 그냥 주행을 무작위로 뽑으면 어떤 클래스의
    # 주행이 전부 val 로 몰려 train 에 그 클래스가 아예 없어질 수 있다.
    groups = drive_groups(df)
    df = df.assign(_grp=groups)
    label_of = df.groupby("_grp")["label_index"].first()

    rng = torch.Generator().manual_seed(seed)
    val_keys = set()
    for lab in sorted(label_of.unique()):
        keys = sorted(label_of[label_of == lab].index)
        if len(keys) == 1:
            continue          # 주행이 하나뿐이면 train 에 남긴다
        n_val = max(1, int(round(len(keys) * val_ratio)))
        n_val = min(n_val, len(keys) - 1)   # train 에 최소 하나는 남긴다
        perm = torch.randperm(len(keys), generator=rng).tolist()
        val_keys.update(keys[i] for i in perm[:n_val])

    val_mask = df["_grp"].isin(val_keys)
    n_drives = len(label_of)
    print("[split] 주행 %d개 중 %d개를 val 로 (프레임 %d / %d)"
          % (n_drives, len(val_keys), int((~val_mask).sum()), int(val_mask.sum())))
    train_df = df[~val_mask].drop(columns=["_grp"]).reset_index(drop=True)
    val_df = df[val_mask].drop(columns=["_grp"]).reset_index(drop=True)
    return train_df, val_df


class TrafficLightDataset(Dataset):
    def __init__(self, df, root, transform=None):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = os.path.join(self.root, row["image_path"])
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label_index"])


def class_counts(df):
    counts = [0] * NUM_CLASSES
    for idx, n in df["label_index"].value_counts().items():
        counts[int(idx)] = int(n)
    return counts


def class_weights(df):
    """
    불균형 보정용 클래스 가중치 (역빈도, 평균 1로 정규화).

    unknown 이 많고 left 계열이 적은 구조라 그대로 두면 모델이 다수 클래스로 쏠린다.
    """
    counts = class_counts(df)
    total = sum(counts)
    w = [(total / (NUM_CLASSES * c)) if c > 0 else 0.0 for c in counts]
    present = [x for x in w if x > 0]
    mean = sum(present) / len(present) if present else 1.0
    return torch.tensor([x / mean for x in w], dtype=torch.float32)
