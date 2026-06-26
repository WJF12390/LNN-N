from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


@dataclass
class TabularDataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_name: str


def load_csv_dataset(data_path: str | Path, features: Sequence[str], target: str,
                     clip_likert: bool = True, likert_min: float = 1.0,
                     likert_max: float = 5.0) -> TabularDataset:
    df = pd.read_csv(data_path)
    missing = [c for c in list(features) + [target] if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in dataset: {missing}")
    cols = list(features) + [target]
    data = df[cols].copy()
    for c in cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna(axis=0).reset_index(drop=True)
    if clip_likert:
        for c in cols:
            data[c] = data[c].clip(lower=likert_min, upper=likert_max)
    X = data[list(features)].to_numpy(dtype=np.float32)
    y = data[target].to_numpy(dtype=np.float32).reshape(-1, 1)
    return TabularDataset(X=X, y=y, feature_names=list(features), target_name=target)


def make_folds(n_samples: int, n_splits: int = 5, seed: int = 42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(np.arange(n_samples)))


def standardize_train_valid_test(X_train: np.ndarray, X_valid: np.ndarray, X_test: np.ndarray):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_valid_s = scaler.transform(X_valid).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    return X_train_s, X_valid_s, X_test_s, scaler


def train_valid_split(train_index: np.ndarray, valid_ratio: float = 0.15, seed: int = 42):
    rng = np.random.default_rng(seed)
    shuffled = np.array(train_index).copy()
    rng.shuffle(shuffled)
    n_valid = max(1, int(len(shuffled) * valid_ratio))
    return shuffled[n_valid:], shuffled[:n_valid]
