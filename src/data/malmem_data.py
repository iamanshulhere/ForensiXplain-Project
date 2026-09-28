from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("datasets/MalMem2022/MalMem2022.csv")

TARGET_COLUMN = "Class"

METADATA_COLUMNS = [
    "Category",
    "Filename",
]


def load_malmem_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"MalMem2022 dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Missing target column: {TARGET_COLUMN}"
        )

    for column in METADATA_COLUMNS:
        if column not in df.columns:
            raise ValueError(
                f"Missing metadata column: {column}"
            )

    excluded_columns = [
        TARGET_COLUMN,
        *METADATA_COLUMNS,
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    if len(feature_columns) != 55:
        raise ValueError(
            f"Expected 55 model features, found {len(feature_columns)}"
        )

    features = df[feature_columns].copy()
    target = df[TARGET_COLUMN].copy()
    metadata = df[METADATA_COLUMNS].copy()

    return features, target, metadata


def split_malmem_data(test_size=0.2, random_state=42):
    features, target, _ = load_malmem_data()

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    grouped = features.groupby(
        list(features.columns),
        sort=False,
        dropna=False,
    )

    group_ids = list(grouped.groups.keys())

    rng = np.random.default_rng(random_state)
    rng.shuffle(group_ids)

    test_group_count = max(
        1,
        int(round(len(group_ids) * test_size)),
    )

    test_groups = set(group_ids[:test_group_count])

    test_mask = features.apply(
        lambda row: tuple(row) in test_groups,
        axis=1,
    )

    X_train = features.loc[~test_mask].copy()
    X_test = features.loc[test_mask].copy()

    y_train = target.loc[~test_mask].copy()
    y_test = target.loc[test_mask].copy()

    return X_train, X_test, y_train, y_test
