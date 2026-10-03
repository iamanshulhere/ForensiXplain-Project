"""Dataset label mapping and ground-truth metadata specifications.

Provides explicit ground-truth loading, label normalization, and evaluation suitability
verification for ForensiXplain datasets: M57-Jean, MalMem2022, and OpTC.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetMetadata:
    """Metadata specification for dataset ground-truth evaluation suitability."""

    dataset_name: str
    has_ground_truth_labels: bool
    label_meaning: str
    unit_of_labeling: str  # event | sample | process | file | session
    mappable_to_logical_event_id: bool
    timestamps_available: bool
    process_ids_available: bool
    graph_entities_available: bool
    temporal_features_possible: bool
    graph_features_possible: bool
    suitable_for_supervised_evaluation: bool
    notes: str


# Pre-defined dataset metadata registry based on dataset audit
DATASET_REGISTRY: Dict[str, DatasetMetadata] = {
    "M57-Jean": DatasetMetadata(
        dataset_name="M57-Jean",
        has_ground_truth_labels=False,
        label_meaning="No independent ground-truth anomaly labels available",
        unit_of_labeling="event",
        mappable_to_logical_event_id=True,
        timestamps_available=True,
        process_ids_available=True,
        graph_entities_available=True,
        temporal_features_possible=True,
        graph_features_possible=True,
        suitable_for_supervised_evaluation=False,
        notes="Unsupervised forensic timeline scenario. Used for ranking overlap, score fusion, and evidence attribution.",
    ),
    "MalMem2022": DatasetMetadata(
        dataset_name="MalMem2022",
        has_ground_truth_labels=True,
        label_meaning="Binary classification: Benign (0) vs Malware (1) memory dump sample",
        unit_of_labeling="sample",
        mappable_to_logical_event_id=False,
        timestamps_available=False,
        process_ids_available=False,
        graph_entities_available=False,
        temporal_features_possible=False,
        graph_features_possible=False,
        suitable_for_supervised_evaluation=False,  # Unsuitable for event-level temporal-graph fusion
        notes="Static Volatility memory dump aggregate features. Evaluated via tabular binary classification, not temporal-graph fusion.",
    ),
    "OpTC": DatasetMetadata(
        dataset_name="OpTC",
        has_ground_truth_labels=True,
        label_meaning="Timestamped Red Team attack documentation (PowerShell Empire, UAC bypass, WMI, Mimikatz, RDP exfil)",
        unit_of_labeling="event",
        mappable_to_logical_event_id=True,
        timestamps_available=True,
        process_ids_available=True,
        graph_entities_available=True,
        temporal_features_possible=True,
        graph_features_possible=True,
        suitable_for_supervised_evaluation=True,
        notes="DARPA OpTC host telemetry logs. PDF ground-truth report provides narrative attack timeline.",
    ),
}


def get_dataset_metadata(dataset_name: str) -> DatasetMetadata:
    """Retrieve dataset metadata or raise ValueError if unsupported."""
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Supported datasets: {sorted(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[dataset_name]


def normalize_binary_labels(
    labels: pd.Series,
    positive_value: Any = 1,
    negative_value: Any = 0,
) -> pd.Series:
    """Normalize binary target series into boolean/integer 0 and 1 values."""
    if labels.empty:
        return pd.Series([], dtype=int)

    str_vals = labels.astype(str).str.strip().str.lower()
    
    pos_set = {str(positive_value).lower(), "1", "true", "malware", "malicious", "anomaly"}
    neg_set = {str(negative_value).lower(), "0", "false", "benign", "normal"}

    mapping = {}
    for val in str_vals.unique():
        if val in pos_set:
            mapping[val] = 1
        elif val in neg_set:
            mapping[val] = 0
        else:
            raise ValueError(f"Unrecognized label value: {val}")

    return str_vals.map(mapping).astype(int)


def validate_label_alignment(
    event_ids: pd.Series,
    labels: pd.Series,
    key_name: str = "logical_event_id",
) -> Tuple[pd.Series, pd.Series]:
    """Validate key uniqueness and completeness for ground-truth label alignment."""
    if len(event_ids) != len(labels):
        raise ValueError(
            f"Length mismatch between {key_name} ({len(event_ids)}) and labels ({len(labels)})"
        )
    
    if event_ids.isna().any():
        raise ValueError(f"Null values found in {key_name}")

    if event_ids.duplicated().any():
        dups = event_ids[event_ids.duplicated()].unique().tolist()
        raise ValueError(f"Duplicate {key_name} keys found: {dups}")

    if labels.isna().any():
        raise ValueError("Null values found in labels series")

    return event_ids, labels
