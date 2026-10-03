"""Standardized Evaluation Protocol for ForensiXplain.

Defines metric calculations, cohort alignment enforcement, no-data-leakage audit,
and comparative evaluation logic for temporal, graph, graph-aware, and fused models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class SupervisedMetrics:
    """Quantitative performance metrics for ground-truth classification."""

    cohort_size: int
    n_positives: int
    n_negatives: int
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    precision_at_k: Optional[float]
    recall_at_k: Optional[float]
    k_value: Optional[int]
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UnsupervisedRankingMetrics:
    """Ranking and score correlation metrics for unsupervised / proxy evaluation."""

    cohort_size: int
    top_k: int
    top_k_overlap_count: int
    jaccard_similarity: float
    kendall_tau: Optional[float]
    kendall_pvalue: Optional[float]
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_supervised_metrics(
    y_true: pd.Series,
    y_pred_binary: pd.Series,
    y_scores: Optional[pd.Series] = None,
    k: Optional[int] = 5,
) -> SupervisedMetrics:
    """Compute rigorous supervised detection metrics with edge-case handling.

    Handles empty positive/negative classes, zero divisions, and optional score-based metrics.
    """
    y_t = y_true.to_numpy(dtype=int)
    y_p = y_pred_binary.to_numpy(dtype=int)

    cohort_size = len(y_t)
    n_positives = int(np.sum(y_t == 1))
    n_negatives = int(np.sum(y_t == 0))

    if cohort_size == 0:
        raise ValueError("Cannot compute metrics on empty cohort")

    # Confusion matrix elements
    if n_positives == 0 or n_negatives == 0:
        # Single class present in ground truth
        tp = int(np.sum((y_t == 1) & (y_p == 1)))
        fp = int(np.sum((y_t == 0) & (y_p == 1)))
        tn = int(np.sum((y_t == 0) & (y_p == 0)))
        fn = int(np.sum((y_t == 1) & (y_p == 0)))
    else:
        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

    # Standard metrics with zero division handling
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    f1 = float(f1_score(y_t, y_p, zero_division=0))

    # ROC-AUC & PR-AUC
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    notes_list: List[str] = []

    if y_scores is not None:
        scores_arr = y_scores.to_numpy(dtype=float)
        if n_positives > 0 and n_negatives > 0 and len(np.unique(scores_arr)) > 1:
            try:
                roc_auc = float(roc_auc_score(y_t, scores_arr))
            except ValueError as err:
                notes_list.append(f"ROC-AUC calculation skipped: {err}")
            try:
                pr_auc = float(average_precision_score(y_t, scores_arr))
            except ValueError as err:
                notes_list.append(f"PR-AUC calculation skipped: {err}")
        else:
            notes_list.append(
                "ROC-AUC/PR-AUC undefined: cohort has single ground-truth class or constant scores."
            )

    # Precision@K & Recall@K
    prec_at_k: Optional[float] = None
    rec_at_k: Optional[float] = None
    if k is not None and k > 0 and y_scores is not None:
        eff_k = min(k, cohort_size)
        top_k_indices = np.argsort(-y_scores.to_numpy(dtype=float))[:eff_k]
        top_k_true = y_t[top_k_indices]
        prec_at_k = float(np.sum(top_k_true == 1)) / eff_k
        if n_positives > 0:
            rec_at_k = float(np.sum(top_k_true == 1)) / n_positives
        else:
            rec_at_k = 0.0

    notes = "; ".join(notes_list) if notes_list else "All metric conditions satisfied."

    return SupervisedMetrics(
        cohort_size=cohort_size,
        n_positives=n_positives,
        n_negatives=n_negatives,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        precision_at_k=prec_at_k,
        recall_at_k=rec_at_k,
        k_value=k,
        notes=notes,
    )


def compute_unsupervised_ranking_metrics(
    scores_a: pd.Series,
    scores_b: pd.Series,
    k: int = 5,
) -> UnsupervisedRankingMetrics:
    """Compute ranking overlap, Jaccard similarity, and Kendall's tau correlation."""
    if len(scores_a) != len(scores_b):
        raise ValueError("Series lengths must match for ranking comparison")
    if len(scores_a) == 0:
        raise ValueError("Cannot compute ranking metrics on empty series")

    cohort_size = len(scores_a)
    eff_k = min(k, cohort_size)

    # Top-K selection (tie-breaking deterministic index)
    top_a_keys = set(scores_a.nlargest(eff_k).index)
    top_b_keys = set(scores_b.nlargest(eff_k).index)

    overlap_count = len(top_a_keys.intersection(top_b_keys))
    union_count = len(top_a_keys.union(top_b_keys))
    jaccard = float(overlap_count / union_count) if union_count > 0 else 1.0

    tau_val: Optional[float] = None
    p_val: Optional[float] = None
    notes = "Ranking comparison completed successfully."

    if cohort_size >= 2:
        res = kendalltau(scores_a.to_numpy(), scores_b.to_numpy())
        tau_val = float(res.statistic) if np.isfinite(res.statistic) else None
        p_val = float(res.pvalue) if np.isfinite(res.pvalue) else None
    else:
        notes = "Kendall tau skipped: cohort size < 2."

    return UnsupervisedRankingMetrics(
        cohort_size=cohort_size,
        top_k=eff_k,
        top_k_overlap_count=overlap_count,
        jaccard_similarity=jaccard,
        kendall_tau=tau_val,
        kendall_pvalue=p_val,
        notes=notes,
    )


def enforce_same_test_cohort(
    model_frames: Dict[str, pd.DataFrame],
    join_key: str = "logical_event_id",
) -> Tuple[Dict[str, pd.DataFrame], Set[str], Dict[str, Set[str]]]:
    """Align multiple model outputs onto the EXACT SAME test cohort.

    Returns aligned frames, common key set, and dictionary of excluded keys per model.
    """
    if not model_frames:
        raise ValueError("model_frames dictionary cannot be empty")

    keys_per_model: Dict[str, Set[str]] = {}
    for model_name, df in model_frames.items():
        if join_key not in df.columns:
            raise ValueError(f"Model frame '{model_name}' missing join key '{join_key}'")
        if df[join_key].isna().any():
            raise ValueError(f"Model frame '{model_name}' contains null '{join_key}' values")
        if df[join_key].duplicated().any():
            dups = df[df[join_key].duplicated()][join_key].unique().tolist()
            raise ValueError(f"Model frame '{model_name}' contains duplicate '{join_key}' keys: {dups}")
        keys_per_model[model_name] = set(df[join_key])

    common_keys = set.intersection(*keys_per_model.values())
    if not common_keys:
        raise ValueError(f"No common '{join_key}' keys found across models: {list(model_frames.keys())}")

    excluded_keys: Dict[str, Set[str]] = {}
    aligned_frames: Dict[str, pd.DataFrame] = {}

    for model_name, df in model_frames.items():
        excluded = keys_per_model[model_name] - common_keys
        excluded_keys[model_name] = excluded
        
        # Sort deterministically by join key to guarantee alignment order
        aligned = df[df[join_key].isin(common_keys)].copy()
        aligned = aligned.sort_values(by=join_key).reset_index(drop=True)
        aligned_frames[model_name] = aligned

    return aligned_frames, common_keys, excluded_keys
