"""Model Comparison module for ForensiXplain.

Executes controlled, side-by-side comparison of Temporal-only, Graph-only, Graph-aware,
and Temporal-Graph Fused models under identical evaluation cohorts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.evaluation.dataset_label_mapping import DATASET_REGISTRY, get_dataset_metadata
from src.evaluation.evaluation_protocol import (
    SupervisedMetrics,
    UnsupervisedRankingMetrics,
    compute_supervised_metrics,
    compute_unsupervised_ranking_metrics,
    enforce_same_test_cohort,
)


def _find_score_col(columns: list[str]) -> str:
    for candidate in [
        "fusion_score_alpha_0_50",
        "fusion_score",
        "temporal_anomaly_score",
        "graph_anomaly_score",
        "graph_aware_score",
        "anomaly_score",
        "score",
    ]:
        if candidate in columns:
            return candidate
    score_cols = [c for c in columns if "score" in c.lower()]
    if score_cols:
        return score_cols[0]
    raise ValueError(f"No score column found in columns: {columns}")


def _find_pred_col(columns: list[str]) -> str:
    for candidate in [
        "fusion_predicted_anomaly_alpha_0_50",
        "fusion_predicted_anomaly",
        "temporal_predicted_anomaly",
        "graph_predicted_anomaly",
        "graph_aware_predicted_anomaly",
        "predicted_anomaly",
        "is_anomaly",
    ]:
        if candidate in columns:
            return candidate
    pred_cols = [c for c in columns if "predict" in c.lower() or "anomaly" in c.lower()]
    if pred_cols:
        return pred_cols[0]
    raise ValueError(f"No prediction column found in columns: {columns}")


def load_and_compare_models(
    temporal_path: Path,
    graph_path: Path,
    fused_path: Path,
    dataset_name: str = "M57-Jean",
    join_key: str = "logical_event_id",
    ground_truth_labels: Optional[pd.Series] = None,
    k: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load, align, and compare component and fused models under a single protocol."""

    meta = get_dataset_metadata(dataset_name)

    # 1. Read input CSV files
    frames: Dict[str, pd.DataFrame] = {}
    for name, path in [
        ("Temporal-only", temporal_path),
        ("Graph-only", graph_path),
        ("Fused (alpha=0.50)", fused_path),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Model output file missing: {path}")
        df = pd.read_csv(path, low_memory=False)
        frames[name] = df

    # 2. Enforce SAME TEST COHORT across all models
    aligned_frames, common_keys, excluded_keys = enforce_same_test_cohort(
        model_frames=frames,
        join_key=join_key,
    )

    cohort_size = len(common_keys)
    report_lines: List[str] = [
        f"ForensiXplain Model Comparison Report — {dataset_name}",
        "=" * 60,
        f"Dataset: {dataset_name}",
        f"Ground Truth Status: {'Labeled' if meta.has_ground_truth_labels else 'Unlabeled (Unsupervised / Proxy evaluation)'}",
        f"Mappable to logical_event_id: {meta.mappable_to_logical_event_id}",
        f"Aligned Cohort Size: {cohort_size} events",
        f"Join Key: {join_key}",
        "-" * 60,
    ]

    for m_name, excl in excluded_keys.items():
        if excl:
            report_lines.append(f"Excluded from {m_name}: {len(excl)} events ({sorted(list(excl))[:3]}...)")
        else:
            report_lines.append(f"Excluded from {m_name}: 0 events")

    report_lines.append("-" * 60)

    comparison_rows: List[Dict[str, Any]] = []

    # 3. Supervised Ground-Truth Evaluation (if labels provided & dataset suitable)
    if meta.suitable_for_supervised_evaluation and ground_truth_labels is not None:
        report_lines.append("Supervised Ground-Truth Evaluation Metrics:")
        report_lines.append(
            f"{'Model':<20} | {'Cohort':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'ROC-AUC':<8} | {'PR-AUC':<8}"
        )
        report_lines.append("-" * 75)

        for m_name, df in aligned_frames.items():
            score_col = _find_score_col(list(df.columns))
            pred_col = _find_pred_col(list(df.columns))
            
            y_scores = df[score_col]
            y_pred = df[pred_col].astype(int)

            # Align ground truth labels
            aligned_labels = ground_truth_labels.loc[df[join_key]].to_numpy()

            metrics: SupervisedMetrics = compute_supervised_metrics(
                y_true=pd.Series(aligned_labels),
                y_pred_binary=y_pred,
                y_scores=y_scores,
                k=k,
            )

            roc_str = f"{metrics.roc_auc:.4f}" if metrics.roc_auc is not None else "N/A"
            pr_str = f"{metrics.pr_auc:.4f}" if metrics.pr_auc is not None else "N/A"

            report_lines.append(
                f"{m_name:<20} | {metrics.cohort_size:<6} | {metrics.precision:.4f} | {metrics.recall:.4f} | {metrics.f1:.4f} | {roc_str:<8} | {pr_str:<8}"
            )

            comparison_rows.append({
                "dataset": dataset_name,
                "model": m_name,
                "cohort_size": metrics.cohort_size,
                "n_positives": metrics.n_positives,
                "n_negatives": metrics.n_negatives,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1,
                "roc_auc": metrics.roc_auc,
                "pr_auc": metrics.pr_auc,
                "precision_at_k": metrics.precision_at_k,
                "recall_at_k": metrics.recall_at_k,
                "evaluation_type": "Supervised Ground Truth",
            })
    else:
        # 4. Unsupervised / Proxy Ranking Comparison (e.g. M57-Jean)
        report_lines.append("Unsupervised Proxy Evaluation (Top-K Overlap & Rank Correlation):")
        report_lines.append(f"{'Model Pair':<35} | {'Top-K':<6} | {'Overlap':<7} | {'Jaccard':<7} | {'Kendall Tau':<11}")
        report_lines.append("-" * 75)

        # Get scores dictionary indexed by join_key
        model_scores: Dict[str, pd.Series] = {}
        for m_name, df in aligned_frames.items():
            score_col = _find_score_col(list(df.columns))
            model_scores[m_name] = pd.Series(df[score_col].values, index=df[join_key])

        model_names = list(model_scores.keys())
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                name_a, name_b = model_names[i], model_names[j]
                pair_name = f"{name_a} vs {name_b}"
                
                rank_metrics: UnsupervisedRankingMetrics = compute_unsupervised_ranking_metrics(
                    scores_a=model_scores[name_a],
                    scores_b=model_scores[name_b],
                    k=k,
                )

                tau_str = f"{rank_metrics.kendall_tau:.4f}" if rank_metrics.kendall_tau is not None else "N/A"

                report_lines.append(
                    f"{pair_name:<35} | {rank_metrics.top_k:<6} | {rank_metrics.top_k_overlap_count:<7} | {rank_metrics.jaccard_similarity:.4f} | {tau_str:<11}"
                )

                comparison_rows.append({
                    "dataset": dataset_name,
                    "model_pair": pair_name,
                    "cohort_size": rank_metrics.cohort_size,
                    "top_k": rank_metrics.top_k,
                    "top_k_overlap": rank_metrics.top_k_overlap_count,
                    "jaccard_similarity": rank_metrics.jaccard_similarity,
                    "kendall_tau": rank_metrics.kendall_tau,
                    "evaluation_type": "Unsupervised Proxy (Rank Overlap)",
                })

    comparison_df = pd.DataFrame(comparison_rows)
    report_text = "\n".join(report_lines)

    # Convert aligned frames into a summary table
    summary_list = []
    for m_name, df in aligned_frames.items():
        summary_list.append(df.assign(model_name=m_name))
    summary_df = pd.concat(summary_list, ignore_index=True)

    return comparison_df, summary_df, report_text
