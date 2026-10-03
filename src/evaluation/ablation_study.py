"""Controlled, retrospective component ablations for M57-Jean."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.evaluation.temporal_graph_fusion import normalize_scores


CASE_ID = "M57-Jean"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = PROJECT_ROOT / "data" / "features" / CASE_ID
RESULTS_DIR = PROJECT_ROOT / "results" / CASE_ID
TEMPORAL_FEATURES_PATH = FEATURE_DIR / "temporal_features.csv"
GRAPH_FEATURES_PATH = FEATURE_DIR / "graph_features.csv"
TEMPORAL_RESULTS_PATH = RESULTS_DIR / "temporal_anomalies.csv"
VALIDATED_FUSION_PATH = RESULTS_DIR / "fused_anomalies.csv"

ABLATION_RESULTS_PATH = RESULTS_DIR / "ablation_results.csv"
ABLATION_COMPARISON_PATH = RESULTS_DIR / "ablation_comparison.csv"
ABLATION_REPORT_PATH = RESULTS_DIR / "ablation_report.txt"
ABLATION_SUMMARY_PATH = RESULTS_DIR / "ablation_summary.json"

N_ESTIMATORS = 500
CONTAMINATION = 0.10
RANDOM_STATE = 42
N_JOBS = -1
FUSION_ALPHA = 0.50
TOP_K = 5
EXPECTED_COHORT_SIZE = 46

TEMPORAL_FEATURES = (
    "gap_log_seconds",
    "time_since_previous_event_seconds",
    "events_prev_10s",
    "events_next_10s",
    "local_density_10s",
    "events_prev_30s",
    "events_next_30s",
    "local_density_30s",
    "events_prev_60s",
    "events_next_60s",
    "local_density_60s",
    "process_changed",
    "rapid_event",
    "short_event_gap",
    "medium_event_gap",
    "long_event_gap",
)

GRAPH_STRUCTURAL_FEATURES = (
    "parent_count",
    "child_count",
    "graph_degree",
    "in_degree",
    "out_degree",
)

PROCESS_ARTIFACT_FEATURES = (
    "command_line_count",
    "module_count",
    "memory_region_count",
    "relationship_type_count",
)

METADATA_FEATURES = (
    "case_id",
    "logical_event_id",
    "process_id",
    "parent_process_id",
    "process",
    "timestamp",
    "is_first_event",
    "provenance",
    "evidence_ids",
)

GRAPH_FULL_FEATURES = GRAPH_STRUCTURAL_FEATURES + PROCESS_ARTIFACT_FEATURES
ALL_MODEL_FEATURES = TEMPORAL_FEATURES + GRAPH_FULL_FEATURES

EXPERIMENT_DEFINITIONS = {
    "E1": {
        "name": "Temporal-only",
        "model_type": "single IsolationForest",
        "groups": ("A",),
        "features": TEMPORAL_FEATURES,
    },
    "E2": {
        "name": "Graph-Structural-only",
        "model_type": "single IsolationForest",
        "groups": ("B",),
        "features": GRAPH_STRUCTURAL_FEATURES,
    },
    "E3": {
        "name": "Graph-Full",
        "model_type": "single IsolationForest",
        "groups": ("B", "C"),
        "features": GRAPH_FULL_FEATURES,
    },
    "E4": {
        "name": "Graph-aware Combined IF",
        "model_type": "joint feature-space IsolationForest",
        "groups": ("A", "B", "C"),
        "features": ALL_MODEL_FEATURES,
    },
    "E5": {
        "name": "Temporal + Graph-Structural Fused",
        "model_type": "score-level fusion",
        "groups": ("A", "B"),
        "features": None,
    },
    "E6": {
        "name": "Temporal + Graph-Full Fused",
        "model_type": "score-level fusion",
        "groups": ("A", "B", "C"),
        "features": None,
    },
}

EXPERIMENT_MATRIX_FEATURES = {
    "E1": {"joint": TEMPORAL_FEATURES},
    "E2": {"joint": GRAPH_STRUCTURAL_FEATURES},
    "E3": {"joint": GRAPH_FULL_FEATURES},
    "E4": {"joint": ALL_MODEL_FEATURES},
    "E5": {"temporal": TEMPORAL_FEATURES, "graph": GRAPH_STRUCTURAL_FEATURES},
    "E6": {"temporal": TEMPORAL_FEATURES, "graph": GRAPH_FULL_FEATURES},
}

RQ_PAIRS = (
    ("RQ1", "E1", "E2", "Temporal anomaly structure vs process-tree topology"),
    ("RQ2", "E2", "E3", "Marginal comparison after adding process/artifact features"),
    ("RQ3", "E3", "E4", "Graph-full model vs graph-full plus temporal features"),
    ("RQ4", "E4", "E6", "Joint graph-aware feature space vs score-level fusion"),
)


def _event_ids(frame: pd.DataFrame, description: str) -> pd.Series:
    if "logical_event_id" not in frame.columns:
        raise KeyError(f"{description} is missing logical_event_id")
    ids = frame["logical_event_id"].astype("string").str.strip()
    if ids.isna().any() or ids.eq("").any():
        raise ValueError(f"{description} contains an empty logical_event_id")
    if ids.duplicated().any():
        duplicated = sorted(ids.loc[ids.duplicated(keep=False)].unique().tolist())
        raise ValueError(f"{description} contains duplicate logical_event_id values: {duplicated}")
    return ids.astype(str)


def validate_scored_cohort(
    temporal_scores: pd.DataFrame,
    validated_fusion: pd.DataFrame,
    expected_size: int = EXPECTED_COHORT_SIZE,
) -> list[str]:
    """Use and cross-check the exact cohort already scored by fusion."""
    temporal_ids = _event_ids(temporal_scores, "Temporal anomaly output")
    fusion_ids = _event_ids(validated_fusion, "Validated fusion output")
    if "temporal_anomaly_rank" not in temporal_scores.columns:
        raise KeyError("Temporal anomaly output is missing temporal_anomaly_rank")
    scored = temporal_scores.loc[
        pd.to_numeric(temporal_scores["temporal_anomaly_rank"], errors="coerce").notna()
    ].copy()
    scored_ids = _event_ids(scored, "Scored temporal anomaly output")
    if set(scored_ids) != set(fusion_ids):
        raise ValueError(
            "Temporal scored IDs differ from the validated fusion cohort: "
            f"temporal_only={sorted(set(scored_ids) - set(fusion_ids))}, "
            f"fusion_only={sorted(set(fusion_ids) - set(scored_ids))}"
        )
    if len(scored_ids) != expected_size:
        raise ValueError(
            f"Expected {expected_size} scored logical events, found {len(scored_ids)}"
        )
    return sorted(scored_ids)


def load_repository_cohort() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Read feature inputs and derive the validated 46-event cohort."""
    temporal_features = pd.read_csv(TEMPORAL_FEATURES_PATH, low_memory=False)
    graph_features = pd.read_csv(GRAPH_FEATURES_PATH, low_memory=False)
    temporal_scores = pd.read_csv(TEMPORAL_RESULTS_PATH, low_memory=False)
    validated_fusion = pd.read_csv(VALIDATED_FUSION_PATH, low_memory=False)
    cohort_ids = validate_scored_cohort(temporal_scores, validated_fusion)
    return temporal_features, graph_features, cohort_ids


def build_feature_matrices(
    temporal_features: pd.DataFrame,
    graph_features: pd.DataFrame,
    cohort_ids: Iterable[str],
    expected_size: int = EXPECTED_COHORT_SIZE,
) -> tuple[pd.DataFrame, dict[str, dict[str, pd.DataFrame]]]:
    """Return event metadata and explicit, numeric A/B/C feature matrices."""
    ids = [str(event_id).strip() for event_id in cohort_ids]
    if len(ids) != expected_size:
        raise ValueError(f"Expected {expected_size} cohort IDs, found {len(ids)}")
    if any(not event_id for event_id in ids):
        raise ValueError("Cohort logical_event_id values cannot be empty")
    if len(ids) != len(set(ids)):
        raise ValueError("Cohort contains duplicate logical_event_id values")
    ids = sorted(ids)

    temporal_ids = _event_ids(temporal_features, "Temporal feature output")
    graph_ids = _event_ids(graph_features, "Graph feature output")
    if not set(ids).issubset(set(temporal_ids)):
        raise ValueError("Temporal features do not cover the complete scored cohort")
    if not set(ids).issubset(set(graph_ids)):
        raise ValueError("Graph features do not cover the complete scored cohort")

    missing_temporal = set(TEMPORAL_FEATURES + METADATA_FEATURES) - set(temporal_features.columns)
    missing_graph = set(GRAPH_FULL_FEATURES) - set(graph_features.columns)
    if missing_temporal:
        raise KeyError(f"Temporal feature output is missing columns: {sorted(missing_temporal)}")
    if missing_graph:
        raise KeyError(f"Graph feature output is missing columns: {sorted(missing_graph)}")

    temporal = temporal_features.copy()
    graph = graph_features.copy()
    temporal["logical_event_id"] = temporal_ids
    graph["logical_event_id"] = graph_ids
    temporal = temporal.loc[temporal["logical_event_id"].isin(ids)].copy()
    graph = graph.loc[graph["logical_event_id"].isin(ids)].copy()
    temporal = temporal.sort_values("logical_event_id", kind="mergesort").reset_index(drop=True)
    graph = graph.sort_values("logical_event_id", kind="mergesort").reset_index(drop=True)

    for column in ("case_id", "process_id", "timestamp"):
        if column in graph.columns:
            left = temporal[column].astype("string").fillna("").tolist()
            right = graph[column].astype("string").fillna("").tolist()
            if left != right:
                raise ValueError(f"Temporal and graph {column} values disagree by logical_event_id")
    if temporal["case_id"].astype(str).ne(CASE_ID).any():
        raise ValueError(f"Cohort contains events outside dataset {CASE_ID}")
    first_event = pd.to_numeric(temporal["is_first_event"], errors="coerce")
    if first_event.isna().any() or first_event.ne(0).any():
        raise ValueError("The excluded boundary event must remain excluded from the ablation cohort")

    metadata = temporal[list(METADATA_FEATURES + TEMPORAL_FEATURES)].copy()
    graph_values = graph[["logical_event_id", *GRAPH_FULL_FEATURES]].copy()
    cohort = metadata.merge(
        graph_values,
        on="logical_event_id",
        how="inner",
        validate="one_to_one",
        sort=False,
    ).sort_values("logical_event_id", kind="mergesort").reset_index(drop=True)
    if len(cohort) != expected_size or cohort["logical_event_id"].nunique() != expected_size:
        raise ValueError("The cohort changed unexpectedly while aligning feature sources")

    def numeric_matrix(columns: tuple[str, ...]) -> pd.DataFrame:
        numeric = cohort.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
        values = numeric.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            bad = numeric.columns[~np.isfinite(values).all(axis=0)].tolist()
            raise ValueError(f"Non-finite values in required model features: {bad}")
        numeric.index = pd.Index(cohort["logical_event_id"].astype(str), name="logical_event_id")
        return numeric

    temporal_matrix = numeric_matrix(TEMPORAL_FEATURES)
    graph_structural_matrix = numeric_matrix(GRAPH_STRUCTURAL_FEATURES)
    graph_full_matrix = numeric_matrix(GRAPH_FULL_FEATURES)
    combined_matrix = numeric_matrix(ALL_MODEL_FEATURES)

    matrices = {
        "E1": {"joint": temporal_matrix.copy()},
        "E2": {"joint": graph_structural_matrix.copy()},
        "E3": {"joint": graph_full_matrix.copy()},
        "E4": {"joint": combined_matrix.copy()},
        "E5": {
            "temporal": temporal_matrix.copy(),
            "graph": graph_structural_matrix.copy(),
        },
        "E6": {
            "temporal": temporal_matrix.copy(),
            "graph": graph_full_matrix.copy(),
        },
    }
    expected_columns = {
        "E1": {"joint": TEMPORAL_FEATURES},
        "E2": {"joint": GRAPH_STRUCTURAL_FEATURES},
        "E3": {"joint": GRAPH_FULL_FEATURES},
        "E4": {"joint": ALL_MODEL_FEATURES},
        "E5": {"temporal": TEMPORAL_FEATURES, "graph": GRAPH_STRUCTURAL_FEATURES},
        "E6": {"temporal": TEMPORAL_FEATURES, "graph": GRAPH_FULL_FEATURES},
    }
    for experiment_id, components in matrices.items():
        for component, matrix in components.items():
            if matrix.columns.tolist() != list(expected_columns[experiment_id][component]):
                raise AssertionError(f"Unexpected feature membership in {experiment_id}/{component}")
            if matrix.shape[0] != expected_size or matrix.index.tolist() != ids:
                raise ValueError(f"{experiment_id}/{component} does not use the common cohort")
            if set(matrix.columns) & set(METADATA_FEATURES):
                raise ValueError(f"Metadata leaked into {experiment_id}/{component}")
    return cohort, matrices


def _fit_isolation_forest(matrix: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix.to_numpy(dtype=float))
    detector = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    detector.fit(scaled)
    anomaly_scores = pd.Series(
        -detector.decision_function(scaled), index=matrix.index, dtype=float
    )
    candidate_flags = pd.Series(
        detector.predict(scaled) == -1, index=matrix.index, dtype=bool
    )
    return anomaly_scores, candidate_flags


def _ranked_result(
    event_ids: pd.Series,
    scores: pd.Series,
    candidates: pd.Series,
    metadata: pd.DataFrame,
    experiment_id: str,
    component_scores: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "logical_event_id": event_ids.astype(str).to_numpy(),
            "score": scores.reindex(event_ids.astype(str)).to_numpy(dtype=float),
            "anomaly_candidate": candidates.reindex(event_ids.astype(str)).to_numpy(dtype=bool),
        }
    )
    result = result.merge(metadata, on="logical_event_id", how="left", validate="one_to_one")
    result["experiment_id"] = experiment_id
    result["experiment_name"] = EXPERIMENT_DEFINITIONS[experiment_id]["name"]
    result["model_type"] = EXPERIMENT_DEFINITIONS[experiment_id]["model_type"]
    result["feature_groups"] = "+".join(EXPERIMENT_DEFINITIONS[experiment_id]["groups"])
    spec_features = EXPERIMENT_DEFINITIONS[experiment_id]["features"]
    matrix_features = EXPERIMENT_MATRIX_FEATURES[experiment_id]
    component_counts = {name: len(columns) for name, columns in matrix_features.items()}
    result["feature_component_counts"] = json.dumps(
        component_counts, separators=(",", ":"), sort_keys=True
    )
    if spec_features is not None:
        feature_columns = list(spec_features)
        result["feature_count"] = len(feature_columns)
        result["feature_columns"] = ";".join(feature_columns)
        result["score_scale"] = "negative IsolationForest decision_function"
        result["fusion_alpha"] = np.nan
        result["temporal_component_score"] = np.nan
        result["graph_component_score"] = np.nan
    else:
        if component_scores is None:
            raise ValueError(f"Component scores are required for fused experiment {experiment_id}")
        temporal_component = component_scores["temporal"].reindex(event_ids.astype(str))
        graph_component = component_scores["graph"].reindex(event_ids.astype(str))
        result["feature_count"] = sum(component_counts.values())
        result["feature_columns"] = "|".join(
            f"{name}:{';'.join(columns)}" for name, columns in matrix_features.items()
        )
        result["score_scale"] = "weighted average of min-max normalized component scores"
        result["fusion_alpha"] = FUSION_ALPHA
        result["temporal_component_score"] = temporal_component.to_numpy(dtype=float)
        result["graph_component_score"] = graph_component.to_numpy(dtype=float)
    result = result.sort_values(
        ["score", "logical_event_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1, dtype=int)
    result["top_5_candidate"] = result["rank"].le(TOP_K)
    return result


def _experiment_summaries(results: pd.DataFrame) -> dict[str, dict[str, object]]:
    summaries: dict[str, dict[str, object]] = {}
    for experiment_id in EXPERIMENT_DEFINITIONS:
        rows = results.loc[results["experiment_id"] == experiment_id].copy()
        rows = rows.sort_values("rank", kind="mergesort")
        top = rows.head(TOP_K)
        evidence_count = int(top["evidence_traceable"].sum())
        matrix_features = EXPERIMENT_MATRIX_FEATURES[experiment_id]
        component_counts = {name: len(columns) for name, columns in matrix_features.items()}
        summaries[experiment_id] = {
            "name": EXPERIMENT_DEFINITIONS[experiment_id]["name"],
            "model_type": EXPERIMENT_DEFINITIONS[experiment_id]["model_type"],
            "feature_groups": list(EXPERIMENT_DEFINITIONS[experiment_id]["groups"]),
            "feature_count": sum(component_counts.values()),
            "feature_component_counts": component_counts,
            "feature_columns": {name: list(columns) for name, columns in matrix_features.items()},
            "cohort_size": int(len(rows)),
            "candidate_count": int(rows["anomaly_candidate"].sum()),
            "top_5_logical_event_ids": top["logical_event_id"].tolist(),
            "top_5_evidence_traceable_count": evidence_count,
            "top_5_evidence_traceability_coverage": evidence_count / len(top),
        }
    return summaries


def compare_experiments(results: pd.DataFrame) -> pd.DataFrame:
    """Calculate only unsupervised ranking and evidence-traceability comparisons."""
    rows: list[dict[str, object]] = []
    for rq_id, left_id, right_id, question in RQ_PAIRS:
        left = results.loc[results["experiment_id"] == left_id].copy()
        right = results.loc[results["experiment_id"] == right_id].copy()
        left_ids = _event_ids(left, left_id)
        right_ids = _event_ids(right, right_id)
        if set(left_ids) != set(right_ids) or len(left) != len(right):
            raise ValueError(f"{left_id} and {right_id} do not use exactly the same event cohort")
        if len(left) != EXPECTED_COHORT_SIZE:
            raise ValueError(f"Pair {left_id}/{right_id} does not contain 46 events")

        left_top = left.sort_values("rank", kind="mergesort").head(TOP_K)
        right_top = right.sort_values("rank", kind="mergesort").head(TOP_K)
        left_top_ids = set(left_top["logical_event_id"].astype(str))
        right_top_ids = set(right_top["logical_event_id"].astype(str))
        top_overlap = left_top_ids & right_top_ids
        top_union = left_top_ids | right_top_ids
        left_candidates = set(left.loc[left["anomaly_candidate"], "logical_event_id"].astype(str))
        right_candidates = set(right.loc[right["anomaly_candidate"], "logical_event_id"].astype(str))
        candidate_overlap = left_candidates & right_candidates

        aligned = left[["logical_event_id", "score"]].merge(
            right[["logical_event_id", "score"]],
            on="logical_event_id",
            suffixes=("_left", "_right"),
            validate="one_to_one",
        ).sort_values("logical_event_id", kind="mergesort")
        tau = kendalltau(aligned["score_left"], aligned["score_right"])
        rows.append(
            {
                "research_question": rq_id,
                "comparison": f"{left_id} vs {right_id}",
                "experiment_a": left_id,
                "experiment_b": right_id,
                "question": question,
                "cohort_size": len(aligned),
                "top_k": TOP_K,
                "top_5_overlap_count": len(top_overlap),
                "top_5_overlap_event_ids": ";".join(sorted(top_overlap)),
                "top_5_jaccard": len(top_overlap) / len(top_union) if top_union else 1.0,
                "kendall_tau": float(tau.statistic),
                "kendall_tau_pvalue": float(tau.pvalue),
                "top_anomaly_candidate_overlap_count": len(candidate_overlap),
                "top_anomaly_candidate_overlap_event_ids": ";".join(sorted(candidate_overlap)),
                "experiment_a_candidate_count": len(left_candidates),
                "experiment_b_candidate_count": len(right_candidates),
                "experiment_a_top5_evidence_traceability_coverage": float(
                    left_top["evidence_traceable"].mean()
                ),
                "experiment_b_top5_evidence_traceability_coverage": float(
                    right_top["evidence_traceable"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _format_report(
    results: pd.DataFrame,
    comparisons: pd.DataFrame,
    cohort_ids: list[str],
) -> str:
    lines = [
        "ForensiXplain Multi-Component Ablation Study",
        "Dataset: M57-Jean",
        "",
        "1. Experiment definitions",
        "--------------------------",
        "E1: Temporal-only; one IsolationForest on Category A.",
        "E2: Graph-Structural-only; one IsolationForest on Category B.",
        "E3: Graph-Full; one IsolationForest on Categories B+C.",
        "E4: Graph-aware Combined IF; one joint feature-space IsolationForest on A+B+C.",
        "E5: Temporal + Graph-Structural Fused; A and B component IsolationForests, then score-level fusion.",
        "E6: Temporal + Graph-Full Fused; A and B+C component IsolationForests, then score-level fusion.",
        "E5 and E6 reuse the matching E1/E2/E3 component scores. E6 is score-level fusion, not a joint A+B+C model.",
        "",
        "2. Feature groups",
        "-----------------",
        f"Category A (temporal; {len(TEMPORAL_FEATURES)}): " + "; ".join(TEMPORAL_FEATURES),
        f"Category B (graph structural; {len(GRAPH_STRUCTURAL_FEATURES)}): "
        + "; ".join(GRAPH_STRUCTURAL_FEATURES),
        f"Category C (process/artifact relationships; {len(PROCESS_ARTIFACT_FEATURES)}): "
        + "; ".join(PROCESS_ARTIFACT_FEATURES),
        "Category D (metadata/identifiers; excluded from every model matrix): "
        + "; ".join(METADATA_FEATURES),
        "",
        "3. Controlled settings",
        "-----------------------",
        f"IsolationForest: n_estimators={N_ESTIMATORS}, contamination={CONTAMINATION:.2f}, "
        f"random_state={RANDOM_STATE}, n_jobs={N_JOBS}.",
        "StandardScaler is fit separately to each exact feature matrix using the full cohort.",
        f"E5/E6: min-max normalize each component score over the full cohort; alpha={FUSION_ALPHA:.2f}.",
        f"Top-K: K={TOP_K}; score ties are ordered by logical_event_id.",
        "",
        "4. Cohort definition",
        "---------------------",
        f"All six experiments use the same {len(cohort_ids)} scored logical events, keyed by logical_event_id.",
        "Cohort IDs were cross-checked against temporal_anomalies.csv ranks and the validated fused_anomalies.csv.",
        "The first timeline boundary event LEVT-M57-Jean-PROCESS-812-20091121013230 remains excluded.",
        "Logical event IDs are unique in each feature source and in each experiment result.",
        "M57-Jean has no independent event-level ground-truth anomaly labels; this report uses only unsupervised/proxy comparisons.",
        "",
        "5. Results",
        "----------",
        "Experiment | Features | Candidate count | Top-5 evidence traceability | Top-5 logical_event_id values",
    ]
    summaries = _experiment_summaries(results)
    for experiment_id, summary in summaries.items():
        component_counts = summary["feature_component_counts"]
        if list(component_counts) == ["joint"]:
            feature_description = f"joint {component_counts['joint']}"
        else:
            feature_description = " + ".join(
                f"{name} {count}" for name, count in component_counts.items()
            )
        top_ids = ";".join(summary["top_5_logical_event_ids"])
        lines.append(
            f"{experiment_id} {summary['name']} | {feature_description} "
            f"(total {summary['feature_count']}) | {summary['candidate_count']} | "
            f"{summary['top_5_evidence_traceable_count']}/5 "
            f"({summary['top_5_evidence_traceability_coverage']:.3f}) | {top_ids}"
        )
    lines.extend(
        [
            "",
            "6. Pairwise comparisons",
            "------------------------",
            "RQ | Pair | Top-5 overlap | Jaccard | Kendall tau | tau p-value | Candidate overlap | Top-5 evidence coverage (A/B)",
        ]
    )
    for row in comparisons.to_dict(orient="records"):
        lines.append(
            f"{row['research_question']} | {row['comparison']} | {row['top_5_overlap_count']}/5 | "
            f"{row['top_5_jaccard']:.3f} | {row['kendall_tau']:.6f} | "
            f"{row['kendall_tau_pvalue']:.6g} | "
            f"{row['top_anomaly_candidate_overlap_count']} | "
            f"{row['experiment_a_top5_evidence_traceability_coverage']:.3f}/"
            f"{row['experiment_b_top5_evidence_traceability_coverage']:.3f}"
        )
    lines.extend(
        [
            "",
            "7. Interpretation limited to proxy metrics",
            "--------------------------------------------",
            "Top-5 overlap and Jaccard summarize agreement among the five highest-ranked events; Kendall tau summarizes rank association across all 46 events.",
            "Candidate overlap summarizes shared IsolationForest flags; for E5/E6, which have no separately fit fusion detector, candidates are the top ceil(0.10*N) fused scores.",
            "Evidence traceability coverage is the share of each experiment's top five with both non-empty evidence_ids and provenance in the source event record.",
            "These measures describe ranking agreement and source traceability. They do not establish which experiment is more accurate or whether an event is malicious.",
            "No supervised classification metrics are calculated or reported.",
            "",
            "8. Evidence traceability",
            "-------------------------",
            "Each event result retains evidence_ids and provenance as audit metadata; neither field is supplied to a model.",
            "The per-event ablation_results.csv records each experiment's score, rank, candidate status, source evidence IDs, and provenance.",
            "",
            "9. Retrospective scope and leakage disclosure",
            "----------------------------------------------",
            "The events_next_* features use post-event lookahead.",
            "Graph metrics are calculated from the complete reconstructed graph.",
            "Global normalization can use information across the complete cohort; StandardScaler and fusion min-max normalization are fit over all 46 events.",
            "Therefore, this is retrospective/post-mortem forensic reconstruction and comparison.",
            "It must not be presented as causal, online, real-time, streaming, or deployment-ready anomaly detection.",
            "",
            "10. Limitations",
            "---------------",
            "The single M57-Jean cohort is small and has no independent event-level anomaly labels.",
            "Top-K overlap and rank correlation measure agreement, not detection quality or forensic truth.",
            "Full-timeline temporal context, complete-graph metrics, and cohort-wide scaling/normalization make these rankings retrospective.",
            "Evidence IDs and provenance support review but do not establish maliciousness or causality.",
            "",
        ]
    )
    return "\n".join(lines)


def run_ablation_study(
    temporal_features: pd.DataFrame,
    graph_features: pd.DataFrame,
    cohort_ids: Iterable[str],
    expected_size: int = EXPECTED_COHORT_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], str]:
    """Run E1-E6 using a single validated cohort and fixed controls."""
    cohort, matrices = build_feature_matrices(
        temporal_features, graph_features, cohort_ids, expected_size=expected_size
    )
    cohort_ids_sorted = cohort["logical_event_id"].astype(str).tolist()
    metadata = cohort[list(METADATA_FEATURES)].copy()
    metadata["evidence_traceable"] = (
        metadata["evidence_ids"].fillna("").astype(str).str.strip().ne("")
        & metadata["provenance"].fillna("").astype(str).str.strip().ne("")
    )

    component_raw: dict[str, pd.Series] = {}
    component_flags: dict[str, pd.Series] = {}
    for experiment_id in ("E1", "E2", "E3", "E4"):
        component_raw[experiment_id], component_flags[experiment_id] = _fit_isolation_forest(
            matrices[experiment_id]["joint"]
        )
    component_normalized = {
        experiment_id: normalize_scores(scores)
        for experiment_id, scores in component_raw.items()
    }

    experiment_scores: dict[str, pd.Series] = {
        "E1": component_raw["E1"],
        "E2": component_raw["E2"],
        "E3": component_raw["E3"],
        "E4": component_raw["E4"],
    }
    experiment_candidates: dict[str, pd.Series] = {
        "E1": component_flags["E1"],
        "E2": component_flags["E2"],
        "E3": component_flags["E3"],
        "E4": component_flags["E4"],
    }

    e5_temporal = component_normalized["E1"]
    e5_graph = component_normalized["E2"]
    e6_temporal = component_normalized["E1"]
    e6_graph = component_normalized["E3"]
    experiment_scores["E5"] = FUSION_ALPHA * e5_temporal + (1.0 - FUSION_ALPHA) * e5_graph
    experiment_scores["E6"] = FUSION_ALPHA * e6_temporal + (1.0 - FUSION_ALPHA) * e6_graph
    candidate_count = int(math.ceil(CONTAMINATION * expected_size))
    for experiment_id in ("E5", "E6"):
        ranks = experiment_scores[experiment_id].rank(ascending=False, method="first")
        experiment_candidates[experiment_id] = ranks.le(candidate_count)

    component_map = {
        "E5": {"temporal": e5_temporal, "graph": e5_graph},
        "E6": {"temporal": e6_temporal, "graph": e6_graph},
    }
    experiment_frames = []
    for experiment_id in EXPERIMENT_DEFINITIONS:
        frame = _ranked_result(
            cohort["logical_event_id"],
            experiment_scores[experiment_id],
            experiment_candidates[experiment_id],
            metadata,
            experiment_id,
            component_scores=component_map.get(experiment_id),
        )
        experiment_frames.append(frame)
    results = pd.concat(experiment_frames, ignore_index=True)

    if len(results) != expected_size * len(EXPERIMENT_DEFINITIONS):
        raise ValueError("Not all experiments scored the complete common cohort")
    for experiment_id in EXPERIMENT_DEFINITIONS:
        rows = results.loc[results["experiment_id"] == experiment_id]
        if rows["logical_event_id"].nunique() != expected_size:
            raise ValueError(f"{experiment_id} does not score the common cohort exactly once")

    comparisons = compare_experiments(results)
    summaries = _experiment_summaries(results)
    summary: dict[str, object] = {
        "dataset": CASE_ID,
        "cohort_size": expected_size,
        "cohort_identity": "logical_event_id",
        "cohort_logical_event_ids": cohort_ids_sorted,
        "excluded_boundary_event": "LEVT-M57-Jean-PROCESS-812-20091121013230",
        "ground_truth_anomaly_labels_available": False,
        "supervised_classification_metrics_generated": False,
        "controls": {
            "model": "sklearn.ensemble.IsolationForest",
            "n_estimators": N_ESTIMATORS,
            "contamination": CONTAMINATION,
            "random_state": RANDOM_STATE,
            "n_jobs": N_JOBS,
            "scaler": "StandardScaler",
            "fusion_normalization": "min-max per component over the full cohort",
            "fusion_alpha": FUSION_ALPHA,
            "top_k": TOP_K,
        },
        "feature_groups": {
            "A_temporal": list(TEMPORAL_FEATURES),
            "B_graph_structural": list(GRAPH_STRUCTURAL_FEATURES),
            "C_process_artifact_relationship": list(PROCESS_ARTIFACT_FEATURES),
            "D_metadata_identifiers_excluded": list(METADATA_FEATURES),
        },
        "experiments": summaries,
        "retrospective_scope": {
            "events_next_uses_post_event_lookahead": True,
            "graph_metrics_use_complete_reconstructed_graph": True,
            "scaling_or_normalization_uses_complete_cohort": True,
            "interpretation": "retrospective/post-mortem only; not causal, online, real-time, streaming, or deployment-ready",
        },
        "comparisons": comparisons.to_dict(orient="records"),
    }
    report = _format_report(results, comparisons, cohort_ids_sorted)
    return results, comparisons, summary, report


def save_ablation_outputs(
    results: pd.DataFrame,
    comparisons: pd.DataFrame,
    summary: dict[str, object],
    report: str,
    output_dir: Path = RESULTS_DIR,
) -> None:
    """Write the four requested ablation artifacts to the selected directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / ABLATION_RESULTS_PATH.name, index=False, encoding="utf-8")
    comparisons.to_csv(output_dir / ABLATION_COMPARISON_PATH.name, index=False, encoding="utf-8")
    (output_dir / ABLATION_REPORT_PATH.name).write_text(report, encoding="utf-8", newline="\n")
    (output_dir / ABLATION_SUMMARY_PATH.name).write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    temporal_features, graph_features, cohort_ids = load_repository_cohort()
    results, comparisons, summary, report = run_ablation_study(
        temporal_features, graph_features, cohort_ids
    )
    save_ablation_outputs(results, comparisons, summary, report)
    print(f"Ablation results: {ABLATION_RESULTS_PATH}")
    print(f"Ablation comparisons: {ABLATION_COMPARISON_PATH}")
    print(f"Ablation cohort size: {summary['cohort_size']}")


if __name__ == "__main__":
    main()
