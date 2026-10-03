"""Retrospective robustness and sensitivity audit for ForensiXplain Phase 9.

This module writes only new ``phase10_*`` artifacts. Existing Phase 9 and
legacy fusion artifacts are treated as read-only references and are checked
for byte-level stability before and after execution.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import kendalltau
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.evaluation import ablation_study as phase9
from src.evaluation.temporal_graph_fusion import (
    FUSION_ALPHAS as LEGACY_ALPHAS,
    FUSION_N_JOBS,
    GRAPH_AWARE_MODEL_FEATURES,
    build_alpha_rankings,
    normalize_scores,
)
from src.anomaly.graph_features import GRAPH_FEATURE_COLUMNS
from src.anomaly.temporal_isolation_forest import MODEL_FEATURES as LEGACY_TEMPORAL_FEATURES


CASE_ID = "M57-Jean"
RESULTS_DIR = phase9.RESULTS_DIR
FEATURE_DIR = phase9.FEATURE_DIR
RANKINGS_PATH = RESULTS_DIR / "phase10_sensitivity_rankings.csv"
COMPARISONS_PATH = RESULTS_DIR / "phase10_sensitivity_comparisons.csv"
MANIFEST_PATH = RESULTS_DIR / "phase10_artifact_manifest.json"
SUMMARY_PATH = RESULTS_DIR / "phase10_summary.json"
REPORT_PATH = RESULTS_DIR / "phase10_report.txt"

ALPHAS = (0.00, 0.25, 0.50, 0.75, 1.00)
TOP_KS = (3, 5, 10)
BASELINE = (500, 0.10, 42)
CONFIGS = (
    ("baseline", 500, 0.10, 42),
    ("n_estimators_250", 250, 0.10, 42),
    ("n_estimators_1000", 1000, 0.10, 42),
    ("contamination_005", 500, 0.05, 42),
    ("contamination_015", 500, 0.15, 42),
    ("random_state_0", 500, 0.10, 0),
    ("random_state_123", 500, 0.10, 123),
)
PARAMETER_PERTURBATIONS = (
    "n_estimators_250", "n_estimators_1000", "random_state_0", "random_state_123"
)
RQ_PAIRS = (
    ("RQ1", "E1", "E2", "Temporal-only vs graph-structural-only"),
    ("RQ2", "E2", "E3", "Graph-structural vs graph-full"),
    ("RQ3", "E3", "E4", "Graph-full vs graph-aware combined IF"),
    ("RQ4", "E4", "E6", "Graph-aware combined IF vs temporal + graph-full fusion"),
)

PROTECTED_PATHS = (
    RESULTS_DIR / "fused_anomalies.csv",
    RESULTS_DIR / "fusion_report.txt",
    RESULTS_DIR / "fusion_evaluation.csv",
    RESULTS_DIR / "ablation_results.csv",
    RESULTS_DIR / "ablation_comparison.csv",
    RESULTS_DIR / "ablation_report.txt",
    RESULTS_DIR / "ablation_summary.json",
)
INPUT_PATHS = (
    FEATURE_DIR / "temporal_features.csv",
    FEATURE_DIR / "graph_features.csv",
    RESULTS_DIR / "temporal_anomalies.csv",
    RESULTS_DIR / "controlled_graph_comparison.csv",
    *PROTECTED_PATHS,
)
LEAKAGE_DISCLOSURES = (
    "events_next_* temporal features use post-event lookahead",
    "graph metrics use the complete reconstructed graph",
    "StandardScaler and fusion min-max normalization use the full 46-event cohort",
    "scope is retrospective/post-mortem forensic reconstruction",
    "no causal, online, real-time, streaming, or deployment-ready interpretation",
    "metadata, identifiers, provenance, evidence IDs, and model outputs are excluded from fit matrices",
)


@dataclass(frozen=True)
class ModelConfig:
    n_estimators: int
    contamination: float
    random_state: int
    n_jobs: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _snapshot(paths: Iterable[Path]) -> dict[str, str]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required Phase 10 reference files are missing: {missing}")
    return {str(path.relative_to(phase9.PROJECT_ROOT)): _sha256(path) for path in paths}


def _parse_bool(values: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    parsed = values.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if parsed.isna().any():
        raise ValueError(f"Invalid boolean values in {label}")
    return parsed.astype(bool)


def _model_configs(track: str) -> dict[str, ModelConfig]:
    if track == "phase9":
        n_jobs = phase9.N_JOBS
    elif track == "legacy_temporal":
        n_jobs = -1
    elif track in {"legacy_graph_only", "legacy_graph_aware"}:
        n_jobs = FUSION_N_JOBS
    else:
        raise ValueError(f"Unknown model track: {track}")
    return {
        key: ModelConfig(n_estimators, contamination, random_state, n_jobs)
        for key, n_estimators, contamination, random_state in CONFIGS
    }


def _make_matrix(frame: pd.DataFrame, ids: list[str], columns: Iterable[str], label: str) -> pd.DataFrame:
    if "logical_event_id" not in frame:
        raise KeyError(f"{label} is missing logical_event_id")
    local = frame.copy()
    local["logical_event_id"] = local["logical_event_id"].astype(str).str.strip()
    if local["logical_event_id"].duplicated().any():
        raise ValueError(f"{label} contains duplicate logical_event_id values")
    selected = local.loc[local["logical_event_id"].isin(ids)].copy()
    if set(selected["logical_event_id"]) != set(ids) or len(selected) != len(ids):
        raise ValueError(f"{label} does not cover the exact 46-event cohort")
    ordered_columns = list(columns)
    missing = set(ordered_columns) - set(selected.columns)
    if missing:
        raise KeyError(f"{label} is missing model features: {sorted(missing)}")
    selected = selected.set_index("logical_event_id", drop=True)
    selected = selected.loc[ids]
    matrix = selected[ordered_columns].apply(pd.to_numeric, errors="coerce")
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} model matrix contains non-finite values")
    if set(matrix.columns) & set(phase9.METADATA_FEATURES):
        raise ValueError(f"Metadata entered the {label} model matrix")
    return matrix


def _fit(matrix: pd.DataFrame, config: ModelConfig) -> dict[str, Any]:
    values = matrix.to_numpy(dtype=float)
    scaled = StandardScaler().fit_transform(values)
    model = IsolationForest(
        n_estimators=config.n_estimators,
        contamination=config.contamination,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )
    model.fit(scaled)
    scores = -model.decision_function(scaled)
    flags = model.predict(scaled) == -1
    return {
        "scores": pd.Series(scores, index=matrix.index, dtype=float),
        "candidates": pd.Series(flags, index=matrix.index, dtype=bool),
        "rank": _rank(scores, matrix.index.astype(str).tolist(), matrix.index.astype(str).tolist()),
    }


def _rank(scores: Iterable[float], ids: list[str], native_order: list[str] | None = None) -> pd.Series:
    score_series = pd.Series(np.asarray(list(scores), dtype=float), index=ids)
    if not np.isfinite(score_series.to_numpy()).all() or score_series.index.duplicated().any():
        raise ValueError("Cannot rank non-finite scores or duplicate event IDs")
    tie_order = native_order or sorted(ids)
    tie_position = {event_id: ix for ix, event_id in enumerate(tie_order)}
    ordered = sorted(ids, key=lambda event_id: (-float(score_series[event_id]), tie_position[event_id]))
    return pd.Series(np.arange(1, len(ordered) + 1, dtype=int), index=ordered, name="rank")


def _rank_rows(
    track: str,
    family: str,
    condition: str,
    matrix: pd.DataFrame | None,
    result: dict[str, Any],
    config: ModelConfig | None,
    alpha: float | None = None,
    analysis_type: str = "model_fit",
    source: str = "fitted_in_phase10",
) -> list[dict[str, Any]]:
    ids = result["scores"].index.astype(str).tolist()
    features = list(matrix.columns) if matrix is not None else []
    rows = []
    for event_id in ids:
        rows.append({
            "track": track,
            "model_family": family,
            "analysis_type": analysis_type,
            "condition_id": condition,
            "logical_event_id": event_id,
            "score": float(result["scores"].loc[event_id]),
            "rank": int(result["rank"].loc[event_id]),
            "anomaly_candidate": bool(result["candidates"].loc[event_id]),
            "alpha": alpha,
            "n_estimators": config.n_estimators if config else None,
            "contamination": config.contamination if config else None,
            "random_state": config.random_state if config else None,
            "n_jobs": config.n_jobs if config else None,
            "feature_count": len(features),
            "feature_columns": ";".join(features),
            "result_source": source,
        })
    return rows


def _ranked_ids(result: dict[str, Any]) -> list[str]:
    return result["rank"].sort_values(kind="mergesort").index.astype(str).tolist()


def _top_ids(result: dict[str, Any], k: int) -> set[str]:
    return set(_ranked_ids(result)[:k])


def _metric_pair(left: dict[str, Any], right: dict[str, Any], k: int) -> dict[str, Any]:
    left_ids = set(left["scores"].index.astype(str))
    right_ids = set(right["scores"].index.astype(str))
    if left_ids != right_ids or len(left_ids) != phase9.EXPECTED_COHORT_SIZE:
        raise ValueError("Ranking comparison does not use the same 46 logical events")
    left_top = _top_ids(left, k)
    right_top = _top_ids(right, k)
    overlap = left_top & right_top
    union = left_top | right_top
    ids = sorted(left_ids)
    left_order = left["scores"].reindex(ids).to_numpy(dtype=float)
    right_order = right["scores"].reindex(ids).to_numpy(dtype=float)
    tau = kendalltau(left_order, right_order)
    left_candidates = set(left["candidates"].loc[lambda s: s].index.astype(str))
    right_candidates = set(right["candidates"].loc[lambda s: s].index.astype(str))
    candidate_overlap = left_candidates & right_candidates
    candidate_union = left_candidates | right_candidates
    return {
        "top_k": k,
        "top_k_overlap_count": len(overlap),
        "top_k_overlap_event_ids": ";".join(sorted(overlap)),
        "top_k_jaccard": len(overlap) / len(union) if union else 1.0,
        "kendall_tau": float(tau.statistic),
        "kendall_tau_pvalue_descriptive_unadjusted": float(tau.pvalue),
        "candidate_overlap_count": len(candidate_overlap),
        "candidate_overlap_event_ids": ";".join(sorted(candidate_overlap)),
        "candidate_jaccard": len(candidate_overlap) / len(candidate_union) if candidate_union else 1.0,
        "left_candidate_count": len(left_candidates),
        "right_candidate_count": len(right_candidates),
        "cohort_size": len(left_ids),
    }


def _compare(
    comparison_rows: list[dict[str, Any]],
    track: str,
    comparison_type: str,
    condition: str,
    left_name: str,
    left: dict[str, Any],
    right_name: str,
    right: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    for k in TOP_KS:
        comparison_rows.append({
            "track": track,
            "comparison_type": comparison_type,
            "condition_id": condition,
            "left_model": left_name,
            "right_model": right_name,
            **(extra or {}),
            **_metric_pair(left, right, k),
        })
    return comparison_rows


def _result_from_frame(frame: pd.DataFrame, score_col: str, flag_col: str, rank_col: str) -> dict[str, Any]:
    ordered = frame.copy()
    ordered["logical_event_id"] = ordered["logical_event_id"].astype(str)
    if ordered["logical_event_id"].duplicated().any() or len(ordered) != 46:
        raise ValueError("Reference result must have exactly 46 unique logical event IDs")
    scores = pd.to_numeric(ordered[score_col], errors="coerce")
    ranks = pd.to_numeric(ordered[rank_col], errors="coerce")
    if scores.isna().any() or ranks.isna().any() or not np.isfinite(scores).all():
        raise ValueError(f"Invalid scores/ranks in reference column {score_col}")
    ids = ordered["logical_event_id"].astype(str).tolist()
    return {
        "scores": pd.Series(scores.to_numpy(dtype=float), index=ids),
        "candidates": pd.Series(_parse_bool(ordered[flag_col], flag_col).to_numpy(), index=ids),
        "rank": pd.Series(ranks.to_numpy(dtype=int), index=ids),
    }


def _assert_same_result(actual: dict[str, Any], expected: dict[str, Any], label: str, tolerance: float = 1e-12) -> None:
    if set(actual["scores"].index) != set(expected["scores"].index):
        raise AssertionError(f"{label}: event cohort differs from its validated reference")
    ids = sorted(expected["scores"].index)
    if not np.allclose(actual["scores"].reindex(ids), expected["scores"].reindex(ids), rtol=0, atol=tolerance):
        delta = float(np.max(np.abs(actual["scores"].reindex(ids) - expected["scores"].reindex(ids))))
        raise AssertionError(f"{label}: baseline scores differ from validated reference (max delta {delta})")
    if not actual["rank"].reindex(ids).equals(expected["rank"].reindex(ids)):
        raise AssertionError(f"{label}: baseline ranks differ from validated reference")
    if not actual["candidates"].reindex(ids).equals(expected["candidates"].reindex(ids)):
        raise AssertionError(f"{label}: baseline candidate flags differ from validated reference")


def _legacy_fusion_audit(fused: pd.DataFrame, temporal_ref: dict[str, Any], graph_ref: dict[str, Any]) -> tuple[dict[float, dict[str, Any]], list[dict[str, Any]]]:
    base = fused[["logical_event_id", "temporal_score", "graph_score", "temporal_anomaly_rank", "temporal_predicted_anomaly"]].copy()
    ids = base["logical_event_id"].astype(str).tolist()
    base["normalized_temporal_score"] = normalize_scores(pd.Series(pd.to_numeric(base["temporal_score"], errors="raise").to_numpy(), index=ids)).to_numpy()
    base["normalized_graph_score"] = normalize_scores(pd.Series(pd.to_numeric(base["graph_score"], errors="raise").to_numpy(), index=ids)).to_numpy()
    base["temporal_predicted_anomaly"] = _parse_bool(base["temporal_predicted_anomaly"], "legacy temporal flags")
    recomputed = build_alpha_rankings(base, alphas=ALPHAS, contamination=0.10)
    outputs: dict[float, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        ranked = recomputed[alpha]
        actual = {
            "scores": pd.Series(ranked["fusion_score"].to_numpy(dtype=float), index=ranked["logical_event_id"].astype(str)),
            "rank": pd.Series(ranked["fusion_rank"].to_numpy(dtype=int), index=ranked["logical_event_id"].astype(str)),
            "candidates": pd.Series(_parse_bool(ranked["fusion_predicted_anomaly"], "legacy fusion flags").to_numpy(), index=ranked["logical_event_id"].astype(str)),
        }
        suffix = f"{alpha:.2f}".replace(".", "_")
        saved = _result_from_frame(
            fused[["logical_event_id", f"fusion_score_alpha_{suffix}", f"fusion_predicted_anomaly_alpha_{suffix}", f"fusion_rank_alpha_{suffix}"]],
            f"fusion_score_alpha_{suffix}",
            f"fusion_predicted_anomaly_alpha_{suffix}",
            f"fusion_rank_alpha_{suffix}",
        )
        _assert_same_result(actual, saved, f"Legacy fusion alpha={alpha:.2f}", tolerance=1e-12)
        outputs[alpha] = actual
        rows.extend(_rank_rows("legacy", f"LegacyFusion_alpha_{alpha:.2f}", "validated_existing_alphas", None, actual, None, alpha, "legacy_alpha_audit", "recomputed_read_only_and_hash_verified"))
    _assert_endpoint(outputs[0.0], graph_ref, "Legacy alpha=0 graph equivalence")
    _assert_endpoint(outputs[1.0], temporal_ref, "Legacy alpha=1 temporal equivalence")
    return outputs, rows


def _reference_results(ids: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any], pd.DataFrame]:
    ablation = pd.read_csv(phase9.ABLATION_RESULTS_PATH, low_memory=False)
    references = {}
    for exp in ("E1", "E2", "E3", "E4"):
        subset = ablation.loc[ablation["experiment_id"] == exp]
        references[exp] = _result_from_frame(subset, "score", "anomaly_candidate", "rank")
        if set(references[exp]["scores"].index) != set(ids):
            raise ValueError(f"Phase 9 reference {exp} does not use the validated cohort")
    temporal_source = pd.read_csv(RESULTS_DIR / "temporal_anomalies.csv", low_memory=False)
    temporal_source = temporal_source.loc[pd.to_numeric(temporal_source["temporal_anomaly_rank"], errors="coerce").notna()].copy()
    temporal_ref = _result_from_frame(temporal_source, "temporal_anomaly_score", "temporal_predicted_anomaly", "temporal_anomaly_rank")
    controlled = pd.read_csv(RESULTS_DIR / "controlled_graph_comparison.csv", low_memory=False)
    graph_ref = _result_from_frame(controlled, "controlled_graph_only_score", "controlled_graph_only_predicted_anomaly", "controlled_graph_only_rank")
    aware_ref = _result_from_frame(controlled, "controlled_graph_aware_score", "controlled_graph_aware_predicted_anomaly", "controlled_graph_aware_rank")
    fused = pd.read_csv(RESULTS_DIR / "fused_anomalies.csv", low_memory=False)
    for label, result in [("temporal legacy", temporal_ref), ("controlled graph-only legacy", graph_ref), ("controlled graph-aware legacy", aware_ref)]:
        if set(result["scores"].index) != set(ids):
            raise ValueError(f"{label} reference does not use the validated cohort")
    return references, temporal_ref, graph_ref, aware_ref, fused


def _load_matrices(ids: list[str]) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, Any]]:
    temporal_features = pd.read_csv(phase9.TEMPORAL_FEATURES_PATH, low_memory=False)
    graph_features = pd.read_csv(phase9.GRAPH_FEATURES_PATH, low_memory=False)
    cohort, phase_matrices = phase9.build_feature_matrices(temporal_features, graph_features, ids)
    matrices = {exp: components["joint"] for exp, components in phase_matrices.items() if exp in {"E1", "E2", "E3", "E4"}}
    legacy_temporal_source = temporal_features.copy()
    legacy_temporal_source["logical_event_id"] = legacy_temporal_source["logical_event_id"].astype(str)
    legacy_temporal_source = legacy_temporal_source.loc[legacy_temporal_source["logical_event_id"].isin(ids)].copy()
    legacy_temporal_source = legacy_temporal_source.sort_values("temporal_sequence", kind="mergesort")
    legacy_temporal_matrix = _make_matrix(legacy_temporal_source, ids_in_order(legacy_temporal_source), LEGACY_TEMPORAL_FEATURES, "Legacy temporal")
    legacy_graph_matrix = _make_matrix(graph_features, ids, GRAPH_FEATURE_COLUMNS, "Legacy graph-only")
    legacy_aware_matrix = _make_matrix(graph_features, ids, GRAPH_AWARE_MODEL_FEATURES, "Legacy graph-aware")
    # Phase 9 E3 and the legacy controlled graph-only model select the same nine
    # numeric values, but remain separately fitted because the fixed pipelines
    # use different n_jobs settings (-1 versus 1) and belong to distinct tracks.
    if not np.array_equal(matrices["E3"].to_numpy(dtype=float), legacy_graph_matrix.to_numpy(dtype=float)):
        raise AssertionError("Phase 9 E3 and legacy graph-only matrices are not exactly equivalent")
    return matrices, {
        "LegacyTemporal5": legacy_temporal_matrix,
        "LegacyGraphOnly9": legacy_graph_matrix,
        "LegacyGraphAware14": legacy_aware_matrix,
    }, cohort


def ids_in_order(frame: pd.DataFrame) -> list[str]:
    return frame["logical_event_id"].astype(str).tolist()


def _fusion_result(temporal: dict[str, Any], graph: dict[str, Any], alpha: float, contamination: float) -> dict[str, Any]:
    temporal_scaled = normalize_scores(temporal["scores"])
    graph_scaled = normalize_scores(graph["scores"])
    scores = alpha * temporal_scaled + (1.0 - alpha) * graph_scaled
    ids = scores.index.astype(str).tolist()
    if alpha == 0.0:
        native = _ranked_ids(graph)
    elif alpha == 1.0:
        native = _ranked_ids(temporal)
    else:
        native = sorted(ids)
    ranks = _rank(scores.to_numpy(dtype=float), ids, native)
    if alpha == 0.0:
        candidates = graph["candidates"].reindex(ids).astype(bool)
    elif alpha == 1.0:
        candidates = temporal["candidates"].reindex(ids).astype(bool)
    else:
        candidate_count = max(1, int(math.ceil(contamination * len(ids))))
        selected = set(ranks.sort_values(kind="mergesort").index[:candidate_count])
        candidates = pd.Series([event_id in selected for event_id in ids], index=ids, dtype=bool)
    return {"scores": scores, "rank": ranks, "candidates": candidates}


def _endpoint_check(fused: dict[str, Any], component: dict[str, Any], label: str) -> None:
    _assert_endpoint(fused, component, label)


def _assert_endpoint(fused: dict[str, Any], component: dict[str, Any], label: str) -> dict[str, bool]:
    ids = sorted(component["scores"].index.astype(str))
    expected_scores = normalize_scores(component["scores"]).reindex(ids).to_numpy(dtype=float)
    actual_scores = fused["scores"].reindex(ids).to_numpy(dtype=float)
    if not np.allclose(actual_scores, expected_scores, rtol=0, atol=1e-15):
        raise AssertionError(f"{label}: endpoint scores are not normalized-component-equivalent")
    if _ranked_ids(fused) != _ranked_ids(component):
        raise AssertionError(f"{label}: endpoint ranking is not component-equivalent")
    if not fused["candidates"].reindex(ids).equals(component["candidates"].reindex(ids)):
        raise AssertionError(f"{label}: endpoint candidates are not component-equivalent")
    if any(_top_ids(fused, k) != _top_ids(component, k) for k in TOP_KS):
        raise AssertionError(f"{label}: endpoint top-K sets are not component-equivalent")
    return {
        "normalized_component_scores_equal": True,
        "full_rank_order_equal": True,
        "top_k_3_5_10_equal": True,
        "candidate_flags_equal": True,
        "temporal_native_tie_order_preserved": label.startswith("E") and "alpha=1.0" in label,
    }


def _compare_existing_phase9_baseline(actual: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    _assert_same_result(actual, expected, label, tolerance=1e-12)


def _classify(count: int, total: int = 4) -> str:
    if count >= 3:
        return "stable"
    if count <= 1:
        return "sensitive"
    return "mixed"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _build_report(summary: dict[str, Any], comparisons: pd.DataFrame) -> str:
    lines = [
        "ForensiXplain Phase 10: Robustness / Sensitivity Analysis",
        f"Dataset: {CASE_ID}",
        "Scope: retrospective/post-mortem ranking robustness; no detection-validity claim.",
        "",
        "A. Research objective",
        "Assess whether Phase 9 ranking comparisons and temporal-graph fusion rankings persist under the approved alpha and limited one-factor IsolationForest sensitivity settings.",
        "",
        "B. Research questions",
        "RQ1 E1 vs E2; RQ2 E2 vs E3; RQ3 E3 vs E4; RQ4 E4 vs E6 at alpha=0.50.",
        "Fusion sensitivity compares E5/E6 alpha values to alpha=0.50 and checks graph/temporal endpoint equivalence.",
        "",
        "C/D. Exact experiments and settings",
        "Alpha grid: 0.00, 0.25, 0.50, 0.75, 1.00. E5 fuses E1/E2; E6 fuses E1/E3.",
        "IsolationForest one-factor-at-a-time settings: baseline (500, 0.10, 42); n_estimators 250/1000; contamination 0.05/0.15; random_state 0/123.",
        "n_jobs remains fixed within each existing model family: Phase 9 -1; legacy temporal -1; controlled legacy graph models 1.",
        "No full factorial sweep was run. E5/E6 alpha changes reuse the same fitted component scores for a parameter condition.",
        "",
        "E. Metrics and interpretation",
        "Top-K overlap counts and Jaccard at K=3,5,10; Kendall tau across all 46 events; candidate overlap counts, event IDs, and Jaccard.",
        "Kendall tau p-values are descriptive and unadjusted. They do not measure anomaly detection validity and are not used as pass/fail thresholds.",
        "No accuracy, precision, recall, F1, ROC-AUC, or PR-AUC is calculated for M57-Jean.",
        "",
        "F. Cohort",
        f"All analyses use the same {summary['cohort']['size']}-event logical-event cohort, cross-checked against temporal_anomalies.csv and fused_anomalies.csv.",
        f"Excluded boundary event: {summary['cohort']['excluded_boundary_event']}.",
        "Phase 9 E1-E6 and Legacy outputs are separate tracks; their rankings are never combined.",
        "",
        "G. Stability classifications",
    ]
    for family, result in summary["model_parameter_stability"].items():
        lines.append(f"{family}: {result['classification']} ({result['passing_perturbations']}/4 parameter perturbations pass the configured top-5/Jaccard/tau rule).")
    for rq, result in summary["research_question_stability"].items():
        lines.append(f"{rq}: {result['classification']} ({result['passing_perturbations']}/4 non-contamination settings remain within the approved agreement and tau-delta rule).")
    for family, result in summary["alpha_local_stability"].items():
        lines.append(f"{family}: {result['classification']} ({result['passing_neighbors']}/2 neighbors, alpha=.25 and .75, pass the top-5/Jaccard/tau rule).")
    lines.extend([
        "",
        "H. Leakage and retrospective-scope controls",
        *[f"- {item}." for item in LEAKAGE_DISCLOSURES],
        "",
        "I. Outputs",
        "phase10_sensitivity_rankings.csv, phase10_sensitivity_comparisons.csv, phase10_artifact_manifest.json, phase10_summary.json, phase10_report.txt.",
        "",
        "J. Determinism and protected artifacts",
        f"Baseline repeated-fit checks: {'PASS' if summary['determinism']['all_repeats_exact'] else 'FAIL'} across {summary['determinism']['repeat_count']} repeated model fits.",
        f"Protected Phase 9 and legacy artifacts byte/hash stable: {'PASS' if summary['artifact_protection']['all_protected_hashes_unchanged'] else 'FAIL'}.",
        "",
        "K. Computational cost",
        f"{summary['computational_cost']['fit_count']} IsolationForest fits, {summary['computational_cost']['total_trees']:,} trees, elapsed {summary['runtime_seconds']:.2f} seconds.",
        "",
        "L. Scientific risks and mitigations",
        "A single cohort of 46 events gives limited precision. Report agreement as ranking stability only; do not infer maliciousness, accuracy, or causal predictive value. Compare same logical IDs and preserve original feature and scaling semantics.",
        "Score-rank sensitivity and candidate-set stability are reported separately. Contamination changes the fitted decision threshold: absolute decision scores, score ordering, ranks, and model.predict candidate flags are audited empirically.",
        "Fused candidate sets use ceil(contamination*N), except alpha endpoints inherit component flags to preserve endpoint equivalence. Alpha and estimator/seed sensitivity remain ranking-agreement questions, not quality comparisons.",
        "",
        "M. Robustness decision criteria",
        "Model family: stable iff at least 3/4 n_estimators/seed perturbations have top-5 overlap >=4/5, Jaccard >=0.60, tau >=0.80; sensitive iff 0-1 pass; mixed iff 2 pass.",
        "RQ: stable iff at least 3/4 perturbations have top-5 overlap >=4/5 against the Phase 9 baseline and |tau - baseline tau| <=0.15; sensitive iff 0-1; mixed iff 2.",
        "Alpha-local E5/E6: stable iff both .25 and .75 against .50 have overlap >=4/5, Jaccard >=0.60, tau >=0.80; sensitive iff neither; mixed iff one.",
        "Endpoint equivalence is a separate correctness check, not a stability criterion. A stable result means ranking agreement under these tested settings, not anomaly-detection validity.",
        "",
        "Observed contamination behavior",
    ])
    standalone_audits = [value for key, value in summary["contamination_audit"].items() if not key.startswith("fused_")]
    rank_checks = [entry[condition]["score_order_unchanged_from_baseline"] for entry in standalone_audits for condition in ("contamination_005", "contamination_015")]
    absolute_checks = [entry[condition]["absolute_decision_scores_unchanged_from_baseline"] for entry in standalone_audits for condition in ("contamination_005", "contamination_015")]
    candidate_counts = sorted({entry[condition]["candidate_count"] for entry in standalone_audits for condition in ("contamination_005", "contamination_015")})
    lines.append(f"Across {len(standalone_audits)} standalone model families and both contamination changes, rank order was unchanged in {sum(rank_checks)}/{len(rank_checks)} checks; absolute decision-score vectors were unchanged in {sum(absolute_checks)}/{len(absolute_checks)} checks; observed candidate counts were {candidate_counts} (baseline contamination=.10 yielded five flags per family).")
    endpoint_values = list(summary["phase9_endpoint_equivalence"].values())
    lines.append(f"Fusion endpoint checks: normalized scores, full rank order, top-K sets at K=3/5/10, and candidate flags were component-equivalent in {len(endpoint_values)}/{len(endpoint_values)} endpoint/configuration checks; alpha=1 retained temporal native tie order.")
    lines.extend([
        "",
        "Contamination candidate-count audit",
    ])
    for family, result in summary["contamination_audit"].items():
        if family.startswith("fused_"):
            conditions = []
            for condition, alpha_entries in result.items():
                counts = ", ".join(f"alpha={alpha}:{entry['candidate_count']}" for alpha, entry in alpha_entries.items())
                conditions.append(f"{condition} ({counts})")
            lines.append(f"{family}: " + "; ".join(conditions) + ".")
        else:
            counts = ", ".join(
                f"{condition}={entry['candidate_count']} (baseline overlap={entry['candidate_overlap_count']}, Jaccard={entry['candidate_jaccard']:.3f})"
                for condition, entry in result.items()
            )
            lines.append(f"{family}: {counts}.")
    lines += ["", "Detailed per-event rankings and comparisons are in the companion CSV files."]
    return "\n".join(lines) + "\n"


def run_phase10() -> dict[str, Any]:
    """Run the approved Phase 10 analysis, without changing protected artifacts."""
    start = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    protected_before = _snapshot(PROTECTED_PATHS)
    input_hashes_before = _snapshot(INPUT_PATHS)

    temporal_scores = pd.read_csv(phase9.TEMPORAL_RESULTS_PATH, low_memory=False)
    validated_fusion = pd.read_csv(phase9.VALIDATED_FUSION_PATH, low_memory=False)
    ids = phase9.validate_scored_cohort(temporal_scores, validated_fusion)
    if len(ids) != 46 or "LEVT-M57-Jean-PROCESS-812-20091121013230" in ids:
        raise AssertionError("The validated Phase 9 cohort or excluded boundary event changed")
    matrices, legacy_matrices, cohort = _load_matrices(ids)
    references, legacy_temporal_ref, legacy_graph_ref, legacy_aware_ref, fused = _reference_results(ids)

    # The seven fitted families are deliberately kept separate by track. The
    # Phase 9 and controlled legacy graph-only feature matrices are compared
    # exactly, but fits are not shared because their fixed n_jobs settings differ.
    families: dict[str, tuple[str, pd.DataFrame, int]] = {
        "E1": ("phase9", matrices["E1"], phase9.N_JOBS),
        "E2": ("phase9", matrices["E2"], phase9.N_JOBS),
        "E3": ("phase9", matrices["E3"], phase9.N_JOBS),
        "E4": ("phase9", matrices["E4"], phase9.N_JOBS),
        "LegacyTemporal5": ("legacy", legacy_matrices["LegacyTemporal5"], -1),
        "LegacyGraphOnly9": ("legacy", legacy_matrices["LegacyGraphOnly9"], FUSION_N_JOBS),
        "LegacyGraphAware14": ("legacy", legacy_matrices["LegacyGraphAware14"], FUSION_N_JOBS),
    }
    fitted: dict[str, dict[str, dict[str, Any]]] = {}
    ranking_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    repeat_checks: list[dict[str, Any]] = []
    fit_count = 0
    tree_count = 0

    for family, (track, matrix, n_jobs) in families.items():
        fitted[family] = {}
        for condition, n_estimators, contamination, random_state in CONFIGS:
            config = ModelConfig(n_estimators, contamination, random_state, n_jobs)
            result = _fit(matrix, config)
            fitted[family][condition] = result
            fit_count += 1
            tree_count += n_estimators
            ranking_rows.extend(_rank_rows(track, family, condition, matrix, result, config))
            if condition == "baseline":
                repeated = _fit(matrix, config)
                fit_count += 1
                tree_count += n_estimators
                exact_scores = np.array_equal(result["scores"].to_numpy(), repeated["scores"].reindex(result["scores"].index).to_numpy())
                exact_ranks = result["rank"].reindex(sorted(ids)).equals(repeated["rank"].reindex(sorted(ids)))
                exact_flags = result["candidates"].reindex(sorted(ids)).equals(repeated["candidates"].reindex(sorted(ids)))
                repeated_metrics = [_metric_pair(result, repeated, k) for k in TOP_KS]
                metrics_exact = repeated_metrics == [_metric_pair(result, result, k) for k in TOP_KS]
                cohort_exact = set(result["scores"].index) == set(repeated["scores"].index) and len(result["scores"]) == 46
                repeat_checks.append({"model_family": family, "scores_exact": bool(exact_scores), "ranks_exact": bool(exact_ranks), "candidate_flags_exact": bool(exact_flags), "cohort_ids_exact": bool(cohort_exact), "comparison_metrics_exact_at_k_3_5_10": bool(metrics_exact)})
                if not (exact_scores and exact_ranks and exact_flags and cohort_exact and metrics_exact):
                    raise AssertionError(f"Repeated baseline fit was not deterministic: {family}")
        # Validate all seven families against their original controlled outputs.
        refs = {"E1": references["E1"], "E2": references["E2"], "E3": references["E3"], "E4": references["E4"], "LegacyTemporal5": legacy_temporal_ref, "LegacyGraphOnly9": legacy_graph_ref, "LegacyGraphAware14": legacy_aware_ref}
        _compare_existing_phase9_baseline(fitted[family]["baseline"], refs[family], f"{family} baseline reproduction")

    # Generate all Phase 9 fused alpha conditions from the matching fitted
    # temporal and graph component scores; no extra model fits occur here.
    phase9_fusions: dict[tuple[str, str, float], dict[str, Any]] = {}
    legacy_alpha_results, legacy_alpha_rows = _legacy_fusion_audit(fused, legacy_temporal_ref, legacy_graph_ref)
    ranking_rows.extend(legacy_alpha_rows)
    legacy_norm_temporal = normalize_scores(legacy_temporal_ref["scores"])
    legacy_norm_graph = normalize_scores(legacy_graph_ref["scores"])
    for alpha in ALPHAS:
        # Existing Legacy Alpha grid has already been audited above. Save a
        # second explicit row family for controlled-fit provenance only if the
        # alpha=0.5 scores line up with the saved historical fusion baseline.
        expected_scores = legacy_alpha_results[alpha]["scores"]
        if alpha == 0.50:
            raw = alpha * legacy_norm_temporal + (1.0 - alpha) * legacy_norm_graph
            if not np.allclose(raw.reindex(sorted(ids)), expected_scores.reindex(sorted(ids)), rtol=0, atol=1e-12):
                raise AssertionError("Legacy alpha=.50 formula audit failed")

    for condition, _, contamination, _ in CONFIGS:
        temporal_e1 = fitted["E1"][condition]
        for exp, graph_family in (("E5", "E2"), ("E6", "E3")):
            graph_component = fitted[graph_family][condition]
            for alpha in ALPHAS:
                fused_result = _fusion_result(temporal_e1, graph_component, alpha, contamination)
                phase9_fusions[(exp, condition, alpha)] = fused_result
                ranking_rows.extend(_rank_rows("phase9", exp, condition, None, fused_result, _model_configs("phase9")[condition], alpha, "score_level_fusion", "computed_from_same_condition_component_fits"))
                if alpha == 0.0:
                    _endpoint_check(fused_result, graph_component, f"{exp} alpha=0 condition={condition}")
                if alpha == 1.0:
                    _endpoint_check(fused_result, temporal_e1, f"{exp} alpha=1 condition={condition}")

    # Audit that the baseline alpha=.50 fused results reproduce Phase 9 E5/E6.
    for exp in ("E5", "E6"):
        _compare_existing_phase9_baseline(phase9_fusions[(exp, "baseline", 0.50)], references[exp] if exp in references else _result_from_frame(pd.read_csv(phase9.ABLATION_RESULTS_PATH).loc[lambda f: f.experiment_id == exp], "score", "anomaly_candidate", "rank"), f"{exp} alpha=.50 Phase 9 reproduction")

    # RQ comparisons are recalculated per parameter setting, with E6 fixed at .50.
    rq_baseline_tau: dict[str, float] = {}
    rq_param_passes: dict[str, list[bool]] = {rq: [] for rq, *_ in RQ_PAIRS}
    for condition, *_ in CONFIGS:
        components = {**{k: fitted[k][condition] for k in ("E1", "E2", "E3", "E4")}, "E5": phase9_fusions[("E5", condition, 0.50)], "E6": phase9_fusions[("E6", condition, 0.50)]}
        for rq, left_name, right_name, description in RQ_PAIRS:
            pair_metrics = _metric_pair(components[left_name], components[right_name], 5)
            for k in TOP_KS:
                comparison_rows.append({
                    "track": "phase9", "comparison_type": "phase9_research_question", "condition_id": condition,
                    "research_question": rq, "left_model": left_name, "right_model": right_name,
                    "question": description, **_metric_pair(components[left_name], components[right_name], k),
                })
            if condition == "baseline":
                rq_baseline_tau[rq] = pair_metrics["kendall_tau"]
            elif condition in PARAMETER_PERTURBATIONS:
                rq_param_passes[rq].append(
                    pair_metrics["top_k_overlap_count"] >= 4
                    and abs(pair_metrics["kendall_tau"] - rq_baseline_tau[rq]) <= 0.15
                )

    # Model-family sensitivity uses only estimator-count/seed perturbations;
    # contamination is analyzed separately because it changes thresholds.
    model_stability: dict[str, Any] = {}
    for family in families:
        passing = []
        for condition in PARAMETER_PERTURBATIONS:
            metrics = _metric_pair(fitted[family]["baseline"], fitted[family][condition], 5)
            passed = metrics["top_k_overlap_count"] >= 4 and metrics["top_k_jaccard"] >= 0.60 and metrics["kendall_tau"] >= 0.80
            passing.append(bool(passed))
            comparison_rows.append({"track": families[family][0], "comparison_type": "model_parameter_sensitivity", "condition_id": condition, "left_model": f"{family}:baseline", "right_model": family, "parameter_family": family, "passes_stability_rule": bool(passed), **metrics})
        model_stability[family] = {"classification": _classify(sum(passing)), "passing_perturbations": int(sum(passing)), "total_perturbations": 4}

    rq_stability = {}
    for rq, passes in rq_param_passes.items():
        rq_stability[rq] = {"classification": _classify(sum(passes)), "passing_perturbations": int(sum(passes)), "total_perturbations": 4}

    alpha_stability: dict[str, Any] = {}
    for exp in ("E5", "E6"):
        pass_count = 0
        for condition, *_ in CONFIGS:
            for alpha in (0.25, 0.75):
                metrics = _metric_pair(phase9_fusions[(exp, condition, 0.50)], phase9_fusions[(exp, condition, alpha)], 5)
                passed = metrics["top_k_overlap_count"] >= 4 and metrics["top_k_jaccard"] >= 0.60 and metrics["kendall_tau"] >= 0.80
                comparison_rows.append({"track": "phase9", "comparison_type": "alpha_local_sensitivity", "condition_id": condition, "left_model": f"{exp}:alpha=.50", "right_model": f"{exp}:alpha={alpha:.2f}", "experiment": exp, "alpha": alpha, "passes_stability_rule": bool(passed), **metrics})
                if condition == "baseline":
                    pass_count += int(passed)
        alpha_stability[exp] = {"classification": "stable" if pass_count == 2 else "sensitive" if pass_count == 0 else "mixed", "passing_neighbors": pass_count, "total_neighbors": 2}

    # Also report alpha pair metrics at all K and parameter settings, plus
    # endpoint equivalence as explicit checks rather than stability claims.
    endpoint_checks: dict[str, Any] = {}
    for condition, *_ in CONFIGS:
        for exp, graph_family in (("E5", "E2"), ("E6", "E3")):
            for alpha in (0.25, 0.50, 0.75, 0.00, 1.00):
                if alpha in (0.25, 0.75):
                    _compare(comparison_rows, "phase9", "alpha_vs_050", condition, f"{exp}:alpha=.50", phase9_fusions[(exp, condition, 0.50)], f"{exp}:alpha={alpha:.2f}", phase9_fusions[(exp, condition, alpha)], {"experiment": exp, "alpha": alpha})
            for alpha, component_name in ((0.0, graph_family), (1.0, "E1")):
                _endpoint_check(phase9_fusions[(exp, condition, alpha)], fitted[component_name][condition], f"{exp} alpha={alpha}")
                endpoint_checks[f"{exp}_{condition}_alpha_{alpha:.2f}"] = _assert_endpoint(
                    phase9_fusions[(exp, condition, alpha)],
                    fitted[component_name][condition],
                    f"{exp} alpha={alpha} condition={condition}",
                )

    # Contamination audits preserve model.predict flags for fitted forests and
    # record ceil(contamination*N) candidate counts for intermediate fusion.
    contamination_audit: dict[str, Any] = {}
    for family in families:
        baseline = fitted[family]["baseline"]
        family_audit = {}
        for condition in ("contamination_005", "contamination_015"):
            result = fitted[family][condition]
            same_scores = np.array_equal(baseline["scores"].reindex(sorted(ids)).to_numpy(), result["scores"].reindex(sorted(ids)).to_numpy())
            same_ranks = baseline["rank"].reindex(sorted(ids)).equals(result["rank"].reindex(sorted(ids)))
            baseline_candidates = set(baseline["candidates"].loc[lambda s: s].index.astype(str))
            changed_candidates = set(result["candidates"].loc[lambda s: s].index.astype(str))
            shared_candidates = baseline_candidates & changed_candidates
            candidate_union = baseline_candidates | changed_candidates
            family_audit[condition] = {
                "candidate_count": int(result["candidates"].sum()),
                "candidate_event_ids": sorted(result["candidates"].loc[lambda s: s].index.astype(str).tolist()),
                "candidate_overlap_count": len(shared_candidates),
                "candidate_overlap_event_ids": sorted(shared_candidates),
                "candidate_jaccard": len(shared_candidates) / len(candidate_union) if candidate_union else 1.0,
                "score_order_unchanged_from_baseline": bool(same_ranks),
                "absolute_decision_scores_unchanged_from_baseline": bool(same_scores),
                "rank_order_unchanged_from_baseline": bool(same_ranks),
            }
            comparison_rows.append({"track": families[family][0], "comparison_type": "contamination_candidate_audit", "condition_id": condition, "left_model": f"{family}:baseline", "right_model": family, "parameter_family": family, "absolute_score_vector_unchanged": bool(same_scores), "score_order_unchanged": bool(same_ranks), "rank_order_unchanged": bool(same_ranks), "candidate_count_delta": int(result["candidates"].sum()) - int(baseline["candidates"].sum()), "candidate_overlap_count": len(shared_candidates), "candidate_overlap_event_ids": ";".join(sorted(shared_candidates)), "candidate_jaccard": len(shared_candidates) / len(candidate_union) if candidate_union else 1.0, "cohort_size": 46})
        contamination_audit[family] = family_audit
    for exp in ("E5", "E6"):
        fused_condition_audit: dict[str, Any] = {}
        for condition, _, contamination, _ in CONFIGS:
            alpha_audit: dict[str, Any] = {}
            for alpha in ALPHAS:
                result = phase9_fusions[(exp, condition, alpha)]
                observed_count = int(result["candidates"].sum())
                expected_count = max(1, int(math.ceil(contamination * 46)))
                if alpha not in (0.0, 1.0) and observed_count != expected_count:
                    raise AssertionError(f"{exp}/{condition}/alpha={alpha}: unexpected fused candidate count")
                alpha_audit[f"{alpha:.2f}"] = {
                    "candidate_count": observed_count,
                    "candidate_event_ids": sorted(result["candidates"].loc[lambda s: s].index.astype(str).tolist()),
                    "expected_top_ceil_count": expected_count,
                    "endpoint_inherits_component_flags": alpha in (0.0, 1.0),
                }
                comparison_rows.append({"track": "phase9", "comparison_type": "fused_candidate_audit", "condition_id": condition, "left_model": exp, "right_model": exp, "experiment": exp, "alpha": alpha, "candidate_count": observed_count, "expected_top_ceil_count": expected_count, "candidate_event_ids": ";".join(alpha_audit[f"{alpha:.2f}"]["candidate_event_ids"]), "cohort_size": 46})
            fused_condition_audit[condition] = alpha_audit
        contamination_audit[f"fused_{exp}"] = fused_condition_audit

    # Legacy alpha ranking agreement is reported separately from Phase 9.
    for alpha in ALPHAS:
        if alpha == 0.50:
            continue
        for k in TOP_KS:
            comparison_rows.append({"track": "legacy", "comparison_type": "legacy_validated_alpha_vs_050", "condition_id": "read_only_existing_fusion", "left_model": "LegacyFusion_alpha_0.50", "right_model": f"LegacyFusion_alpha_{alpha:.2f}", "alpha": alpha, **_metric_pair(legacy_alpha_results[0.50], legacy_alpha_results[alpha], k)})

    # Add one record per Phase 9 and legacy sensitivity family, with all metrics.
    comparisons = pd.DataFrame(comparison_rows)
    rankings = pd.DataFrame(ranking_rows)
    if rankings["logical_event_id"].isna().any() or (rankings.groupby(["track", "model_family", "condition_id", "alpha"], dropna=False).size().mod(46) != 0).any():
        raise AssertionError("Ranking rows do not form complete event cohorts")
    if any(metric.lower() in {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"} for metric in comparisons.columns):
        raise AssertionError("Supervised metrics are prohibited for this cohort")

    # Hash guard before writing any new artifacts.
    protected_after_analysis = _snapshot(PROTECTED_PATHS)
    if protected_after_analysis != protected_before:
        raise RuntimeError("A protected Phase 9/legacy artifact changed during analysis")

    elapsed = time.perf_counter() - start
    legacy_alpha_status = {
        "grid": list(ALPHAS),
        "source_read_only": True,
        "recomputed_scores_ranks_candidates_match_saved_outputs": True,
        "alpha_0_graph_component_equivalent": True,
        "alpha_1_temporal_component_equivalent_and_native_ties_preserved": True,
    }
    summary: dict[str, Any] = {
        "project": "ForensiXplain",
        "phase": 10,
        "analysis": "retrospective robustness and sensitivity",
        "dataset": CASE_ID,
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "cohort": {"size": len(ids), "logical_event_ids": ids, "excluded_boundary_event": "LEVT-M57-Jean-PROCESS-812-20091121013230", "shared_across_all_comparisons": True},
        "feature_groups": {"A_temporal": {"count": 16, "columns": list(phase9.TEMPORAL_FEATURES)}, "B_graph_structural": {"count": 5, "columns": list(phase9.GRAPH_STRUCTURAL_FEATURES)}, "B_plus_C_graph_full": {"count": 9, "columns": list(phase9.GRAPH_FULL_FEATURES)}, "A_plus_B_plus_C": {"count": 25, "columns": list(phase9.ALL_MODEL_FEATURES)}, "legacy_temporal": {"count": 5, "columns": list(LEGACY_TEMPORAL_FEATURES)}, "legacy_graph_only": {"count": 9, "columns": list(GRAPH_FEATURE_COLUMNS)}, "legacy_graph_aware": {"count": 14, "columns": list(GRAPH_AWARE_MODEL_FEATURES)}},
        "parameter_grid": {"alphas": list(ALPHAS), "configs": [{"condition_id": key, "n_estimators": n, "contamination": c, "random_state": seed} for key, n, c, seed in CONFIGS], "n_jobs_by_track": {"phase9": phase9.N_JOBS, "legacy_temporal": -1, "legacy_controlled_graph_models": FUSION_N_JOBS}, "one_factor_at_a_time": True},
        "legacy_alpha_audit": legacy_alpha_status,
        "phase9_endpoint_equivalence": endpoint_checks,
        "model_parameter_stability": model_stability,
        "research_question_stability": rq_stability,
        "alpha_local_stability": alpha_stability,
        "contamination_audit": contamination_audit,
        "determinism": {"all_repeats_exact": all(r["scores_exact"] and r["ranks_exact"] and r["candidate_flags_exact"] and r["cohort_ids_exact"] and r["comparison_metrics_exact_at_k_3_5_10"] for r in repeat_checks), "repeat_count": len(repeat_checks), "checks": repeat_checks},
        "computational_cost": {"fit_count": fit_count, "total_trees": tree_count, "base_families": len(families), "sensitivity_configurations_per_family": len(CONFIGS)},
        "runtime_seconds": round(elapsed, 6),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "statistical_interpretation": {"kendall_p_values": "descriptive, unadjusted, not used for acceptance or evidence of detection quality", "small_sample": "n=46; rankings agreement only", "supervised_metrics_calculated": False},
        "retrospective_leakage_disclosures": list(LEAKAGE_DISCLOSURES),
        "artifact_protection": {"all_protected_hashes_unchanged": True, "protected_hashes_before": protected_before, "protected_hashes_after_analysis": protected_after_analysis},
        "output_files": [str(path.relative_to(phase9.PROJECT_ROOT)) for path in (RANKINGS_PATH, COMPARISONS_PATH, MANIFEST_PATH, SUMMARY_PATH, REPORT_PATH)],
    }
    report = _build_report(summary, comparisons)
    rankings.to_csv(RANKINGS_PATH, index=False, encoding="utf-8")
    comparisons.to_csv(COMPARISONS_PATH, index=False, encoding="utf-8")
    _write_json(SUMMARY_PATH, summary)
    REPORT_PATH.write_text(report, encoding="utf-8")
    protected_after_write = _snapshot(PROTECTED_PATHS)
    if protected_after_write != protected_before:
        raise RuntimeError("Protected Phase 9/legacy artifact hash changed while writing Phase 10 outputs")

    # Include analysis, CSV serialization, and report/summary serialization in
    # the measured runtime recorded by the human-readable artifacts.
    summary["runtime_seconds"] = round(time.perf_counter() - start, 6)
    report = _build_report(summary, comparisons)
    _write_json(SUMMARY_PATH, summary)
    REPORT_PATH.write_text(report, encoding="utf-8")
    protected_after_write = _snapshot(PROTECTED_PATHS)
    if protected_after_write != protected_before:
        raise RuntimeError("Protected artifact hash changed while finalizing Phase 10 outputs")

    output_hashes = {
        str(path.relative_to(phase9.PROJECT_ROOT)): _sha256(path)
        for path in (RANKINGS_PATH, COMPARISONS_PATH, SUMMARY_PATH, REPORT_PATH)
    }
    manifest = {
        "manifest_version": 1,
        "phase": 10,
        "dataset": CASE_ID,
        "cohort_size": len(ids),
        "logical_event_id_definition": "logical_event_id is the unique event identity used for all cohort alignment and ranking comparisons",
        "cohort_logical_event_ids": ids,
        "excluded_boundary_event": "LEVT-M57-Jean-PROCESS-812-20091121013230",
        "feature_groups": summary["feature_groups"],
        "estimator_settings": {
            "baseline": {"n_estimators": 500, "contamination": 0.10, "random_state": 42},
            "n_estimators_grid": [250, 500, 1000],
            "contamination_grid": [0.05, 0.10, 0.15],
            "random_state_grid": [0, 42, 123],
            "n_jobs_fixed_by_track": summary["parameter_grid"]["n_jobs_by_track"],
            "configurations": summary["parameter_grid"]["configs"],
            "one_factor_at_a_time": True,
        },
        "alpha_grid": list(ALPHAS),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "input_sha256": input_hashes_before,
        "protected_artifact_sha256_before": protected_before,
        "protected_artifact_sha256_after": protected_after_write,
        "protected_artifacts_byte_stable": protected_before == protected_after_write,
        "phase10_output_sha256": output_hashes,
        "runtime_seconds": summary["runtime_seconds"],
        "deterministic_repeat_checks_passed": summary["determinism"]["all_repeats_exact"],
        "feature_matrix_equivalence": {"phase9_E3_vs_legacy_graph_only9_exact_values": True, "fits_shared": False, "reason": "separate track and fixed n_jobs differ (-1 versus 1)"},
        "leakage_disclosures": list(LEAKAGE_DISCLOSURES),
    }
    _write_json(MANIFEST_PATH, manifest)
    if _snapshot(PROTECTED_PATHS) != protected_before:
        raise RuntimeError("Protected artifact hash changed after manifest write")
    return {"summary": summary, "manifest": manifest, "report": report, "rankings_rows": len(rankings), "comparison_rows": len(comparisons)}


def main() -> None:
    result = run_phase10()
    print(json.dumps({
        "runtime_seconds": result["summary"]["runtime_seconds"],
        "fit_count": result["summary"]["computational_cost"]["fit_count"],
        "trees": result["summary"]["computational_cost"]["total_trees"],
        "rankings_rows": result["rankings_rows"],
        "comparison_rows": result["comparison_rows"],
        "outputs": result["summary"]["output_files"],
    }, indent=2))


if __name__ == "__main__":
    main()
