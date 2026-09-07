"""Run the active ForensiXplain M57-Jean pipeline from repository root."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from pipeline_config import (
    EVENTS_FILE,
    EVALUATION_CANDIDATES_FILE,
    EVALUATION_REPORT_FILE,
    EVALUATION_SUMMARY_FILE,
    GRAPH_ANOMALIES_FILE,
    GRAPH_EVIDENCE_FILE,
    GRAPH_EXPLANATIONS_FILE,
    GRAPH_FEATURES_FILE,
    GRAPH_ONLY_ANOMALIES_FILE,
    GRAPH_ONLY_EVIDENCE_FILE,
    GRAPH_ONLY_SHAP_FILE,
    GRAPH_REPORT_FILE,
    GRAPH_SHAP_FILE,
    LOGICAL_TIMELINE_FILE,
    MODEL_COMPARISON_FILE,
    MODEL_COMPARISON_REPORT_FILE,
    PROJECT_ROOT,
    TEMPORAL_ANOMALIES_FILE,
    TEMPORAL_EVIDENCE_FILE,
    TEMPORAL_EXPLANATIONS_FILE,
    TEMPORAL_FEATURES_FILE,
    TEMPORAL_GRAPH_FILE,
    TEMPORAL_REPORT_FILE,
    TEMPORAL_SHAP_FILE,
    TIMELINE_FILE,
    UNTIMESTAMPED_EVENTS_FILE,
)
from src.anomaly import graph_features
from src.anomaly import graph_isolation_forest
from src.anomaly import graph_only_isolation_forest
from src.anomaly import temporal_isolation_forest
from src.explainability import graph_evidence_attribution
from src.explainability import graph_explanation_generator
from src.explainability import graph_only_evidence_attribution
from src.explainability import graph_only_shap
from src.explainability import graph_shap
from src.explainability import temporal_evidence_attribution
from src.explainability import temporal_explanation_generator
from src.explainability import temporal_shap
from src.graph import temporal_graph
from src.reporting import evaluation_report
from src.reporting import model_comparison
from src.temporal import logical_timeline
from src.temporal import temporal_features
from src.timeline import timeline_builder


@dataclass(frozen=True)
class PipelineStage:
    """One executable stage and its required artifacts."""

    name: str
    run: Callable[[], None]
    inputs: Sequence[Path]
    outputs: Sequence[Path]


STAGES = (
    PipelineStage(
        "Build timestamped timeline",
        timeline_builder.main,
        (EVENTS_FILE,),
        (TIMELINE_FILE, UNTIMESTAMPED_EVENTS_FILE),
    ),
    PipelineStage(
        "Build logical timeline",
        logical_timeline.main,
        (TIMELINE_FILE,),
        (LOGICAL_TIMELINE_FILE,),
    ),
    PipelineStage(
        "Build temporal features",
        temporal_features.main,
        (LOGICAL_TIMELINE_FILE,),
        (TEMPORAL_FEATURES_FILE,),
    ),
    PipelineStage(
        "Run temporal Isolation Forest",
        temporal_isolation_forest.main,
        (TEMPORAL_FEATURES_FILE,),
        (TEMPORAL_ANOMALIES_FILE,),
    ),
    PipelineStage(
        "Generate temporal SHAP explanations",
        temporal_shap.main,
        (TEMPORAL_FEATURES_FILE, TEMPORAL_ANOMALIES_FILE),
        (TEMPORAL_SHAP_FILE,),
    ),
    PipelineStage(
        "Attribute temporal evidence",
        temporal_evidence_attribution.main,
        (
            EVENTS_FILE,
            LOGICAL_TIMELINE_FILE,
            TEMPORAL_FEATURES_FILE,
            TEMPORAL_SHAP_FILE,
        ),
        (TEMPORAL_EVIDENCE_FILE,),
    ),
    PipelineStage(
        "Generate temporal investigator explanations",
        temporal_explanation_generator.generate_explanations,
        (TEMPORAL_EVIDENCE_FILE, TEMPORAL_SHAP_FILE),
        (TEMPORAL_EXPLANATIONS_FILE, TEMPORAL_REPORT_FILE),
    ),
    PipelineStage(
        "Build temporal forensic graph",
        temporal_graph.main,
        (EVENTS_FILE,),
        (TEMPORAL_GRAPH_FILE,),
    ),
    PipelineStage(
        "Build graph features",
        graph_features.main,
        (TEMPORAL_GRAPH_FILE, TEMPORAL_FEATURES_FILE),
        (GRAPH_FEATURES_FILE,),
    ),
    PipelineStage(
        "Run graph-aware Isolation Forest",
        graph_isolation_forest.main,
        (GRAPH_FEATURES_FILE,),
        (GRAPH_ANOMALIES_FILE,),
    ),
    PipelineStage(
        "Generate graph-aware SHAP explanations",
        graph_shap.main,
        (GRAPH_FEATURES_FILE, GRAPH_ANOMALIES_FILE),
        (GRAPH_SHAP_FILE,),
    ),
    PipelineStage(
        "Attribute graph-aware evidence",
        graph_evidence_attribution.main,
        (
            EVENTS_FILE,
            LOGICAL_TIMELINE_FILE,
            GRAPH_FEATURES_FILE,
            GRAPH_ANOMALIES_FILE,
            GRAPH_SHAP_FILE,
        ),
        (GRAPH_EVIDENCE_FILE,),
    ),
    PipelineStage(
        "Generate graph-aware investigator explanations",
        graph_explanation_generator.main,
        (GRAPH_EVIDENCE_FILE,),
        (GRAPH_EXPLANATIONS_FILE, GRAPH_REPORT_FILE),
    ),
    PipelineStage(
        "Run graph-only Isolation Forest",
        graph_only_isolation_forest.main,
        (GRAPH_FEATURES_FILE,),
        (GRAPH_ONLY_ANOMALIES_FILE,),
    ),
    PipelineStage(
        "Generate graph-only SHAP explanations",
        graph_only_shap.main,
        (GRAPH_FEATURES_FILE, GRAPH_ONLY_ANOMALIES_FILE),
        (GRAPH_ONLY_SHAP_FILE,),
    ),
    PipelineStage(
        "Attribute graph-only evidence",
        graph_only_evidence_attribution.main,
        (
            EVENTS_FILE,
            LOGICAL_TIMELINE_FILE,
            GRAPH_FEATURES_FILE,
            GRAPH_ONLY_ANOMALIES_FILE,
            GRAPH_ONLY_SHAP_FILE,
        ),
        (GRAPH_ONLY_EVIDENCE_FILE,),
    ),
    PipelineStage(
        "Compare anomaly models",
        model_comparison.main,
        (
            TEMPORAL_ANOMALIES_FILE,
            GRAPH_ANOMALIES_FILE,
            GRAPH_ONLY_ANOMALIES_FILE,
        ),
        (MODEL_COMPARISON_FILE, MODEL_COMPARISON_REPORT_FILE),
    ),
    PipelineStage(
        "Generate final evaluation",
        evaluation_report.main,
        (
            EVENTS_FILE,
            TEMPORAL_ANOMALIES_FILE,
            GRAPH_ANOMALIES_FILE,
            GRAPH_ONLY_ANOMALIES_FILE,
            MODEL_COMPARISON_FILE,
            TEMPORAL_EXPLANATIONS_FILE,
            GRAPH_EXPLANATIONS_FILE,
            GRAPH_ONLY_EVIDENCE_FILE,
        ),
        (
            EVALUATION_SUMMARY_FILE,
            EVALUATION_CANDIDATES_FILE,
            EVALUATION_REPORT_FILE,
        ),
    ),
)


def verify_files(paths: Sequence[Path], artifact_type: str, stage_name: str) -> None:
    """Raise a clear error for missing stage artifacts."""

    missing = [path for path in paths if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"{stage_name}: missing required {artifact_type}:\n{formatted}"
        )


def run_stage(stage: PipelineStage) -> None:
    """Validate, run, and validate one pipeline stage."""

    print(f"\n=== Stage: {stage.name} ===")
    verify_files(stage.inputs, "input file(s)", stage.name)
    stage.run()
    verify_files(stage.outputs, "output file(s)", stage.name)
    print(f"=== Completed: {stage.name} ===")


def main() -> None:
    os.chdir(PROJECT_ROOT)
    print("=== ForensiXplain Active Pipeline ===")
    print(f"Repository root: {PROJECT_ROOT}")

    for stage in STAGES:
        run_stage(stage)

    print("\n=== Pipeline completed successfully ===")


if __name__ == "__main__":
    main()
