from pathlib import Path

import networkx as nx
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results" / "M57-Jean"
NORMALIZED_DIR = PROJECT_ROOT / "data" / "normalized" / "M57-Jean"
MALMEM_PATH = PROJECT_ROOT / "datasets" / "MalMem2022" / "MalMem2022.csv"


def load_csv(path):
    """Load a CSV safely, returning an empty DataFrame if unavailable."""
    path = Path(path)

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path, low_memory=False)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def load_m57_results():
    """Load existing M57-Jean outputs without rerunning detection."""
    filenames = {
        "graph_anomalies": "graph_anomalies.csv",
        "graph_shap": "graph_shap_explanations.csv",
        "graph_evidence": "graph_evidence_attribution.csv",
        "graph_explanations": "graph_investigator_explanations.csv",
        "temporal_anomalies": "isolation_forest_results.csv",
        "temporal_shap": "shap_explanations.csv",
        "temporal_evidence": "evidence_attribution.csv",
        "temporal_explanations": "investigator_explanations.csv",
    }

    results = {
        name: load_csv(RESULTS_DIR / filename)
        for name, filename in filenames.items()
    }

    results["timeline"] = load_csv(
        NORMALIZED_DIR / "logical_timeline.csv"
    )

    results["events"] = load_csv(
        NORMALIZED_DIR / "events.csv"
    )

    return results


def load_process_graph():
    """Load the saved process graph without rebuilding it."""
    graph_path = NORMALIZED_DIR / "temporal_graph.graphml"

    if not graph_path.exists():
        return None

    try:
        return nx.read_graphml(graph_path)
    except (OSError, nx.NetworkXError, ValueError):
        return None


def load_malmem_data():
    """Load MalMem2022 data with its existing project data loader."""
    try:
        from src.data.malmem_data import load_malmem_data as project_loader

        features, target, metadata = project_loader()
        return features, target, metadata, None
    except (FileNotFoundError, ValueError, OSError) as exc:
        return None, None, None, str(exc)


def load_malmem_model_workflow():
    """
    Train and evaluate the existing model only when explicitly requested.
    Returns results without changing the existing model implementation.
    """
    from src.data.malmem_data import split_malmem_data
    from src.models.malmem_model import train_malmem_model
    from src.models.malmem_evaluation import evaluate_malmem_model

    X_train, X_test, y_train, y_test = split_malmem_data(
        test_size=0.2,
        random_state=42,
    )

    model = train_malmem_model(X_train, y_train, random_state=42)
    metrics = evaluate_malmem_model(model, X_test, y_test)

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "metrics": metrics,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }
