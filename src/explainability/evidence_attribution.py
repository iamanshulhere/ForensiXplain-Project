from pathlib import Path

import pandas as pd


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

EVENTS_PATH = (
    BASE_DIR
    / "data"
    / "normalized"
    / "M57-Jean"
    / "events.csv"
)

SHAP_PATH = (
    BASE_DIR
    / "results"
    / "M57-Jean"
    / "shap_explanations.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "M57-Jean"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "evidence_attribution.csv"
)


# =========================================================
# Helpers
# =========================================================

def safe_int(value):
    """Safely convert a value to integer."""
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


# =========================================================
# Main
# =========================================================

def main():

    print("=== ForensiXplain Evidence Attribution ===")

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    events = pd.read_csv(EVENTS_PATH)
    shap_df = pd.read_csv(SHAP_PATH)

    print(f"Events: {len(events)}")
    print(f"SHAP rows: {len(shap_df)}")

    # -----------------------------------------------------
    # Normalize process IDs
    # -----------------------------------------------------

    events["process_id_int"] = (
        events["process_id"]
        .apply(safe_int)
    )

    events["parent_process_id_int"] = (
        events["parent_process_id"]
        .apply(safe_int)
    )

    shap_df["process_id_int"] = (
        shap_df["process_id"]
        .apply(safe_int)
    )

    # -----------------------------------------------------
    # Select anomalous processes
    # -----------------------------------------------------

    anomalies = shap_df[
        shap_df["predicted_anomaly"] == 1
    ].copy()

    anomalies = anomalies.sort_values(
        "anomaly_score",
        ascending=False,
    )

    print(
        f"Anomalous processes: {len(anomalies)}"
    )

    # -----------------------------------------------------
    # Create evidence records
    # -----------------------------------------------------

    records = []

    for _, anomaly in anomalies.iterrows():

        process_id = safe_int(
            anomaly["process_id"]
        )

        process_name = anomaly["process_name"]

        # -------------------------------------------------
        # Find all events associated with this PID
        # -------------------------------------------------

        process_events = events[
            events["process_id_int"] == process_id
        ].copy()

        # -------------------------------------------------
        # Artifact counts
        # -------------------------------------------------

        artifact_counts = (
            process_events["artifact_type"]
            .value_counts()
            .to_dict()
        )

        # -------------------------------------------------
        # Evidence IDs
        # -------------------------------------------------

        evidence_ids = (
            process_events["evidence_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        # -------------------------------------------------
        # Event types
        # -------------------------------------------------

        event_types = (
            process_events["event_type"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        # -------------------------------------------------
        # Parent processes
        #
        # Current PID is the CHILD.
        # Therefore find relationship events where:
        #
        # process_id == current PID
        #
        # and collect parent_process_id.
        # -------------------------------------------------

        relationship_events = events[
            events["event_type"]
            == "process_relationship"
        ].copy()

        parent_ids = (
            relationship_events[
                relationship_events["process_id_int"]
                == process_id
            ]["parent_process_id_int"]
            .dropna()
            .astype(int)
            .astype(str)
            .unique()
            .tolist()
        )

        # -------------------------------------------------
        # Child processes
        #
        # Current PID is the PARENT.
        # Therefore find relationship events where:
        #
        # parent_process_id == current PID
        #
        # and collect process_id.
        # -------------------------------------------------

        child_ids = (
            relationship_events[
                relationship_events[
                    "parent_process_id_int"
                ]
                == process_id
            ]["process_id_int"]
            .dropna()
            .astype(int)
            .astype(str)
            .unique()
            .tolist()
        )

        # -------------------------------------------------
        # Command lines
        # -------------------------------------------------

        command_lines = (
            process_events[
                process_events["event_type"]
                == "command_line"
            ]["command_line"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        # -------------------------------------------------
        # SHAP contributions
        # -------------------------------------------------

        feature_shap = {}

        shap_columns = [
            column
            for column in shap_df.columns
            if column.endswith("_shap")
        ]

        for column in shap_columns:

            value = anomaly[column]

            if pd.isna(value):
                continue

            feature = column.replace(
                "_shap",
                "",
            )

            feature_shap[feature] = float(value)

        # -------------------------------------------------
        # Rank SHAP features
        # -------------------------------------------------

        top_features = sorted(
            feature_shap.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:3]

        top_feature_names = [
            feature
            for feature, _ in top_features
        ]

        top_feature_values = [
            value
            for _, value in top_features
        ]

        # -------------------------------------------------
        # Store compact evidence record
        # -------------------------------------------------

        records.append(
            {
                "case_id": anomaly["case_id"],

                "process_id": process_id,

                "process_name": process_name,

                "anomaly_score": float(
                    anomaly["anomaly_score"]
                ),

                "predicted_anomaly": int(
                    anomaly["predicted_anomaly"]
                ),

                # -------------------------------
                # SHAP
                # -------------------------------

                "top_feature_1": (
                    top_feature_names[0]
                    if len(top_feature_names) > 0
                    else ""
                ),

                "top_feature_1_shap": (
                    top_feature_values[0]
                    if len(top_feature_values) > 0
                    else 0
                ),

                "top_feature_2": (
                    top_feature_names[1]
                    if len(top_feature_names) > 1
                    else ""
                ),

                "top_feature_2_shap": (
                    top_feature_values[1]
                    if len(top_feature_values) > 1
                    else 0
                ),

                "top_feature_3": (
                    top_feature_names[2]
                    if len(top_feature_names) > 2
                    else ""
                ),

                "top_feature_3_shap": (
                    top_feature_values[2]
                    if len(top_feature_values) > 2
                    else 0
                ),

                # -------------------------------
                # Evidence counts
                # -------------------------------

                "process_event_count": len(
                    process_events
                ),

                "pslist_count": artifact_counts.get(
                    "pslist",
                    0,
                ),

                "pstree_count": artifact_counts.get(
                    "pstree",
                    0,
                ),

                "cmdline_count": artifact_counts.get(
                    "cmdline",
                    0,
                ),

                "dlllist_count": artifact_counts.get(
                    "dlllist",
                    0,
                ),

                "malfind_count": artifact_counts.get(
                    "malfind",
                    0,
                ),

                # -------------------------------
                # Relationships
                # -------------------------------

                "event_types": ";".join(
                    event_types
                ),

                "parent_process_ids": ";".join(
                    parent_ids
                ),

                "child_process_ids": ";".join(
                    child_ids
                ),

                # -------------------------------
                # Command lines
                # -------------------------------

                "command_lines": ";".join(
                    command_lines
                ),

                # -------------------------------
                # Evidence references
                # -------------------------------

                "evidence_ids": ";".join(
                    evidence_ids
                ),
            }
        )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    output = pd.DataFrame(records)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(output)}")
    print(f"Columns: {len(output.columns)}")

    # -----------------------------------------------------
    # Display top records
    # -----------------------------------------------------

    print(
        "\nTop evidence-attributed anomalies:"
    )

    print(
        output[
            [
                "process_id",
                "process_name",
                "anomaly_score",
                "top_feature_1",
                "top_feature_1_shap",
                "top_feature_2",
                "top_feature_2_shap",
                "top_feature_3",
                "top_feature_3_shap",
                "process_event_count",
                "pslist_count",
                "pstree_count",
                "cmdline_count",
                "dlllist_count",
                "malfind_count",
                "parent_process_ids",
                "child_process_ids",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()