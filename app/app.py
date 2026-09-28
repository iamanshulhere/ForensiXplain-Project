from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import (
    load_m57_results,
    load_process_graph,
    load_malmem_data,
    load_malmem_model_workflow,
)


st.set_page_config(
    page_title="ForensiXplain | Investigator Console",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        background: rgba(100, 116, 139, 0.08);
        border: 1px solid rgba(100, 116, 139, 0.2);
        padding: 16px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def first_column(df, candidates):
    """Return the first matching column name, ignoring case and punctuation."""
    if df is None or df.empty:
        return None

    normalized = {
        "".join(ch for ch in str(col).lower() if ch.isalnum()): col
        for col in df.columns
    }
    for candidate in candidates:
        key = "".join(ch for ch in candidate.lower() if ch.isalnum())
        if key in normalized:
            return normalized[key]
    return None


def display_frame(df, limit=200):
    if df is None or df.empty:
        st.info("No records are available for this section.")
        return
    st.dataframe(df.head(limit), width="stretch", hide_index=True)
    if len(df) > limit:
        st.caption(f"Showing the first {limit} of {len(df):,} records.")


def prepare_candidates(results):
    """Combine available anomaly outputs while preserving their source."""
    frames = []

    for key, label in [
        ("graph_anomalies", "Graph"),
        ("temporal_anomalies", "Temporal"),
    ]:
        df = results.get(key, pd.DataFrame())
        if df.empty:
            continue

        item = df.copy()
        item["_source"] = label
        item["_row_id"] = [f"{label}-{i}" for i in range(len(item))]
        frames.append(item)

    if not frames:
        return pd.DataFrame()

    candidates = pd.concat(frames, ignore_index=True, sort=False)
    score_col = first_column(
        candidates,
        ["anomaly_score", "score", "decision_function", "anomaly_score_normalized"],
    )
    if score_col:
        candidates["_sort_score"] = pd.to_numeric(
            candidates[score_col], errors="coerce"
        )
        candidates = candidates.sort_values(
            "_sort_score", ascending=False, na_position="last"
        )
        candidates = candidates.drop(columns=["_sort_score"])

    return candidates.reset_index(drop=True)


def render_timeline(timeline):
    if timeline.empty:
        st.info(
            "The logical timeline is not available. Check "
            "data/normalized/M57-Jean/logical_timeline.csv."
        )
        return

    df = timeline.copy()
    time_col = first_column(
        df, ["timestamp", "datetime", "time", "event_time", "start_time"]
    )

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        df = df.sort_values(time_col, na_position="last")

        valid_times = df[time_col].dropna()
        if not valid_times.empty:
            min_time = valid_times.min().date()
            max_time = valid_times.max().date()
            start, end = st.date_input(
                "Filter timeline dates",
                value=(min_time, max_time),
                min_value=min_time,
                max_value=max_time,
            )
            dates = df[time_col].dt.date
            df = df[(dates >= start) & (dates <= end)]

    search = st.text_input("Search timeline records")
    if search:
        mask = df.astype(str).apply(
            lambda col: col.str.contains(search, case=False, na=False)
        ).any(axis=1)
        df = df[mask]

    st.caption(f"{len(df):,} timeline records match the filters.")
    display_frame(df, limit=500)


def render_process_graph(graph):
    if graph is None:
        st.info(
            "The saved process graph is unavailable. Expected file: "
            "data/normalized/M57-Jean/temporal_graph.graphml."
        )
        return

    st.write(
        f"**Nodes:** {graph.number_of_nodes():,}  |  "
        f"**Relationships:** {graph.number_of_edges():,}"
    )

    if graph.number_of_nodes() == 0:
        st.info("The saved graph contains no nodes.")
        return

    node_items = list(graph.nodes(data=True))
    labels = {
        str(node): str(
            attrs.get("label")
            or attrs.get("name")
            or attrs.get("type")
            or node
        )
        for node, attrs in node_items
    }
    node_ids = [str(node) for node, _ in node_items]

    selected_node = st.selectbox(
        "Inspect a graph node",
        options=node_ids,
        format_func=lambda node: f"{labels[node]} [{node}]",
    )

    node_data = dict(graph.nodes(data=True)).get(selected_node)
    if node_data is None:
        # GraphML node IDs are normally strings; handle alternate ID types safely.
        original_node = next(
            (node for node in graph.nodes if str(node) == selected_node),
            None,
        )
        node_data = graph.nodes[original_node] if original_node is not None else {}

    st.markdown("#### Selected node attributes")
    st.json({str(k): str(v) for k, v in node_data.items()})

    related = []
    for source, target, attrs in graph.edges(data=True):
        if str(source) == selected_node or str(target) == selected_node:
            related.append(
                {
                    "source": str(source),
                    "target": str(target),
                    **{str(k): str(v) for k, v in attrs.items()},
                }
            )

    st.markdown("#### Connected relationships")
    display_frame(pd.DataFrame(related), limit=300)

    # Render a readable local neighborhood instead of attempting to draw
    # every node in a potentially large forensic graph.
    neighborhood = {selected_node}
    for source, target in graph.edges():
        if str(source) == selected_node:
            neighborhood.add(str(target))
        if str(target) == selected_node:
            neighborhood.add(str(source))

    neighborhood_graph = nx.Graph()
    neighborhood_graph.add_nodes_from(neighborhood)
    for source, target, attrs in graph.edges(data=True):
        s, t = str(source), str(target)
        if s in neighborhood and t in neighborhood:
            neighborhood_graph.add_edge(
                s, t, relationship=str(attrs.get("type", "related"))
            )

    if neighborhood_graph.number_of_nodes() > 1:
        positions = nx.spring_layout(neighborhood_graph, seed=42)
        edge_x, edge_y = [], []
        for source, target in neighborhood_graph.edges():
            x0, y0 = positions[source]
            x1, y1 = positions[target]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        node_x = [positions[node][0] for node in neighborhood_graph.nodes()]
        node_y = [positions[node][1] for node in neighborhood_graph.nodes()]
        node_text = [
            f"{labels.get(node, node)}<br>ID: {node}"
            for node in neighborhood_graph.nodes()
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                hoverinfo="none",
                line={"width": 1},
                name="Relationships",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=[labels.get(node, node) for node in neighborhood_graph.nodes()],
                textposition="top center",
                hovertext=node_text,
                hoverinfo="text",
                marker={"size": 14},
                name="Nodes",
            )
        )
        fig.update_layout(
            height=500,
            showlegend=False,
            margin={"l": 10, "r": 10, "t": 20, "b": 10},
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "This view shows the selected node and its immediate neighbors, "
            "not the entire graph."
        )


@st.cache_data(show_spinner=False)
def cached_m57_results():
    return load_m57_results()


@st.cache_resource(show_spinner="Training and evaluating the existing model...")
def cached_malmem_workflow():
    return load_malmem_model_workflow()


st.title("🔎 ForensiXplain")
st.caption(
    "Digital forensics investigator console · Evidence exploration, "
    "anomaly review, and explainable machine learning"
)

with st.sidebar:
    st.header("Investigation")
    page = st.radio(
        "Navigate",
        [
            "Case Overview",
            "Anomaly Candidates",
            "Candidate Explanation",
            "Timeline",
            "Process Graph",
            "MalMem2022",
            "Evidence Details",
        ],
    )
    st.divider()
    st.caption("Case: M57-Jean")
    if st.button("Refresh loaded results", width="stretch"):
        cached_m57_results.clear()
        cached_malmem_workflow.clear()
        st.rerun()

results = cached_m57_results()
graph = load_process_graph()
candidates = prepare_candidates(results)

graph_anomalies = results.get("graph_anomalies", pd.DataFrame())
temporal_anomalies = results.get("temporal_anomalies", pd.DataFrame())
timeline = results.get("timeline", pd.DataFrame())
events = results.get("events", pd.DataFrame())

if page == "Case Overview":
    st.header("Case Overview")
    st.write(
        "Review the existing M57-Jean analysis outputs. This dashboard loads "
        "saved results and does not rerun anomaly detection automatically."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Graph anomaly rows", f"{len(graph_anomalies):,}")
    col2.metric("Temporal anomaly rows", f"{len(temporal_anomalies):,}")
    col3.metric("Timeline events", f"{len(timeline):,}")
    col4.metric(
        "Graph nodes",
        f"{graph.number_of_nodes():,}" if graph is not None else "Unavailable",
    )

    st.subheader("Available investigation artifacts")
    artifact_rows = []
    for name, df in results.items():
        artifact_rows.append(
            {
                "Artifact": name.replace("_", " ").title(),
                "Records": len(df) if isinstance(df, pd.DataFrame) else 0,
                "Status": (
                    "Available"
                    if isinstance(df, pd.DataFrame) and not df.empty
                    else "Missing or empty"
                ),
            }
        )
    display_frame(pd.DataFrame(artifact_rows))

    if not candidates.empty:
        st.subheader("Anomaly records by source")
        counts = candidates["_source"].value_counts().rename_axis("Source").reset_index(
            name="Records"
        )
        st.plotly_chart(
            px.bar(counts, x="Source", y="Records", title="Loaded anomaly records"),
            width="stretch",
        )

    st.info(
        "An anomaly or SHAP contribution is an investigative lead, not proof "
        "of malicious activity. Validate findings against source evidence."
    )

elif page == "Anomaly Candidates":
    st.header("Anomaly Candidates")
    if candidates.empty:
        st.warning(
            "No anomaly candidates were loaded. Check the saved graph and "
            "temporal anomaly CSV files under results/M57-Jean."
        )
    else:
        st.caption(f"{len(candidates):,} candidate records loaded.")
        source_filter = st.multiselect(
            "Candidate source",
            options=sorted(candidates["_source"].unique()),
            default=sorted(candidates["_source"].unique()),
        )
        filtered = candidates[candidates["_source"].isin(source_filter)].copy()

        query = st.text_input("Search candidates")
        if query:
            mask = filtered.astype(str).apply(
                lambda col: col.str.contains(query, case=False, na=False)
            ).any(axis=1)
            filtered = filtered[mask]

        display_frame(filtered.drop(columns=["_row_id"], errors="ignore"))

        if not filtered.empty:
            options = filtered["_row_id"].tolist()
            chosen = st.selectbox(
                "Select a candidate for inspection",
                options=options,
                format_func=lambda row_id: (
                    f"{row_id} · "
                    + str(
                        filtered.loc[
                            filtered["_row_id"] == row_id, "_source"
                        ].iloc[0]
                    )
                ),
            )
            selected = filtered[filtered["_row_id"] == chosen].iloc[0]
            st.session_state["selected_candidate"] = selected.to_dict()
            if st.button("Open candidate explanation"):
                st.session_state["selected_candidate"] = selected.to_dict()
                st.session_state["requested_page"] = "Candidate Explanation"
                st.rerun()

elif page == "Candidate Explanation":
    st.header("Candidate Explanation")
    selected_data = st.session_state.get("selected_candidate")

    if not selected_data:
        st.info(
            "Select a candidate on the Anomaly Candidates page first. "
            "You can also inspect the available explanation tables below."
        )
    else:
        selected = pd.Series(selected_data)
        st.subheader(f"Selected {selected.get('_source', 'anomaly')} candidate")
        st.json(
            {
                str(k): (v.item() if hasattr(v, "item") else v)
                for k, v in selected.items()
                if k != "_row_id"
            }
        )

        source = selected.get("_source")
        explanation_key = (
            "graph_explanations" if source == "Graph" else "temporal_explanations"
        )
        shap_key = "graph_shap" if source == "Graph" else "temporal_shap"
        evidence_key = "graph_evidence" if source == "Graph" else "temporal_evidence"

        def show_related_table(key, title):
            df = results.get(key, pd.DataFrame())
            st.markdown(f"#### {title}")
            if df.empty:
                st.info(f"No saved {title.lower()} are available.")
                return

            candidate_id_col = first_column(
                pd.DataFrame([selected]),
                ["event_id", "candidate_id", "id", "process_id", "entity_id"],
            )
            table_id_col = first_column(
                df,
                ["event_id", "candidate_id", "id", "process_id", "entity_id"],
            )
            selected_id = selected.get(candidate_id_col) if candidate_id_col else None

            if (
                candidate_id_col
                and table_id_col
                and selected_id is not None
                and not pd.isna(selected_id)
            ):
                matched = df[
                    df[table_id_col].astype(str) == str(selected_id)
                ]
                if not matched.empty:
                    display_frame(matched)
                    return

            st.caption(
                "No reliable identifier match was found for this candidate. "
                "The full saved table is shown rather than matching by row position."
            )
            display_frame(df)

        show_related_table(explanation_key, "Investigator explanations")
        show_related_table(shap_key, "SHAP feature contributions")
        show_related_table(evidence_key, "Evidence attribution")

        st.warning(
            "Model explanations describe model behavior and feature contributions. "
            "They do not independently establish malicious intent or causality."
        )

elif page == "Timeline":
    st.header("Forensic Timeline")
    st.write("Existing logical timeline, sorted chronologically when timestamps exist.")
    render_timeline(timeline)

elif page == "Process Graph":
    st.header("Process and Relationship Graph")
    st.write("Explore the saved GraphML graph and inspect connected relationships.")
    render_process_graph(graph)

elif page == "MalMem2022":
    st.header("MalMem2022 Classification")
    st.write(
        "This is a separate supervised classification workflow. It does not "
        "replace or combine with the M57-Jean unsupervised anomaly detection."
    )

    features, target, metadata, data_error = load_malmem_data()
    if data_error:
        st.error(f"Could not load MalMem2022: {data_error}")
    elif features is None:
        st.error("MalMem2022 data is unavailable.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Dataset rows", f"{len(features):,}")
        col2.metric("Numeric features", f"{features.shape[1]:,}")
        col3.metric("Target classes", f"{target.nunique():,}")

        st.subheader("Dataset class distribution")
        distribution = target.value_counts().rename_axis("Class").reset_index(
            name="Rows"
        )
        st.plotly_chart(
            px.bar(distribution, x="Class", y="Rows"),
            width="stretch",
        )

        with st.expander("Preview feature data"):
            display_frame(features.head(20))
        with st.expander("Preview metadata"):
            display_frame(metadata.head(20))

        st.markdown("#### Model evaluation")
        st.caption(
            "The existing Random Forest model will train only when requested. "
            "The current workflow uses the project's reproducible 80/20 split."
        )
        if st.button("Train and evaluate Random Forest", type="primary"):
            try:
                workflow = cached_malmem_workflow()
                st.session_state["malmem_workflow"] = workflow
            except Exception as exc:
                st.error(f"Model evaluation failed: {exc}")

        workflow = st.session_state.get("malmem_workflow")
        if workflow:
            metrics = workflow["metrics"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            m2.metric("Precision (Malware)", f"{metrics['precision']:.3f}")
            m3.metric("Recall (Malware)", f"{metrics['recall']:.3f}")
            m4.metric("F1 (Malware)", f"{metrics['f1']:.3f}")

            st.write(
                f"Training rows: **{workflow['train_rows']:,}** · "
                f"Test rows: **{workflow['test_rows']:,}**"
            )

            matrix = metrics["confusion_matrix"]
            st.subheader("Confusion matrix")
            st.dataframe(
                pd.DataFrame(
                    matrix,
                    index=["Actual Benign", "Actual Malware"],
                    columns=["Predicted Benign", "Predicted Malware"],
                ),
                width="stretch",
            )

            st.subheader("Sample prediction")
            X_test = workflow["X_test"]
            y_test = workflow["y_test"]
            model = workflow["model"]
            sample_count = min(10, len(X_test))
            sample = X_test.head(sample_count)
            predictions = model.predict(sample)

            sample_results = sample.copy()
            sample_results.insert(0, "Actual class", y_test.loc[sample.index].values)
            sample_results.insert(1, "Predicted class", predictions)
            display_frame(sample_results, limit=10)

            if st.button("Explain sample with SHAP"):
                try:
                    from src.models.malmem_shap import explain_malmem_model
                    from src.models.malmem_explanation import generate_malmem_explanations

                    shap_values = explain_malmem_model(model, sample)
                    explanations = generate_malmem_explanations(
                        model, sample, shap_values, top_n=5
                    )
                    st.session_state["malmem_shap"] = explanations
                except Exception as exc:
                    st.error(f"SHAP explanation failed: {exc}")

            shap_explanations = st.session_state.get("malmem_shap")
            if shap_explanations is not None:
                st.subheader("Sample SHAP explanations")
                display_frame(shap_explanations, limit=10)

elif page == "Evidence Details":
    st.header("Evidence Details")
    st.write(
        "Inspect loaded source events and evidence attribution. Correlation "
        "indicates a relationship for investigation, not ground truth."
    )

    evidence_tabs = st.tabs(
        ["Source events", "Graph attribution", "Temporal attribution"]
    )
    with evidence_tabs[0]:
        if events.empty:
            st.info("No normalized events are available.")
        else:
            query = st.text_input("Search source events")
            view = events
            if query:
                mask = view.astype(str).apply(
                    lambda col: col.str.contains(query, case=False, na=False)
                ).any(axis=1)
                view = view[mask]
            display_frame(view, limit=500)

    with evidence_tabs[1]:
        display_frame(results.get("graph_evidence", pd.DataFrame()))

    with evidence_tabs[2]:
        display_frame(results.get("temporal_evidence", pd.DataFrame()))

st.divider()
st.caption(
    "ForensiXplain · Human-in-the-loop forensic analysis · "
    "Validate model-generated leads against original evidence."
)

