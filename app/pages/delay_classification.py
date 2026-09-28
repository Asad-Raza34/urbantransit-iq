"""Delay Severity Classification Dashboard Page.

Renders only persisted evaluation artifacts (`models/python/delay_classification_*
_metrics.json`) — every number on this page is the frozen result of the staged
selection → single-test-evaluation protocol in `src.ml.delay_classification`.
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ml.delay_classification import (
    TARGETS,
    build_all_classifiers,
    evaluate_delay_classifier,
    load_saved_model_results,
    predict_delay_severity,
)
from src.services.data_access import load_route_metrics

CLASS_COLORS = {
    "on time": "#2ca02c",
    "moderate": "#ff7f0e",
    "severe": "#d62728",
    "critical": "#8b0000",
    "delayed": "#d62728",
    "minor": "#ff7f0e",
}


def _plot_confusion_matrix(cm, labels, title="Confusion Matrix"):
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 14},
        hoverongaps=False,
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=450,
    )
    return fig


def _plot_class_metrics(per_class: dict, title="Per-Class Metrics (Test)"):
    classes = list(per_class.keys())
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Precision", x=classes,
                         y=[per_class[c]["precision"] for c in classes],
                         marker_color="#1f77b4"))
    fig.add_trace(go.Bar(name="Recall", x=classes,
                         y=[per_class[c]["recall"] for c in classes],
                         marker_color="#ff7f0e"))
    fig.add_trace(go.Bar(name="F1-Score", x=classes,
                         y=[per_class[c]["f1-score"] for c in classes],
                         marker_color="#2ca02c"))
    fig.update_layout(title=title, barmode="group",
                      yaxis=dict(range=[0, 1.05], title="Score"), height=400)
    return fig


def _plot_class_distribution(counts: dict, title="Class Distribution (Test)"):
    labels = list(counts.keys())
    values = list(counts.values())
    fig = px.bar(x=labels, y=values, color=labels,
                 title=title,
                 color_discrete_map=CLASS_COLORS)
    fig.update_layout(xaxis_title="Class", yaxis_title="Trips", height=350,
                      showlegend=False)
    return fig


def _plot_feature_importance(entries: list[dict], title: str):
    frame = pd.DataFrame(entries).sort_values("importance", ascending=True).tail(15)
    fig = px.bar(frame, x="importance", y="feature", orientation="h",
                 title=title, labels={"importance": "Permutation importance (macro-F1 drop)"})
    fig.update_layout(height=max(380, 25 * len(frame)))
    return fig


def _format_metric(value, digits=4):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _metric_block(test: dict, validation: dict) -> None:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Test Accuracy", _format_metric(test.get("accuracy")))
        st.metric("Validation Accuracy", _format_metric(validation.get("accuracy")))
    with col2:
        st.metric("Test Macro F1", _format_metric(test.get("macro_f1")))
        st.metric("Validation Macro F1", _format_metric(validation.get("macro_f1")))
    with col3:
        st.metric("Test Weighted F1", _format_metric(test.get("weighted_f1")))
        st.metric("Balanced Accuracy (Test)", _format_metric(test.get("balanced_accuracy")))
    with col4:
        gap_acc = (test.get("accuracy", 0) - validation.get("accuracy", 0)) \
            if test.get("accuracy") and validation.get("accuracy") else 0
        gap_f1 = (test.get("macro_f1", 0) - validation.get("macro_f1", 0)) \
            if test.get("macro_f1") and validation.get("macro_f1") else 0
        st.metric("Overfitting Gap (Acc, Test−Val)", f"{gap_acc:+.4f}")
        st.metric("Overfitting Gap (Macro F1, Test−Val)", f"{gap_f1:+.4f}")


def render() -> None:
    st.markdown('<div class="main-header">🎯 Delay Severity Classification</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div style="background: #fff3e0; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <strong>What this model does:</strong> predicts each trip's <strong>operational delay
    category before departure</strong> using schedule, route, vehicle, weather, event and
    historical-performance features. The target is the platform's own severity definition
    (shared with the analytics pipeline), so a prediction means exactly what the dashboards
    mean by the same word.
    <ul>
    <li><strong>on time</strong>: delay ≤ 5 min</li>
    <li><strong>moderate</strong>: 5 &lt; delay ≤ 9 min</li>
    <li><strong>severe</strong>: 9 &lt; delay ≤ 30 min</li>
    <li><strong>critical</strong>: delay &gt; 30 min</li>
    </ul>
    <span style="color:#555">A binary <em>delay status</em> variant (on time vs delayed,
    5-minute threshold) is trained alongside. Selection uses validation folds only; the
    December test period is evaluated exactly once and every figure below is read from the
    persisted metrics files.</span>
    </div>
    """, unsafe_allow_html=True)

    results = st.session_state.get("classifier_results")
    if results is None:
        results = load_saved_model_results()
        if results:
            st.session_state["classifier_results"] = results

    if not results:
        st.info("No trained delay-classification artifacts found. Run the staged protocol "
                "below (selection → final) or "
                "`python -m src.ml.delay_classification --stage all`.")
    else:
        st.markdown('<div class="sub-header">Model Performance (frozen evaluations)</div>',
                    unsafe_allow_html=True)
        names = {"severity": "🎯 Severity (4-class, primary)",
                 "delay_status": "🚦 Delay Status (on time vs delayed)"}
        for name, art in results.items():
            title = names.get(name, name)
            if "error" in art:
                st.error(f"{title}: {art['error']}")
                continue
            test_m = art.get("test", {})
            val_m = art.get("validation", {})
            candidate = art.get("candidate", {})
            with st.expander(
                f"{title} — Test Acc {_format_metric(test_m.get('accuracy'))} · "
                f"Macro F1 {_format_metric(test_m.get('macro_f1'))}", expanded=(name == "severity")):

                st.caption(f"Algorithm: `{candidate.get('family', '—')}` "
                           f"(`{candidate.get('config', '—')}`, class-weight "
                           f"`{candidate.get('weight', '—')}`) · final fit on "
                           f"{art.get('split', {}).get('final_fit_rows', 0):,} rows · "
                           f"test period {art.get('split', {}).get('test', {}).get('from')} → "
                           f"{art.get('split', {}).get('test', {}).get('to')}")
                _metric_block(test_m, val_m)

                if test_m.get("per_class"):
                    st.plotly_chart(_plot_class_metrics(test_m["per_class"]),
                                    use_container_width=True)
                if test_m.get("confusion_matrix"):
                    st.plotly_chart(
                        _plot_confusion_matrix(test_m["confusion_matrix"],
                                               test_m.get("labels", list(test_m["per_class"])),
                                               f"Confusion Matrix — {title} (Test)"),
                        use_container_width=True)
                if art.get("class_distribution", {}).get("test"):
                    st.plotly_chart(
                        _plot_class_distribution(art["class_distribution"]["test"]),
                        use_container_width=True)
                if art.get("feature_importance"):
                    st.plotly_chart(
                        _plot_feature_importance(art["feature_importance"],
                                                 f"Feature Importance — {title}"),
                        use_container_width=True)

        # model comparison -------------------------------------------------
        comparison = [
            {"model": names.get(name, name),
             "test accuracy": art.get("test", {}).get("accuracy"),
             "test macro F1": art.get("test", {}).get("macro_f1")}
            for name, art in results.items() if "error" not in art
        ]
        if comparison:
            st.markdown('<div class="sub-header">Model Comparison</div>',
                        unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)

    st.markdown("---")

    # re-training ---------------------------------------------------------
    st.markdown('<div class="sub-header">Staged Re-training</div>', unsafe_allow_html=True)
    st.caption("Selection evaluates candidate models on validation folds only; the final "
               "stage refits the winner and evaluates the December test period once "
               "(re-runs are reproductions of the frozen configuration).")
    task_names = {"severity": "🎯 Severity (4-class, primary)",
                  "delay_status": "🚦 Delay Status (on time vs delayed)",
                  "severity_balanced": "⚖️ Severity, balanced thresholds (comparison)"}
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        target_name = st.selectbox("Task", list(TARGETS),
                                   format_func=lambda t: task_names.get(t, t))
    with col2:
        if st.button("🔍 Selection sweep", type="primary", key="classifier_select_btn"):
            with st.spinner("Running the selection sweep (validation folds only)..."):
                from src.ml.delay_classification import run_selection
                artifact = run_selection(target_name, full_sweep=False)
                st.success(f"Selected {artifact['selected']['candidate']['family']} "
                           f"({len(artifact['candidates'])} candidates)")
                st.rerun()
    with col3:
        if st.button("🧊 Final fit + test eval", key="classifier_final_btn"):
            with st.spinner("Refitting the frozen configuration and evaluating the test "
                            "period once..."):
                from src.ml.delay_classification import run_final
                run_final(target_name, force=True)
                st.session_state.pop("classifier_results", None)
                st.success("Frozen evaluation updated.")
                st.rerun()

    st.markdown("---")

    # test-set evaluation recap --------------------------------------------
    st.markdown('<div class="sub-header">Re-evaluate from stored predictions</div>',
                unsafe_allow_html=True)
    if st.button("📊 Recompute test metrics from predictions parquet", key="eval_btn"):
        for name in results or {}:
            result = evaluate_delay_classifier(name)
            if "error" in result:
                st.error(result["error"])
            else:
                st.success(f"{names.get(name, name)}: recomputed test accuracy "
                           f"{_format_metric(result.get('accuracy'))}, macro F1 "
                           f"{_format_metric(result.get('macro_f1'))} — matches the stored "
                           "metrics (single source of truth).")

    st.markdown("---")

    # predictions ------------------------------------------------------------
    st.markdown('<div class="sub-header">Generate Predictions</div>', unsafe_allow_html=True)
    route_m = load_route_metrics()
    all_routes = sorted(route_m["route_id"].unique().tolist())

    col1, col2 = st.columns(2)
    with col1:
        selected_routes = st.multiselect("Select Routes (empty = all)", all_routes,
                                         default=[], key="classifier_routes")
    with col2:
        pred_date = st.date_input("Date (optional)", value=None, key="classifier_date")

    if st.button("🔮 Predict Severity", type="primary", key="predict_btn"):
        with st.spinner("Generating predictions..."):
            try:
                pred = predict_delay_severity(
                    route_ids=selected_routes or None,
                    date=pred_date.strftime("%Y-%m-%d") if pred_date else None)
                st.session_state["pred_df"] = pred
                st.success(f"Generated {len(pred)} predictions")
            except FileNotFoundError as exc:
                st.warning(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Prediction failed: {exc}")

    if "pred_df" in st.session_state:
        pred_df = st.session_state["pred_df"]
        if pred_df.empty:
            st.info("No trips match the selected filters.")
        else:
            st.markdown("**Prediction Summary**")
            counts = pred_df["predicted_target"].value_counts().to_dict()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Predictions", len(pred_df))
            with col2:
                st.metric("On Time", int(counts.get("on time", 0)))
            with col3:
                st.metric("Moderate", int(counts.get("moderate", 0)))
            with col4:
                st.metric("Severe + Critical", int(counts.get("severe", 0)
                                                  + counts.get("critical", 0)))

            dist = pred_df["predicted_target"].value_counts().reset_index()
            dist.columns = ["Severity", "Count"]
            fig = px.bar(dist, x="Severity", y="Count", color="Severity",
                         title="Predicted Severity Distribution",
                         color_discrete_map=CLASS_COLORS)
            st.plotly_chart(fig, use_container_width=True)

            display_cols = ["trip_id", "route_id", "date", "time_band", "is_peak",
                            "actual_target", "predicted_target",
                            "predicted_target_adjusted", "departure_delay_min"]
            prob_cols = [c for c in pred_df.columns if c.startswith("prob_")]
            available = [c for c in display_cols + prob_cols if c in pred_df.columns]
            st.dataframe(pred_df[available].head(200), use_container_width=True, height=400)


if __name__ == "__main__":
    render()
