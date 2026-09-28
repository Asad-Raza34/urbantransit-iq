"""Recommendations Dashboard Page."""

import ast
import json

import streamlit as st
import pandas as pd
import plotly.express as px

from src.audit import record
from src.services.data_access import load_recommendations, load_recommendations_explainable
from src.recommend.engine import build_recommendations


def render():
    st.markdown('<div class="main-header">🎯 Recommendations</div>', unsafe_allow_html=True)

    # Build/refresh button
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown('<div class="sub-header">Evidence-Based Recommendations</div>', unsafe_allow_html=True)
    with col2:
        if st.button("🔄 Regenerate Recommendations", type="primary"):
            with st.spinner("Building recommendations..."):
                result = build_recommendations()
                st.success(f"Generated {result['count']} recommendations")
                st.rerun()
    with col3:
        from src.services.data_access import load_recommendations
        recs = load_recommendations()
        st.metric("Total Recommendations", len(recs))

    # Load recommendations
    recs_df = load_recommendations()
    explainable = load_recommendations_explainable()

    if recs_df.empty:
        st.warning("No recommendations available. Click 'Regenerate Recommendations' to create them.")
        return

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        priorities = ["All"] + sorted(recs_df["priority"].unique().tolist())
        selected_priority = st.selectbox("Priority", priorities, key="rec_priority_filter")
    with col2:
        categories = ["All"] + sorted(recs_df["category"].unique().tolist())
        selected_category = st.selectbox("Category", categories, key="rec_category_filter")
    with col3:
        routes = ["All"] + sorted([r for r in recs_df["affected_route"].dropna().unique().tolist()])
        selected_route = st.selectbox("Route", routes, key="rec_route_filter")

    # Apply filters
    filtered = recs_df.copy()
    if selected_priority != "All":
        filtered = filtered[filtered["priority"] == selected_priority]
    if selected_category != "All":
        filtered = filtered[filtered["category"] == selected_category]
    if selected_route != "All":
        filtered = filtered[filtered["affected_route"] == selected_route]

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total", len(filtered))
    with col2:
        st.metric("Critical", len(filtered[filtered["priority"] == "CRITICAL"]))
    with col3:
        st.metric("High", len(filtered[filtered["priority"] == "HIGH"]))
    with col4:
        st.metric("Avg Confidence", f"{filtered['confidence'].mean():.0%}" if not filtered.empty else "0%")

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📋 Recommendation List", "📝 Explainable Details", "📊 Summary"])

    with tab1:
        render_recommendation_list(filtered)

    with tab2:
        render_explainable_details(filtered, explainable)

    with tab3:
        render_summary(filtered)


def render_recommendation_list(df: pd.DataFrame):
    """Render filterable recommendation list."""
    if df.empty:
        st.info("No recommendations match the current filters.")
        return

    # Display columns
    display_cols = [
        "recommendation_id", "category", "priority", "affected_route", "affected_stop",
        "title", "description", "metric_value", "threshold_used",
        "suggested_action", "confidence", "generated_at"
    ]
    available = [c for c in display_cols if c in df.columns]

    # Sort by priority then confidence
    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    df = df.sort_values(["priority", "confidence"],
                        key=lambda x: x.map(priority_order) if x.name == "priority" else x,
                        ascending=[True, False])

    # Format for display
    display_df = df[available].copy()
    if "confidence" in display_df.columns:
        display_df["confidence"] = (display_df["confidence"] * 100).round(0).astype(int).astype(str) + "%"
    if "metric_value" in display_df.columns:
        display_df["metric_value"] = display_df["metric_value"].round(2)
    if "generated_at" in display_df.columns:
        display_df["generated_at"] = pd.to_datetime(display_df["generated_at"]).dt.strftime("%Y-%m-%d %H:%M")

    # Color code by priority
    def highlight_priority(row):
        colors = {
            "CRITICAL": "background-color: #f8d7da",
            "HIGH": "background-color: #fff3cd",
            "MEDIUM": "background-color: #e2e3e5",
            "LOW": "background-color: #d1ecf1",
        }
        color = colors.get(row.get("priority", ""), "")
        return [color] * len(row)

    styled = display_df.style.apply(highlight_priority, axis=1)
    st.dataframe(styled, use_container_width=True, height=600)

    # Download
    csv = display_df.to_csv(index=False)
    st.download_button(
        "📥 Download Recommendations (CSV)",
        csv,
        f"recommendations_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        "text/csv",
    )


def render_explainable_details(df: pd.DataFrame, explainable: list):
    """Render full explainable recommendations."""
    if df.empty:
        st.info("No recommendations to explain.")
        return

    # Create lookup for explainable details. This JSON is produced by another
    # process, so an entry with an unexpected shape must not take the whole page
    # down with a bare KeyError.
    exp_lookup = {}
    malformed = 0
    for entry in explainable or []:
        recommendation = entry.get("recommendation") if isinstance(entry, dict) else None
        if not isinstance(recommendation, dict) or "recommendation_id" not in recommendation:
            malformed += 1
            continue
        exp_lookup[recommendation["recommendation_id"]] = entry.get("explanation")

    if malformed:
        plural = "y" if malformed == 1 else "ies"
        st.warning(
            f"{malformed} explainable entr{plural} could not be read "
            "(unexpected format). Click Regenerate Recommendations to rebuild them."
        )

    for _, rec in df.iterrows():
        rec_id = rec["recommendation_id"]
        explanation = exp_lookup.get(rec_id, "No explanation available.")

        # Priority badge color
        priority_colors = {
            "CRITICAL": "🔴",
            "HIGH": "🟠",
            "MEDIUM": "🟡",
            "LOW": "🟢",
        }
        badge = priority_colors.get(rec.get("priority", ""), "⚪")

        with st.expander(f"{badge} {rec.get('title', rec_id)} ({rec.get('priority', 'UNKNOWN')})"):
            st.markdown(explanation)

            # Evidence details
            if rec.get("evidence"):
                st.markdown("### Evidence Metrics")
                evidence = rec["evidence"]
                if isinstance(evidence, str):
                    # ``ast.literal_eval`` parses literals only. ``eval`` would
                    # execute whatever the stored artifact contains, which turns a
                    # tampered parquet file into arbitrary code execution.
                    try:
                        evidence = ast.literal_eval(evidence)
                    except (ValueError, SyntaxError):
                        evidence = {}
                if isinstance(evidence, dict):
                    for k, v in evidence.items():
                        st.write(f"- **{k}**: {v}")

            # Quick actions. Acknowledge/Resolve persist an audit entry (there is
            # no status column on recommendations, so the audit trail is the
            # record of the decision) and the Export button downloads this
            # recommendation as JSON.
            decisions = st.session_state.setdefault("rec_decisions", {})
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"✅ Acknowledge", key=f"ack_{rec_id}"):
                    decisions[rec_id] = "acknowledged"
                    record(action="recommendation:acknowledge", component="recommendations",
                           details=json.dumps({"recommendation_id": rec_id, "title": rec.get("title", "")}, default=str))
                    st.success("Acknowledged!")
            with col2:
                if st.button(f"✅ Resolve", key=f"res_{rec_id}"):
                    decisions[rec_id] = "resolved"
                    record(action="recommendation:resolve", component="recommendations",
                           details=json.dumps({"recommendation_id": rec_id, "title": rec.get("title", "")}, default=str))
                    st.success("Resolved!")
            with col3:
                st.download_button(
                    "📋 Export",
                    json.dumps({k: (v if not hasattr(v, "isoformat") else str(v))
                                for k, v in rec.to_dict().items()}, indent=2, default=str),
                    f"{rec_id}.json",
                    "application/json",
                    key=f"exp_{rec_id}",
                )
            if rec_id in decisions:
                st.caption(f"Decision recorded this session: **{decisions[rec_id]}** (audit entry written).")


def render_summary(df: pd.DataFrame):
    """Render recommendation summary charts."""
    if df.empty:
        st.info("No data to summarize.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="sub-header">Recommendations by Priority</div>', unsafe_allow_html=True)
        priority_counts = df["priority"].value_counts().reset_index()
        priority_counts.columns = ["Priority", "Count"]

        fig = px.bar(
            priority_counts, x="Priority", y="Count",
            title="Recommendations by Priority",
            color="Priority",
            color_discrete_map={
                "CRITICAL": "#dc3545",
                "HIGH": "#fd7e14",
                "MEDIUM": "#ffc107",
                "LOW": "#28a745",
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="sub-header">Recommendations by Category</div>', unsafe_allow_html=True)
        cat_counts = df["category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]

        fig = px.pie(
            cat_counts, values="Count", names="Category",
            title="Recommendations by Category"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Routes affected
    st.markdown('<div class="sub-header">Routes with Most Recommendations</div>', unsafe_allow_html=True)
    route_counts = df["affected_route"].value_counts().reset_index()
    route_counts.columns = ["Route", "Count"]
    route_counts = route_counts.head(20)

    fig = px.bar(
        route_counts, x="Route", y="Count",
        title="Top 20 Routes by Recommendation Count",
        labels={"Count": "Recommendations", "Route": "Route ID"},
        color="Count", color_continuous_scale="Reds"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Confidence distribution
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="sub-header">Confidence Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(
            df, x="confidence", nbins=20,
            title="Recommendation Confidence Distribution",
            labels={"confidence": "Confidence", "count": "Count"}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="sub-header">Action Types</div>', unsafe_allow_html=True)
        # Extract action keywords
        actions = df["suggested_action"].apply(lambda x: x.split(" ")[0] if isinstance(x, str) else "Unknown")
        action_counts = actions.value_counts().reset_index()
        action_counts.columns = ["Action", "Count"]

        fig = px.bar(
            action_counts.head(10), x="Action", y="Count",
            title="Top 10 Action Types",
            labels={"Count": "Count", "Action": "Action"}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)