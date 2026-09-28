"""Audit Timeline Dashboard Page."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.services.data_access import (
    load_audit_timeline, load_audit_summary, load_audit_summary_by_component,
    load_recent_activity, calculate_kpis
)
from src.audit import record


def render():
    st.markdown('<div class="main-header">📋 Operational Activity & Audit Timeline</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background: #e7f3ff; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <strong>ℹ️ Audit Timeline:</strong> Tracks all significant operational activities including data generation,
    cleaning, validation, integration, analytics, ML training, Spark jobs, recommendations, alerts,
    and scenario simulations. Every entry is timestamped with component, status, and duration.
    </div>
    """, unsafe_allow_html=True)

    # Refresh button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh", type="primary"):
            st.rerun()

    # Load data
    timeline = load_audit_timeline(limit=2000)
    summary = load_audit_summary()
    by_component = load_audit_summary_by_component()
    recent = load_recent_activity(24)
    kpis = calculate_kpis()

    if timeline.empty:
        st.warning("No audit entries found. Run pipeline components to generate activity.")
        return

    # Time range filter
    col1, col2, col3 = st.columns(3)
    with col1:
        hours = st.selectbox("Time Range", [6, 24, 72, 168, 720], index=1, format_func=lambda x: f"Last {x}h" if x < 168 else f"Last {x//24}d", key="audit_timerange")
    with col2:
        components = ["All"] + sorted(timeline["component"].unique().tolist())
        selected_comp = st.selectbox("Component", components, key="audit_comp_filter")
    with col3:
        statuses = ["All"] + sorted(timeline["status"].unique().tolist())
        selected_status = st.selectbox("Status", statuses, key="audit_status_filter")

    # Apply filters
    from datetime import timedelta, datetime, timezone
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    cutoff_str = cutoff.isoformat(timespec="seconds")

    filtered = timeline[timeline["ts"] >= cutoff_str].copy()
    if selected_comp != "All":
        filtered = filtered[filtered["component"] == selected_comp]
    if selected_status != "All":
        filtered = filtered[filtered["status"] == selected_status]

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Activities (period)", len(filtered))
    with col2:
        st.metric("Success Rate", f"{(filtered['status']=='ok').mean()*100:.1f}%" if len(filtered) > 0 else "0%")
    with col3:
        avg_dur = filtered[filtered["duration_ms"].notna()]["duration_ms"].mean()
        st.metric("Avg Duration", f"{avg_dur/1000:.1f}s" if pd.notna(avg_dur) else "N/A")
    with col4:
        st.metric("Components Active", filtered["component"].nunique())

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Timeline", "📊 Analytics", "🔧 Component View", "📈 Trends"])

    with tab1:
        render_timeline(filtered)

    with tab2:
        render_analytics(summary, by_component, filtered)

    with tab3:
        render_component_view(by_component, filtered)

    with tab4:
        render_trends(timeline)


def render_timeline(df: pd.DataFrame):
    """Render the activity timeline."""
    if df.empty:
        st.info("No activities match the current filters.")
        return

    # Format for display
    display_df = df.copy()
    display_df["ts"] = pd.to_datetime(display_df["ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    if "duration_ms" in display_df.columns:
        display_df["duration_ms"] = display_df["duration_ms"].apply(
            lambda x: f"{x/1000:.1f}s" if pd.notna(x) else "N/A"
        )

    # Color by status
    def highlight_status(row):
        colors = {
            "ok": "background-color: #d4edda",
            "failed": "background-color: #f8d7da",
            "warning": "background-color: #fff3cd",
            "skipped": "background-color: #e2e3e5",
        }
        color = colors.get(row.get("status", ""), "")
        return [color] * len(row)

    display_cols = ["ts", "action", "component", "status", "details", "duration_ms", "version"]
    available = [c for c in display_cols if c in display_df.columns]

    styled = display_df[available].head(500).style.apply(highlight_status, axis=1)
    st.dataframe(styled, use_container_width=True, height=600)

    # Download
    csv = display_df[available].to_csv(index=False)
    st.download_button(
        "📥 Download Timeline (CSV)",
        csv,
        f"audit_timeline_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        "text/csv",
    )


def render_analytics(summary: pd.DataFrame, by_component: pd.DataFrame, filtered: pd.DataFrame):
    """Render audit analytics."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="sub-header">Activity by Action & Status</div>', unsafe_allow_html=True)
        if not summary.empty:
            fig = px.bar(
                summary, x="action", y="count", color="status",
                title="Activity Counts by Action & Status",
                labels={"count": "Count", "action": "Action"},
                barmode="group"
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="sub-header">Activity by Component</div>', unsafe_allow_html=True)
        if not by_component.empty:
            fig = px.pie(
                by_component, values="count", names="component",
                title="Activity Distribution by Component",
                hole=0.3
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Status distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="sub-header">Status Distribution</div>', unsafe_allow_html=True)
        status_counts = filtered["status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]

        fig = px.pie(
            status_counts, values="Count", names="Status",
            title="Status Distribution (Filtered)",
            color="Status",
            color_discrete_map={
                "ok": "#28a745",
                "failed": "#dc3545",
                "warning": "#ffc107",
                "skipped": "#6c757d",
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="sub-header">Average Duration by Action</div>', unsafe_allow_html=True)
        dur_df = filtered[filtered["duration_ms"].notna()].copy()
        if not dur_df.empty:
            avg_dur = dur_df.groupby("action")["duration_ms"].mean().reset_index()
            avg_dur["duration_s"] = avg_dur["duration_ms"] / 1000

            fig = px.bar(
                avg_dur.sort_values("duration_s", ascending=True).tail(15),
                x="duration_s", y="action", orientation="h",
                title="Avg Duration by Action (Top 15)",
                labels={"duration_s": "Duration (seconds)", "action": "Action"}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

    # Detailed summary table
    st.markdown('<div class="sub-header">Detailed Summary</div>', unsafe_allow_html=True)
    if not summary.empty:
        st.dataframe(summary.sort_values("count", ascending=False), use_container_width=True)


def render_component_view(by_component: pd.DataFrame, filtered: pd.DataFrame):
    """Render per-component activity view."""
    components = sorted(filtered["component"].unique().tolist())
    selected = st.selectbox("Select Component", components, key="comp_view_select")

    comp_data = filtered[filtered["component"] == selected].copy()

    if comp_data.empty:
        st.info(f"No activity for {selected} in the current filter.")
        return

    st.markdown(f'<div class="sub-header">{selected} Activity</div>', unsafe_allow_html=True)

    # Action breakdown
    col1, col2 = st.columns(2)

    with col1:
        action_counts = comp_data["action"].value_counts().reset_index()
        action_counts.columns = ["Action", "Count"]

        fig = px.bar(
            action_counts, x="Action", y="Count",
            title=f"{selected} - Actions",
            color="Action"
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        status_counts = comp_data["status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]

        fig = px.pie(
            status_counts, values="Count", names="Status",
            title=f"{selected} - Status"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Timeline for this component
    st.markdown(f'<div class="sub-header">{selected} - Recent Activity</div>', unsafe_allow_html=True)

    display_cols = ["ts", "action", "status", "details", "duration_ms"]
    available = [c for c in display_cols if c in comp_data.columns]
    display_df = comp_data[available].head(100).copy()
    display_df["ts"] = pd.to_datetime(display_df["ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    if "duration_ms" in display_df.columns:
        display_df["duration_ms"] = display_df["duration_ms"].apply(
            lambda x: f"{x/1000:.1f}s" if pd.notna(x) else "N/A"
        )

    st.dataframe(display_df, use_container_width=True, height=400)


def render_trends(full_timeline: pd.DataFrame):
    """Render activity trends over time."""
    if full_timeline.empty:
        st.info("No data for trends.")
        return

    # Daily activity count
    full_timeline["date"] = pd.to_datetime(full_timeline["ts"]).dt.date
    daily = full_timeline.groupby("date").size().reset_index(name="count")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="sub-header">Daily Activity Volume</div>', unsafe_allow_html=True)
        fig = px.line(
            daily, x="date", y="count",
            title="Daily Activity Count",
            labels={"count": "Activities", "date": "Date"}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Daily by status
        daily_status = full_timeline.groupby(["date", "status"]).size().reset_index(name="count")
        fig = px.area(
            daily_status, x="date", y="count", color="status",
            title="Daily Activity by Status",
            labels={"count": "Activities", "date": "Date"},
            color_discrete_map={
                "ok": "#28a745",
                "failed": "#dc3545",
                "warning": "#ffc107",
                "skipped": "#6c757d",
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Weekly trend
    st.markdown('<div class="sub-header">Weekly Activity Trend</div>', unsafe_allow_html=True)
    daily["week"] = pd.to_datetime(daily["date"]).dt.to_period("W").dt.start_time
    weekly = daily.groupby("week")["count"].sum().reset_index()

    fig = px.bar(
        weekly, x="week", y="count",
        title="Weekly Activity Volume",
        labels={"count": "Activities", "week": "Week"}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Component activity over time
    st.markdown('<div class="sub-header">Component Activity Over Time</div>', unsafe_allow_html=True)
    comp_daily = full_timeline.groupby(["date", "component"]).size().reset_index(name="count")

    fig = px.line(
        comp_daily, x="date", y="count", color="component",
        title="Daily Activity by Component",
        labels={"count": "Activities", "date": "Date"}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)