"""Smart Alerts Dashboard Page."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.services.data_access import load_alerts
from src.alerts.center import build_alerts, update_alert_status


def render():
    st.markdown('<div class="main-header">🚨 Smart Alert Center</div>', unsafe_allow_html=True)

    # Build/refresh button
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown('<div class="sub-header">Operational Alerts</div>', unsafe_allow_html=True)
    with col2:
        if st.button("🔄 Regenerate Alerts", type="primary"):
            with st.spinner("Building alerts..."):
                result = build_alerts()
                st.success(f"Generated {result['count']} alerts")
                st.rerun()
    with col3:
        alerts_df = load_alerts()
        st.metric("Total Alerts", len(alerts_df))

    if alerts_df.empty:
        st.warning("No alerts available. Click 'Regenerate Alerts' to create them.")
        return

    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        severities = ["All"] + ["critical", "high", "medium", "low"]
        selected_sev = st.selectbox("Severity", severities, key="alert_sev_filter")
    with col2:
        statuses = ["All", "new", "acknowledged", "resolved"]
        selected_status = st.selectbox("Status", statuses, key="alert_status_filter")
    with col3:
        types = ["All"] + sorted(alerts_df["alert_type"].unique().tolist())
        selected_type = st.selectbox("Alert Type", types, key="alert_type_filter")
    with col4:
        routes = ["All"] + sorted([r for r in alerts_df["affected_route"].dropna().unique().tolist()])
        selected_route = st.selectbox("Route", routes, key="alert_route_filter")

    # Apply filters
    filtered = alerts_df.copy()
    if selected_sev != "All":
        filtered = filtered[filtered["severity"] == selected_sev]
    if selected_status != "All":
        filtered = filtered[filtered["status"] == selected_status]
    if selected_type != "All":
        filtered = filtered[filtered["alert_type"] == selected_type]
    if selected_route != "All":
        filtered = filtered[filtered["affected_route"] == selected_route]

    # Sort by severity then timestamp
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    filtered = filtered.sort_values(
        ["severity", "timestamp"],
        key=lambda x: x.map(severity_order) if x.name == "severity" else x,
        ascending=[True, False],
    )

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total", len(filtered))
    with col2:
        st.metric("Critical", len(filtered[filtered["severity"] == "critical"]))
    with col3:
        st.metric("High", len(filtered[filtered["severity"] == "high"]))
    with col4:
        st.metric("New", len(filtered[filtered["status"] == "new"]))

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📋 Alert List", "📊 Alert Analytics", "⚙️ Alert Actions"])

    with tab1:
        render_alert_list(filtered)

    with tab2:
        render_alert_analytics(filtered)

    with tab3:
        render_alert_actions(filtered)


def render_alert_list(df: pd.DataFrame):
    """Render filterable alert list."""
    if df.empty:
        st.info("No alerts match the current filters.")
        return

    # Display columns
    display_cols = [
        "alert_id", "alert_type", "severity", "status", "affected_route", "affected_stop",
        "title", "description", "metric_value", "threshold_used", "timestamp"
    ]
    available = [c for c in display_cols if c in df.columns]

    display_df = df[available].copy()
    if "timestamp" in display_df.columns:
        display_df["timestamp"] = pd.to_datetime(display_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")

    # Color code by severity
    def highlight_severity(row):
        colors = {
            "critical": "background-color: #f8d7da",
            "high": "background-color: #fff3cd",
            "medium": "background-color: #e2e3e5",
            "low": "background-color: #d1ecf1",
        }
        color = colors.get(row.get("severity", ""), "")
        return [color] * len(row)

    styled = display_df.style.apply(highlight_severity, axis=1)
    st.dataframe(styled, use_container_width=True, height=600)

    # Download
    csv = display_df.to_csv(index=False)
    st.download_button(
        "📥 Download Alerts (CSV)",
        csv,
        f"alerts_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        "text/csv",
    )


def render_alert_analytics(df: pd.DataFrame):
    """Render alert analytics charts."""
    if df.empty:
        st.info("No data for analytics.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="sub-header">Alerts by Severity</div>', unsafe_allow_html=True)
        sev_counts = df["severity"].value_counts().reset_index()
        sev_counts.columns = ["Severity", "Count"]

        fig = px.bar(
            sev_counts, x="Severity", y="Count",
            title="Alerts by Severity",
            color="Severity",
            color_discrete_map={
                "critical": "#dc3545",
                "high": "#fd7e14",
                "medium": "#ffc107",
                "low": "#28a745",
            }
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="sub-header">Alerts by Type</div>', unsafe_allow_html=True)
        type_counts = df["alert_type"].value_counts().reset_index()
        type_counts.columns = ["Alert Type", "Count"]

        fig = px.pie(
            type_counts, values="Count", names="Alert Type",
            title="Alerts by Type"
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Status breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="sub-header">Alerts by Status</div>', unsafe_allow_html=True)
        status_counts = df["status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]

        fig = px.bar(
            status_counts, x="Status", y="Count",
            title="Alerts by Status",
            color="Status",
            color_discrete_map={
                "new": "#dc3545",
                "acknowledged": "#fd7e14",
                "resolved": "#28a745",
            }
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="sub-header">Alerts by Route</div>', unsafe_allow_html=True)
        route_counts = df["affected_route"].value_counts().head(15).reset_index()
        route_counts.columns = ["Route", "Count"]

        fig = px.bar(
            route_counts, x="Route", y="Count",
            title="Top 15 Routes by Alert Count",
            labels={"Count": "Alerts", "Route": "Route ID"},
            color="Count", color_continuous_scale="Reds"
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Timeline
    st.markdown('<div class="sub-header">Alert Timeline</div>', unsafe_allow_html=True)
    if "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        timeline = df.groupby(["date", "severity"]).size().reset_index(name="count")

        fig = px.line(
            timeline, x="date", y="count", color="severity",
            title="Daily Alert Count by Severity",
            labels={"count": "Alerts", "date": "Date"},
            color_discrete_map={
                "critical": "#dc3545",
                "high": "#fd7e14",
                "medium": "#ffc107",
                "low": "#28a745",
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def render_alert_actions(df: pd.DataFrame):
    """Render alert action controls."""
    if df.empty:
        st.info("No alerts to act on.")
        return

    st.markdown('<div class="sub-header">Bulk Actions</div>', unsafe_allow_html=True)

    # Select alerts for bulk action
    new_alerts = df[df["status"] == "new"]
    if not new_alerts.empty:
        offered = new_alerts["alert_id"].tolist()

        # Drop ids that are no longer offered before the widget is created.
        # After a status update, a filter change or "Regenerate Alerts", the
        # session-state selection could still hold ids absent from ``df``; the
        # label lookup then indexed an empty frame and raised
        # "IndexError: single positional indexer is out-of-bounds".
        stored = st.session_state.get("bulk_alert_select")
        if stored is not None:
            kept = [a for a in stored if a in offered]
            if kept != list(stored):
                st.session_state["bulk_alert_select"] = kept

        # Title lookup that cannot fail, whatever the widget holds.
        title_by_id = dict(zip(df["alert_id"], df["title"]))

        def _label(alert_id: str) -> str:
            title = title_by_id.get(alert_id)
            return f"{alert_id} - {title[:50]}" if isinstance(title, str) else str(alert_id)

        selected = st.multiselect(
            "Select Alerts for Bulk Action",
            offered,
            format_func=_label,
            key="bulk_alert_select",
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ Acknowledge Selected", type="primary", disabled=not selected):
                n = update_alert_status(selected, "acknowledged")
                st.success(f"Acknowledged {n} alerts")
                st.rerun()
        with col2:
            if st.button("✅ Resolve Selected", disabled=not selected):
                n = update_alert_status(selected, "resolved")
                st.success(f"Resolved {n} alerts")
                st.rerun()
        with col3:
            if st.button("📋 Export Selected", disabled=not selected):
                # Stash the selection so the download button can be rendered
                # unconditionally below: a download_button nested inside a button
                # branch disappears on the click's own rerun and never downloads.
                st.session_state["alerts_export_df"] = df[df["alert_id"].isin(selected)]

        export_df = st.session_state.get("alerts_export_df")
        if isinstance(export_df, pd.DataFrame) and not export_df.empty:
            st.download_button(
                "📥 Download Selected",
                export_df.to_csv(index=False),
                "selected_alerts.csv",
                "text/csv",
            )

    # Individual alert actions
    st.markdown('<div class="sub-header">Individual Alert Actions</div>', unsafe_allow_html=True)

    for _, alert in df.head(20).iterrows():
        with st.expander(f"{alert['alert_id']} - {alert['title']} ({alert['severity'].upper()})"):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.write(f"**Type:** {alert['alert_type']}")
                st.write(f"**Severity:** {alert['severity'].upper()}")
                st.write(f"**Status:** {alert['status'].upper()}")
                st.write(f"**Route:** {alert['affected_route'] or 'N/A'}")
                st.write(f"**Stop:** {alert['affected_stop'] or 'N/A'}")
                st.write(f"**Description:** {alert['description']}")
                st.write(f"**Evidence:** {alert['evidence']}")
                st.write(f"**Timestamp:** {alert['timestamp']}")

            with col2:
                if alert["status"] == "new":
                    if st.button("✅ Acknowledge", key=f"ack_{alert['alert_id']}"):
                        update_alert_status([alert["alert_id"]], "acknowledged")
                        st.success("Acknowledged!")
                        st.rerun()
                if alert["status"] in ["new", "acknowledged"]:
                    if st.button("✅ Resolve", key=f"res_{alert['alert_id']}"):
                        update_alert_status([alert["alert_id"]], "resolved")
                        st.success("Resolved!")
                        st.rerun()
                if st.button("📋 View Evidence", key=f"ev_{alert['alert_id']}"):
                    st.json(alert["evidence"])