"""Route Analytics Dashboard Page."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.services.data_access import load_route_metrics, load_stop_metrics
from src.comparison.routes import compare_routes, rank_routes, get_route_profile


def render():
    st.markdown('<div class="main-header">🛣️ Route Analytics</div>', unsafe_allow_html=True)

    route_m = load_route_metrics()
    stop_m = load_stop_metrics()

    if route_m.empty:
        st.warning("No route metrics data available.")
        return

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 All Routes", "📊 Comparison", "🏆 Rankings", "🔍 Route Detail"])

    with tab1:
        render_all_routes(route_m)

    with tab2:
        render_comparison(route_m)

    with tab3:
        render_rankings(route_m)

    with tab4:
        render_route_detail(route_m)


def render_all_routes(route_m: pd.DataFrame):
    """Render filterable route table with all metrics."""
    st.markdown('<div class="sub-header">Route Metrics Table</div>', unsafe_allow_html=True)

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        categories = ["All"] + sorted(route_m["category"].unique().tolist())
        selected_cat = st.selectbox("Category", categories, key="route_cat_filter")
    with col2:
        classes = ["All"] + sorted(route_m["route_class"].unique().tolist())
        selected_class = st.selectbox("Classification", classes, key="route_class_filter")
    with col3:
        score_range = st.slider("Route Score Range",
                                float(route_m["route_score"].min()),
                                float(route_m["route_score"].max()),
                                (float(route_m["route_score"].min()), float(route_m["route_score"].max())),
                                key="route_score_filter")

    # Apply filters
    df = route_m.copy()
    if selected_cat != "All":
        df = df[df["category"] == selected_cat]
    if selected_class != "All":
        df = df[df["route_class"] == selected_class]
    df = df[(df["route_score"] >= score_range[0]) & (df["route_score"] <= score_range[1])]

    # Select display columns
    display_cols = [
        "route_id", "route_name", "category", "route_class", "route_score", "route_rank",
        "passengers", "revenue", "trips", "avg_boardings_per_trip",
        "avg_delay_min", "on_time_share", "adherence_share",
        "occupancy_avg_pct", "occupancy_max_pct", "crowding_share",
        "bunching_share", "avg_headway_min", "avg_actual_travel_min",
    ]
    available = [c for c in display_cols if c in df.columns]

    # Format for display
    display_df = df[available].copy()
    if "on_time_share" in display_df.columns:
        display_df["on_time_share"] = (display_df["on_time_share"] * 100).round(1)
    if "adherence_share" in display_df.columns:
        display_df["adherence_share"] = (display_df["adherence_share"] * 100).round(1)
    if "crowding_share" in display_df.columns:
        display_df["crowding_share"] = (display_df["crowding_share"] * 100).round(1)
    if "bunching_share" in display_df.columns:
        display_df["bunching_share"] = (display_df["bunching_share"] * 100).round(1)

    st.dataframe(display_df, use_container_width=True, height=600)

    # Download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        "📥 Download Filtered Routes (CSV)",
        csv,
        f"routes_filtered_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        "text/csv",
    )


def render_comparison(route_m: pd.DataFrame):
    """Render route comparison tool."""
    st.markdown('<div class="sub-header">Route Comparison Tool</div>', unsafe_allow_html=True)

    route_ids = sorted(route_m["route_id"].unique().tolist())
    selected = st.multiselect(
        "Select Routes to Compare (2-10)",
        route_ids,
        default=route_ids[:3] if len(route_ids) >= 3 else route_ids,
        max_selections=10,
        key="compare_routes_select",
    )

    if len(selected) < 2:
        st.info("Select at least 2 routes to compare.")
        return

    metric_group = st.selectbox(
        "Metric Group",
        ["All", "Demand", "Delay", "Occupancy", "Headway", "Overall", "Route Info"],
        key="compare_metric_group",
    )

    groups = {
        "Demand": ["passengers", "revenue", "trips", "avg_boardings_per_trip"],
        "Delay": ["avg_delay_min", "p90_delay_min", "on_time_share", "adherence_share", "severe_critical_share"],
        "Occupancy": ["occupancy_avg_pct", "occupancy_max_pct", "crowding_share", "critical_crowding_share", "underutilized_share"],
        "Headway": ["avg_headway_min", "headway_std_min", "headway_cv", "bunching_share", "gapping_share"],
        "Overall": ["route_score", "route_rank", "route_class"],
        "Route Info": ["category", "length_km"],
    }

    if metric_group == "All":
        metric_groups = list(groups.keys())
    else:
        metric_groups = [metric_group]

    # Get comparison
    cmp = compare_routes(selected, metric_groups=metric_groups, include_all_metrics=False)

    if not cmp.metrics:
        st.warning("No data for selected routes.")
        return

    # Display comparison table
    st.markdown('<div class="sub-header">Side-by-Side Comparison</div>', unsafe_allow_html=True)

    # Build comparison DataFrame
    rows = []
    for rid in selected:
        row = {"route_id": rid}
        row.update(cmp.metrics.get(rid, {}))
        rows.append(row)

    cmp_df = pd.DataFrame(rows)

    # Format for display
    for col in cmp_df.columns:
        if col in ["on_time_share", "adherence_share", "crowding_share", "bunching_share", "underutilized_share", "critical_crowding_share", "severe_critical_share", "on_time_rate"]:
            if col in cmp_df.columns:
                cmp_df[col] = (cmp_df[col] * 100).round(1)

    st.dataframe(cmp_df, use_container_width=True)

    # Differences from median
    if cmp.differences:
        st.markdown('<div class="sub-header">Differences from Median</div>', unsafe_allow_html=True)
        for metric, diff_info in cmp.differences.items():
            spec = diff_info.get("spec", {})
            label = spec.get("label", metric)
            median = diff_info.get("median", 0)
            diffs = diff_info.get("differences", {})

            cols = st.columns(len(selected) + 1)
            with cols[0]:
                st.write(f"**{label}** (median: {median})")
            for i, rid in enumerate(selected):
                with cols[i + 1]:
                    diff = diffs.get(rid, 0)
                    color = "normal"
                    if spec.get("higher_better") is True:
                        color = "inverse" if diff < 0 else "normal"
                    elif spec.get("higher_better") is False:
                        color = "normal" if diff < 0 else "inverse"
                    st.metric(rid, f"{diff:+.2f}", delta_color=color)

    # Visual comparison
    st.markdown('<div class="sub-header">Visual Comparison</div>', unsafe_allow_html=True)

    # Radar chart for key metrics
    key_metrics = ["route_score", "on_time_share", "occupancy_avg_pct", "avg_delay_min", "bunching_share", "crowding_share"]
    key_metrics = [m for m in key_metrics if m in cmp_df.columns]

    if len(key_metrics) >= 3:
        fig = go.Figure()
        for _, row in cmp_df.iterrows():
            values = [row[m] for m in key_metrics]
            # Normalize for radar (0-1 scale)
            normalized = []
            for i, m in enumerate(key_metrics):
                col_vals = cmp_df[m].dropna()
                if len(col_vals) > 1:
                    mn, mx = col_vals.min(), col_vals.max()
                    if mx > mn:
                        normalized.append((row[m] - mn) / (mx - mn))
                    else:
                        normalized.append(0.5)
                else:
                    normalized.append(0.5)

            # Close the loop
            normalized.append(normalized[0])
            labels = [METRIC_LABELS.get(m, m) for m in key_metrics]
            labels.append(labels[0])

            fig.add_trace(go.Scatterpolar(
                r=normalized,
                theta=labels,
                fill='toself',
                name=row["route_id"],
                opacity=0.7,
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Normalized Metric Comparison (0=worst, 1=best)",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)


def render_rankings(route_m: pd.DataFrame):
    """Render route rankings by various metrics."""
    st.markdown('<div class="sub-header">Route Rankings</div>', unsafe_allow_html=True)

    ranking_metrics = {
        "route_score": ("Route Score", False),
        "avg_delay_min": ("Avg Delay", True),
        "on_time_share": ("On-Time %", False),
        "occupancy_avg_pct": ("Avg Occupancy", False),
        "crowding_share": ("Crowding %", True),
        "bunching_share": ("Bunching %", True),
        "passengers": ("Total Passengers", False),
        "avg_actual_travel_min": ("Actual Travel Time", True),
        "adherence_share": ("Adherence %", False),
        "avg_headway_min": ("Avg Headway", True),
    }

    metric = st.selectbox(
        "Rank by",
        list(ranking_metrics.keys()),
        format_func=lambda x: ranking_metrics[x][0],
        key="ranking_metric",
    )

    ascending = ranking_metrics[metric][1]
    top_n = st.slider("Top N", 5, 50, 20, key="ranking_top_n")

    ranked = rank_routes(metric, ascending=ascending, top_n=top_n)

    display_cols = ["route_rank", "route_id", "route_name", "category", "route_class",
                    "route_score", metric, "passengers", "avg_delay_min",
                    "on_time_share", "occupancy_avg_pct", "bunching_share"]
    # ``metric`` can equal one of the fixed columns (ranking by route_score is the
    # default), which used to build a frame with duplicate column names and crash
    # st.dataframe. De-duplicate while preserving the intended display order.
    display_cols = list(dict.fromkeys(c for c in display_cols if c))
    available = [c for c in display_cols if c in ranked.columns]

    df = ranked[available].copy()
    if "on_time_share" in df.columns:
        df["on_time_share"] = (df["on_time_share"] * 100).round(1)
    if "crowding_share" in df.columns:
        df["crowding_share"] = (df["crowding_share"] * 100).round(1)
    if "bunching_share" in df.columns:
        df["bunching_share"] = (df["bunching_share"] * 100).round(1)

    st.dataframe(df, use_container_width=True)

    # Bar chart
    fig = px.bar(
        ranked.head(top_n), x=metric, y="route_id",
        orientation="h", color="route_class",
        title=f"Top {top_n} Routes by {ranking_metrics[metric][0]}",
        color_discrete_map={
            "Top Performer": "#28a745",
            "Solid": "#1f77b4",
            "Needs Attention": "#ffc107",
            "Action Required": "#dc3545",
        }
    )
    fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)


def render_route_detail(route_m: pd.DataFrame):
    """Render detailed route profile."""
    from src.comparison.routes import get_route_profile

    route_ids = sorted(route_m["route_id"].unique().tolist())
    selected = st.selectbox("Select Route", route_ids, key="detail_route_select")

    if st.button("Load Profile", key="load_profile_btn"):
        profile = get_route_profile(selected)
        if "error" in profile:
            st.error(profile["error"])
        else:
            render_route_profile(profile)


def render_route_profile(profile: dict):
    """Render detailed route profile."""
    st.markdown(f'<div class="main-header">Route Profile: {profile["route_id"]}</div>', unsafe_allow_html=True)

    basic = profile["basic_info"]
    perf = profile["performance_summary"]

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Route Score", f"{perf['route_score']:.1f}", help="Composite 0-100 score")
    with col2:
        st.metric("Classification", perf["route_class"])
    with col3:
        st.metric("Rank", f"#{perf['route_rank']}")
    with col4:
        st.metric("Category", basic.get("category", "N/A"))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Passengers", f"{perf['passengers']:,}")
    with col2:
        st.metric("Avg Delay", f"{perf['avg_delay_min']:.1f} min")
    with col3:
        st.metric("On-Time %", f"{perf['on_time_pct']:.1f}%")
    with col4:
        st.metric("Adherence %", f"{perf.get('adherence_share', 0)*100:.1f}%")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Occupancy", f"{perf['occupancy_avg_pct']:.1f}%")
    with col2:
        st.metric("Max Occupancy", f"{perf['occupancy_max_pct']:.1f}%")
    with col3:
        st.metric("Crowding %", f"{perf['crowding_share']:.1f}%")
    with col4:
        st.metric("Bunching %", f"{perf['bunching_share']:.1f}%")

    st.markdown("---")

    # Detailed sections
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="sub-header">Crowding Analysis</div>', unsafe_allow_html=True)
        crowding = profile["crowding"]
        st.metric("Total Events", crowding["events"])
        st.metric("Persistent Days", crowding["persistent_days"])
        if crowding["peak_bands"]:
            st.write("**Peak Crowding Bands:**", ", ".join(crowding["peak_bands"]))
        if crowding["severity_breakdown"]:
            st.write("**Severity Breakdown:**")
            for sev, count in crowding["severity_breakdown"].items():
                st.write(f"  - {sev}: {count}")

    with col2:
        st.markdown('<div class="sub-header">Bunching Analysis</div>', unsafe_allow_html=True)
        bunching = profile["bunching"]
        st.metric("Total Events", bunching["events"])
        st.metric("Event Days", bunching["event_days"])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="sub-header">Reliability</div>', unsafe_allow_html=True)
        rel = profile["reliability"]
        st.metric("Avg Quality Score", f"{rel['avg_quality_score']:.1f}" if rel['avg_quality_score'] else "N/A")
        st.metric("Current Status", rel["current_status"] or "N/A")
        st.metric("Degraded Days", rel["degraded_days"])
        st.metric("Critical Days", rel["critical_days"])

    with col2:
        st.markdown('<div class="sub-header">Demand</div>', unsafe_allow_html=True)
        demand = profile["demand"]
        st.metric("Avg Daily Passengers", f"{demand['avg_daily']:,.0f}")
        st.metric("Peak Daily", f"{demand['peak_daily']:,.0f}")
        if demand["trend_7d"]:
            st.metric("7-Day Trend", f"{demand['trend_7d']:,.0f}")

    # Stop bottlenecks
    st.markdown('<div class="sub-header">Stop Bottlenecks</div>', unsafe_allow_html=True)
    stops = profile["stops"]
    st.metric("Total Stops", stops["total"])
    st.metric("Bottleneck Stops", stops["bottlenecks"])
    if stops["top_boarding_stops"]:
        st.dataframe(pd.DataFrame(stops["top_boarding_stops"]), use_container_width=True)

    if st.button("← Back"):
        st.rerun()


# Metric labels for radar chart
METRIC_LABELS = {
    "route_score": "Route Score",
    "on_time_share": "On-Time %",
    "occupancy_avg_pct": "Avg Occupancy %",
    "avg_delay_min": "Avg Delay (inv)",
    "bunching_share": "Bunching % (inv)",
    "crowding_share": "Crowding % (inv)",
}