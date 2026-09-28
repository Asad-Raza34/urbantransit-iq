"""Crowding & Capacity Dashboard Page."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.services.data_access import load_route_metrics, load_crowding_analytics, load_stop_metrics
from src.comparison.routes import get_route_profile


def render():
    st.markdown('<div class="main-header">📈 Crowding & Capacity</div>', unsafe_allow_html=True)

    route_m = load_route_metrics()
    crowding = load_crowding_analytics()
    stop_m = load_stop_metrics()

    if route_m.empty:
        st.warning("No route metrics data available.")
        return

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        overcrowding = crowding.get("overcrowding_events", pd.DataFrame())
        st.metric("Overcrowding Events", len(overcrowding))
    with col2:
        persistent = crowding.get("persistent_overcrowding", pd.DataFrame())
        st.metric("Persistent Overcrowding Routes", len(persistent))
    with col3:
        underutil = crowding.get("underutilization", pd.DataFrame())
        st.metric("Underutilized Route-Bands", len(underutil))
    with col4:
        st.metric("Avg Max Occupancy", f"{route_m['occupancy_max_pct'].mean():.1f}%")

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔴 Overcrowding", "🟢 Underutilization", "🚌 Stop Analysis", "📊 Capacity Planning"])

    with tab1:
        render_overcrowding(overcrowding, persistent, route_m)

    with tab2:
        render_underutilization(underutil, route_m)

    with tab3:
        render_stop_analysis(stop_m, route_m)

    with tab4:
        render_capacity_planning(route_m, crowding)


def render_overcrowding(overcrowding: pd.DataFrame, persistent: pd.DataFrame, route_m: pd.DataFrame):
    """Render overcrowding analysis."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="sub-header">Overcrowding Events by Severity</div>', unsafe_allow_html=True)
        if not overcrowding.empty:
            sev_counts = overcrowding["severity"].value_counts().reset_index()
            sev_counts.columns = ["Severity", "Count"]

            fig = px.pie(
                sev_counts, values="Count", names="Severity",
                title="Overcrowding Events by Severity",
                color="Severity",
                color_discrete_map={"critical": "#dc3545", "high": "#fd7e14"}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No overcrowding events detected.")

    with col2:
        st.markdown('<div class="sub-header">Overcrowding by Time Band</div>', unsafe_allow_html=True)
        if not overcrowding.empty:
            band_counts = overcrowding.groupby("time_band").size().reset_index(name="count")
            band_order = ["night", "morning peak", "midday", "evening peak"]
            band_counts["time_band"] = pd.Categorical(band_counts["time_band"], categories=band_order, ordered=True)
            band_counts = band_counts.sort_values("time_band")

            fig = px.bar(
                band_counts, x="time_band", y="count",
                title="Overcrowding Events by Time Band",
                labels={"count": "Events", "time_band": "Time Band"},
                color="time_band"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Persistent overcrowding
    st.markdown('<div class="sub-header">Persistent Overcrowding Routes</div>', unsafe_allow_html=True)
    if not persistent.empty:
        display_cols = ["route_id", "crowded_days", "critical_days", "all_crowded_trips", "peak_crowding_bands"]
        available = [c for c in display_cols if c in persistent.columns]
        df = persistent[available].copy()
        st.dataframe(df.sort_values("crowded_days", ascending=False), use_container_width=True)

        # Chart
        fig = px.bar(
            persistent.head(20), x="route_id", y="crowded_days",
            color="critical_days", title="Top 20 Persistently Overcrowded Routes",
            labels={"crowded_days": "Crowded Days", "route_id": "Route"},
            color_continuous_scale="Reds"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No persistent overcrowding detected.")

    # Overcrowding events table
    st.markdown('<div class="sub-header">All Overcrowding Events</div>', unsafe_allow_html=True)
    if not overcrowding.empty:
        display_cols = ["route_id", "date", "time_band", "severity", "trips_total",
                        "crowded_trips", "crowded_share", "avg_occupancy_max_pct"]
        available = [c for c in display_cols if c in overcrowding.columns]
        st.dataframe(overcrowding[available].sort_values("date", ascending=False).head(100), use_container_width=True)


def render_underutilization(underutil: pd.DataFrame, route_m: pd.DataFrame):
    """Render underutilization analysis."""
    if underutil.empty:
        st.info("No underutilization detected.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="sub-header">Underutilized Route-Timeband Pairs</div>', unsafe_allow_html=True)
        display_cols = ["route_id", "time_band", "trips_total", "underused_trips",
                        "underused_share", "avg_occupancy_pct", "avg_boardings"]
        available = [c for c in display_cols if c in underutil.columns]
        st.dataframe(underutil[available].sort_values("underused_share", ascending=False), use_container_width=True)

    with col2:
        st.markdown('<div class="sub-header">Underutilization by Time Band</div>', unsafe_allow_html=True)
        band_counts = underutil.groupby("time_band").size().reset_index(name="count")
        band_order = ["night", "morning peak", "midday", "evening peak"]
        band_counts["time_band"] = pd.Categorical(band_counts["time_band"], categories=band_order, ordered=True)
        band_counts = band_counts.sort_values("time_band")

        fig = px.bar(
            band_counts, x="time_band", y="count",
            title="Underutilized Route-Bands by Time Band",
            labels={"count": "Count", "time_band": "Time Band"},
            color="time_band"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Route-level summary
    st.markdown('<div class="sub-header">Routes with Multiple Underutilized Time Bands</div>', unsafe_allow_html=True)
    by_route = underutil.groupby("route_id").agg(
        underutil_bands=("time_band", "count"),
        avg_occupancy=("avg_occupancy_pct", "mean"),
        avg_boardings=("avg_boardings", "mean"),
    ).reset_index()

    multi_band = by_route[by_route["underutil_bands"] >= 2]
    if not multi_band.empty:
        fig = px.scatter(
            multi_band, x="avg_boardings", y="avg_occupancy",
            size="underutil_bands", color="underutil_bands",
            hover_data=["route_id"],
            title="Routes with Multiple Underutilized Bands",
            labels={"avg_boardings": "Avg Boardings", "avg_occupancy": "Avg Occupancy %"}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        display_cols = ["route_id", "underutil_bands", "avg_occupancy", "avg_boardings"]
        available = [c for c in display_cols if c in multi_band.columns]
        st.dataframe(multi_band[available].sort_values("underutil_bands", ascending=False), use_container_width=True)


def render_stop_analysis(stop_m: pd.DataFrame, route_m: pd.DataFrame):
    """Render stop-level analysis."""
    if stop_m.empty:
        st.info("No stop metrics available.")
        return

    st.markdown('<div class="sub-header">Stop Bottleneck Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Bottleneck score distribution
        fig = px.histogram(
            stop_m, x="bottleneck_score", nbins=30,
            title="Stop Bottleneck Score Distribution",
            labels={"bottleneck_score": "Bottleneck Score", "count": "Stops"},
            color_discrete_sequence=["#1f77b4"]
        )
        # Add threshold line
        threshold = stop_m["bottleneck_score"].quantile(0.6)
        fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                     annotation_text=f"60th percentile: {threshold:.1f}")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Bottleneck by zone
        zone_bottlenecks = stop_m[stop_m["is_bottleneck"] == True].groupby("zone").size().reset_index(name="count")
        if not zone_bottlenecks.empty:
            fig = px.bar(
                zone_bottlenecks, x="zone", y="count",
                title="Bottleneck Stops by Zone",
                labels={"count": "Bottleneck Stops", "zone": "Zone"},
                color="zone"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Bottleneck stops table
    st.markdown('<div class="sub-header">Bottleneck Stops (Top 50)</div>', unsafe_allow_html=True)
    bottlenecks = stop_m[stop_m["is_bottleneck"] == True].sort_values("bottleneck_score", ascending=False)
    if not bottlenecks.empty:
        display_cols = ["stop_id", "stop_name", "zone", "bottleneck_score",
                        "boardings", "avg_arr_delay_min", "p95_arr_delay_min",
                        "crowding_share", "avg_dwell_sec", "peak_hour"]
        available = [c for c in display_cols if c in bottlenecks.columns]
        df = bottlenecks[available].head(50).copy()
        if "crowding_share" in df.columns:
            df["crowding_share"] = (df["crowding_share"] * 100).round(1)
        st.dataframe(df, use_container_width=True)

    # Stop time-band analysis
    st.markdown('<div class="sub-header">Stop Performance by Time Band</div>', unsafe_allow_html=True)
    stop_tb = st.session_state.get("stop_timeband")
    if stop_tb is None:
        from src.services.data_access import load_stop_metrics
        stop_tb = load_stop_metrics()  # This loads stop_metrics, need stop_timeband
        # Actually we need to load stop_timeband separately
        from src.paths import DATA_ANALYTICS
        from src.storage import read_dataset
        try:
            stop_tb = read_dataset(DATA_ANALYTICS / "stop_timeband.parquet")
            st.session_state["stop_timeband"] = stop_tb
        except:
            stop_tb = pd.DataFrame()

    if not stop_tb.empty:
        # Aggregate by time_band
        tb_agg = stop_tb.groupby("time_band").agg(
            avg_arr_delay=("avg_arr_delay_min", "mean"),
            crowding_share=("crowding_share", "mean"),
            total_boardings=("boardings", "sum"),
        ).reset_index()

        fig = px.bar(
            tb_agg, x="time_band", y="avg_arr_delay",
            title="Average Arrival Delay by Time Band",
            labels={"avg_arr_delay": "Avg Arr Delay (min)", "time_band": "Time Band"},
            color="time_band"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def render_capacity_planning(route_m: pd.DataFrame, crowding: dict):
    """Render capacity planning recommendations."""
    st.markdown('<div class="sub-header">Capacity Utilization Overview</div>', unsafe_allow_html=True)

    # Occupancy vs capacity scatter
    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            route_m, x="passengers", y="occupancy_avg_pct",
            size="trips", color="category",
            hover_data=["route_id", "route_name", "avg_boardings_per_trip"],
            labels={"passengers": "Annual Passengers", "occupancy_avg_pct": "Avg Occupancy %"},
            title="Passenger Volume vs Occupancy"
        )
        fig.add_hline(y=85, line_dash="dash", line_color="red", annotation_text="High Crowding (85%)")
        fig.add_hline(y=25, line_dash="dash", line_color="blue", annotation_text="Underutilized (25%)")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Boardings per trip vs capacity
        fig = px.scatter(
            route_m, x="avg_boardings_per_trip", y="occupancy_max_pct",
            size="trips", color="category",
            hover_data=["route_id", "route_name"],
            labels={"avg_boardings_per_trip": "Avg Boardings/Trip", "occupancy_max_pct": "Max Occupancy %"},
            title="Boardings per Trip vs Max Occupancy"
        )
        fig.add_hline(y=85, line_dash="dash", line_color="red", annotation_text="High Crowding")
        fig.add_hline(y=25, line_dash="dash", line_color="blue", annotation_text="Underutilized")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    # Capacity pressure matrix
    st.markdown('<div class="sub-header">Capacity Pressure Matrix</div>', unsafe_allow_html=True)

    # Classify routes by pressure
    def classify_pressure(row):
        occ_max = row["occupancy_max_pct"]
        crowding = row["crowding_share"]
        underutil = row["underutilized_share"]

        if occ_max >= 100 or crowding > 0.3:
            return "Critical Pressure"
        elif occ_max >= 85 or crowding > 0.2:
            return "High Pressure"
        elif underutil > 0.5:
            return "Underutilized"
        elif occ_max >= 70:
            return "Moderate Pressure"
        else:
            return "Normal"

    route_m["pressure"] = route_m.apply(classify_pressure, axis=1)
    pressure_counts = route_m["pressure"].value_counts().reset_index()
    pressure_counts.columns = ["Pressure Level", "Routes"]

    fig = px.pie(
        pressure_counts, values="Routes", names="Pressure Level",
        title="Routes by Capacity Pressure Level",
        color="Pressure Level",
        color_discrete_map={
            "Critical Pressure": "#8b0000",
            "High Pressure": "#dc3545",
            "Moderate Pressure": "#fd7e14",
            "Normal": "#1f77b4",
            "Underutilized": "#6c757d",
        }
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Recommended actions by pressure
    st.markdown('<div class="sub-header">Recommended Actions by Pressure Level</div>', unsafe_allow_html=True)

    pressure_actions = {
        "Critical Pressure": [
            "Immediate capacity increase (articulated buses)",
            "Frequency increase during peak bands",
            "Express service to bypass crowded segments",
            "Demand management (peak pricing, alternative routing)",
        ],
        "High Pressure": [
            "Add trips during peak crowding bands",
            "Consider articulated buses",
            "Monitor for escalation to critical",
        ],
        "Moderate Pressure": [
            "Monitor crowding trends",
            "Optimize headway management",
            "Review stop dwell times",
        ],
        "Normal": [
            "Maintain current service levels",
            "Standard monitoring",
        ],
        "Underutilized": [
            "Review route alignment and frequency",
            "Consider vehicle reallocation to high-pressure routes",
            "Demand stimulation (marketing, fare incentives)",
        ],
    }

    for pressure, actions in pressure_actions.items():
        count = (route_m["pressure"] == pressure).sum()
        if count > 0:
            with st.expander(f"{pressure} ({count} routes)"):
                for action in actions:
                    st.write(f"• {action}")

    # Route-level detail table
    st.markdown('<div class="sub-header">Route Capacity Details</div>', unsafe_allow_html=True)

    detail_cols = ["route_id", "route_name", "category", "pressure", "passengers",
                   "avg_boardings_per_trip", "occupancy_avg_pct", "occupancy_max_pct",
                   "crowding_share", "underutilized_share", "avg_capacity"]
    available = [c for c in detail_cols if c in route_m.columns]
    df = route_m[available].copy()
    if "crowding_share" in df.columns:
        df["crowding_share"] = (df["crowding_share"] * 100).round(1)
    if "underutilized_share" in df.columns:
        df["underutilized_share"] = (df["underutilized_share"] * 100).round(1)

    st.dataframe(df.sort_values("pressure"), use_container_width=True, height=500)