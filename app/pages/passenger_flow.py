"""Passenger Flow Dashboard Page."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.services.data_access import load_demand_analytics, load_route_metrics, load_od_matrices
from src.comparison.routes import get_route_profile


def render():
    st.markdown('<div class="main-header">👥 Passenger Flow</div>', unsafe_allow_html=True)

    # Load data
    demand_data = load_demand_analytics()
    daily_demand = demand_data.get("daily_demand", pd.DataFrame())
    peak_patterns = demand_data.get("peak_patterns", pd.DataFrame())
    anomalies = demand_data.get("anomalies", pd.DataFrame())
    od_zone = demand_data.get("od_zone_flow", pd.DataFrame())

    route_m = load_route_metrics()

    # Route selector for drill-down
    route_ids = sorted(route_m["route_id"].unique().tolist())
    selected_route = st.selectbox("Select Route for Details", ["All"] + route_ids, key="passenger_flow_route")

    if selected_route != "All":
        profile = get_route_profile(selected_route)
        render_route_profile(profile)
        return

    # KPIs
    col1, col2, col3, col4 = st.columns(4)

    if not daily_demand.empty:
        with col1:
            st.metric("Avg Daily Passengers", f"{daily_demand.groupby('date')['passengers'].sum().mean():,.0f}")
        with col2:
            st.metric("Peak Daily Passengers", f"{daily_demand.groupby('date')['passengers'].sum().max():,.0f}")
        with col3:
            st.metric("Avg Daily Revenue", f"${daily_demand.groupby('date')['revenue'].sum().mean():,.0f}")
        with col4:
            st.metric("Total Routes", len(route_m))

    st.markdown("---")

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Daily Demand", "🕐 Peak Patterns", "🔍 Anomalies", "🔄 OD Flow"])

    with tab1:
        render_daily_demand(daily_demand)

    with tab2:
        render_peak_patterns(peak_patterns, demand_data.get("peak_summary", pd.DataFrame()))

    with tab3:
        render_anomalies(anomalies)

    with tab4:
        render_od_flow(od_zone)


def render_daily_demand(daily_demand: pd.DataFrame):
    """Render daily demand trends."""
    if daily_demand.empty:
        st.info("No daily demand data available.")
        return

    # Aggregate city-wide
    city_daily = daily_demand.groupby("date").agg(
        passengers=("passengers", "sum"),
        revenue=("revenue", "sum"),
    ).reset_index()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="sub-header">City-Wide Daily Passengers</div>', unsafe_allow_html=True)
        fig = px.line(
            city_daily, x="date", y="passengers",
            title="Daily Passenger Boardings",
            labels={"passengers": "Passengers", "date": "Date"}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="sub-header">Weekday vs Weekend</div>', unsafe_allow_html=True)
        daily_demand["weekday"] = pd.to_datetime(daily_demand["date"]).dt.weekday
        daily_demand["day_type"] = daily_demand["weekday"].apply(lambda x: "Weekend" if x >= 5 else "Weekday")
        day_type = daily_demand.groupby("day_type")["passengers"].mean().reset_index()

        fig = px.bar(
            day_type, x="day_type", y="passengers",
            title="Average Daily Passengers by Day Type",
            labels={"passengers": "Avg Passengers", "day_type": "Day Type"},
            color="day_type"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Per-route daily demand heatmap (top routes)
    st.markdown('<div class="sub-header">Top Routes - Daily Demand Heatmap</div>', unsafe_allow_html=True)
    route_m = load_route_metrics()
    top_routes = route_m.nlargest(15, "passengers")["route_id"].tolist()
    top_demand = daily_demand[daily_demand["route_id"].isin(top_routes)]

    if not top_demand.empty:
        pivot = top_demand.pivot_table(
            index="route_id", columns="date", values="passengers", aggfunc="sum"
        ).fillna(0)

        fig = px.imshow(
            pivot, aspect="auto",
            title="Daily Passengers by Route (Top 15)",
            labels=dict(x="Date", y="Route", color="Passengers"),
            color_continuous_scale="Blues"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)


def render_peak_patterns(peak_patterns: pd.DataFrame, peak_summary: pd.DataFrame):
    """Render peak period analysis."""
    if peak_patterns.empty:
        st.info("No peak pattern data available.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="sub-header">Demand Share by Time Band</div>', unsafe_allow_html=True)
        # Aggregate across routes
        band_share = peak_patterns.groupby("time_band")["band_share"].mean().reset_index()
        band_order = ["night", "morning peak", "midday", "evening peak"]
        band_share["time_band"] = pd.Categorical(band_share["time_band"], categories=band_order, ordered=True)
        band_share = band_share.sort_values("time_band")

        fig = px.bar(
            band_share, x="time_band", y="band_share",
            title="Average Demand Share by Time Band",
            labels={"band_share": "Share of Daily Demand", "time_band": "Time Band"},
            color="time_band"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="sub-header">Primary & Secondary Peaks by Route</div>', unsafe_allow_html=True)
        if "primary_peak" in peak_patterns.columns:
            peak_counts = peak_patterns.groupby("primary_peak").size().reset_index(name="count")
            fig = px.pie(
                peak_counts, values="count", names="primary_peak",
                title="Routes by Primary Peak Period"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Per-route peak detail
    st.markdown('<div class="sub-header">Route Peak Details</div>', unsafe_allow_html=True)
    display_cols = ["route_id", "time_band", "band_share", "primary_peak", "secondary_peak"]
    available = [c for c in display_cols if c in peak_patterns.columns]
    st.dataframe(peak_patterns[available].head(50), use_container_width=True)


def render_anomalies(anomalies: pd.DataFrame):
    """Render demand anomalies."""
    if anomalies.empty:
        st.info("No demand anomalies detected.")
        return

    st.markdown('<div class="sub-header">Demand Anomalies (Z-Score Based)</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            anomalies, x="z_score", color="anomaly_type",
            title="Anomaly Z-Score Distribution",
            labels={"z_score": "Z-Score", "count": "Number of Anomalies"}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        anomaly_counts = anomalies.groupby(["route_id", "anomaly_type"]).size().reset_index(name="count")
        fig = px.bar(
            anomaly_counts, x="route_id", y="count", color="anomaly_type",
            title="Anomalies by Route",
            labels={"count": "Anomaly Count", "route_id": "Route"}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Anomaly table
    st.markdown('<div class="sub-header">Anomaly Details</div>', unsafe_allow_html=True)
    display_cols = ["route_id", "date", "passengers", "z_score", "anomaly_type", "is_weekend"]
    available = [c for c in display_cols if c in anomalies.columns]
    anomalies_sorted = anomalies.sort_values("z_score", key=abs, ascending=False)
    st.dataframe(anomalies_sorted[available].head(100), use_container_width=True)


def render_od_flow(od_zone: pd.DataFrame):
    """Render origin-destination zone flow."""
    if od_zone.empty:
        st.info("No OD zone flow data available.")
        return

    st.markdown('<div class="sub-header">Zone-to-Zone Passenger Flow</div>', unsafe_allow_html=True)

    # Sankey-like visualization using bar chart
    col1, col2 = st.columns(2)

    with col1:
        top_od = od_zone.head(20)
        fig = px.bar(
            top_od, x="passengers", y="boarding_zone",
            color="alighting_zone", orientation="h",
            title="Top 20 Zone-to-Zone Flows",
            labels={"passengers": "Passengers", "boarding_zone": "Origin Zone"}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Zone summary
        zone_summary = od_zone.groupby("boarding_zone").agg(
            outbound=("passengers", "sum"),
            outbound_revenue=("revenue", "sum"),
        ).reset_index()

        inbound = od_zone.groupby("alighting_zone").agg(
            inbound=("passengers", "sum"),
        ).reset_index().rename(columns={"alighting_zone": "zone"})

        zone_summary = zone_summary.merge(inbound, left_on="boarding_zone", right_on="zone", how="outer")
        zone_summary = zone_summary.fillna(0)
        zone_summary["net_flow"] = zone_summary["outbound"] - zone_summary["inbound"]

        fig = px.bar(
            zone_summary, x="zone", y=["outbound", "inbound"],
            title="Zone Inbound vs Outbound Passengers",
            barmode="group", labels={"value": "Passengers", "zone": "Zone"}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Full OD matrix table
    st.markdown('<div class="sub-header">Full OD Matrix</div>', unsafe_allow_html=True)
    st.dataframe(od_zone.head(100), use_container_width=True)


def render_route_profile(profile: dict):
    """Render detailed route profile."""
    st.markdown(f'<div class="main-header">Route Profile: {profile["route_id"]}</div>', unsafe_allow_html=True)

    basic = profile["basic_info"]
    perf = profile["performance_summary"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Route Score", f"{perf['route_score']:.1f}")
    with col2:
        st.metric("Classification", perf["route_class"])
    with col3:
        st.metric("Passengers", f"{perf['passengers']:,}")
    with col4:
        st.metric("Avg Delay", f"{perf['avg_delay_min']:.1f} min")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("On-Time %", f"{perf['on_time_pct']:.1f}%")
    with col2:
        st.metric("Avg Occupancy", f"{perf['occupancy_avg_pct']:.1f}%")
    with col3:
        st.metric("Max Occupancy", f"{perf['occupancy_max_pct']:.1f}%")
    with col4:
        st.metric("Bunching %", f"{perf['bunching_share']:.1f}%")

    st.markdown("---")

    # Crowding details
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="sub-header">Crowding Analysis</div>', unsafe_allow_html=True)
        crowding = profile["crowding"]
        st.metric("Total Events", crowding["events"])
        st.metric("Persistent Days", crowding["persistent_days"])
        if crowding["peak_bands"]:
            st.write("**Peak Bands:**", ", ".join(crowding["peak_bands"]))
        if crowding["severity_breakdown"]:
            st.write("**Severity:**", crowding["severity_breakdown"])

    with col2:
        st.markdown('<div class="sub-header">Bunching Analysis</div>', unsafe_allow_html=True)
        bunching = profile["bunching"]
        st.metric("Total Events", bunching["events"])
        st.metric("Event Days", bunching["event_days"])

    # Reliability
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
    if stops["bottlenecks"] > 0:
        st.warning(f"{stops['bottlenecks']} bottleneck stops detected out of {stops['total']} total stops")
        if stops["top_boarding_stops"]:
            st.dataframe(pd.DataFrame(stops["top_boarding_stops"]), use_container_width=True)
    else:
        st.info("No bottleneck stops detected.")

    # Back button
    if st.button("← Back to Overview"):
        st.rerun()