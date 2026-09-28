"""Frequency Analysis Dashboard Page."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.services.data_access import load_route_metrics, load_frequency_analytics


def render():
    st.markdown('<div class="main-header">🕐 Frequency Analysis</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background: rbg(0 0 0); padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <strong>🕐 Frequency Analysis:</strong> Service frequency metrics computed from actual trip data.
    Shows scheduled vs actual trips per route/date/time-period, headway reliability, and peak/off-peak frequency ratios.
    </div>
    """, unsafe_allow_html=True)

    route_m = load_route_metrics()
    freq_data = load_frequency_analytics()

    freq_analysis = freq_data.get("frequency_analysis", pd.DataFrame())
    headway_analysis = freq_data.get("headway_analysis", pd.DataFrame())
    service_level = freq_data.get("service_level", pd.DataFrame())
    peak_analysis = freq_data.get("frequency_peak_analysis", pd.DataFrame())

    if freq_analysis.empty:
        st.warning("No frequency analysis data available. Run analytics pipeline first.")
        return

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_trips_hour = freq_analysis["trips_per_hour"].mean()
        st.metric("Avg Trips/Hour", f"{avg_trips_hour:.1f}")
    with col2:
        avg_headway_reliability = freq_analysis["headway_reliability"].mean() * 100
        st.metric("Headway Reliability", f"{avg_headway_reliability:.1f}%")
    with col3:
        avg_service_reliability = service_level["service_reliability"].mean() if not service_level.empty else 0
        st.metric("Service Reliability", f"{avg_service_reliability:.1f}")
    with col4:
        peak_ratio = peak_analysis["peak_ratio"].mean()
        st.metric("Peak/Off-Peak Ratio", f"{peak_ratio:.2f}")

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Service Frequency", "🚌 Headway Analysis", "📈 Service Level", "🏔️ Peak Analysis"])

    with tab1:
        render_frequency_analysis(freq_analysis, route_m)

    with tab2:
        render_headway_analysis(headway_analysis, route_m)

    with tab3:
        render_service_level(service_level, route_m)

    with tab4:
        render_peak_analysis(peak_analysis, route_m)


def render_frequency_analysis(freq: pd.DataFrame, route_m: pd.DataFrame):
    """Render service frequency analysis."""
    st.markdown('<div class="sub-header">Service Frequency by Route & Time Period</div>', unsafe_allow_html=True)

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        all_routes = sorted(freq["route_id"].unique().tolist())
        selected_routes = st.multiselect("Filter Routes", all_routes, default=[], key="freq_route_filter")
    with col2:
        time_periods = ["morning peak", "midday", "evening peak", "night"]
        selected_periods = st.multiselect("Filter Time Periods", time_periods, default=time_periods, key="freq_period_filter")

    # Apply filters
    filtered = freq.copy()
    if selected_routes:
        filtered = filtered[filtered["route_id"].isin(selected_routes)]
    if selected_periods:
        filtered = filtered[filtered["time_period"].isin(selected_periods)]

    # Trips per hour by time period
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(
            filtered, x="time_period", y="trips_per_hour",
            title="Trips per Hour by Time Period",
            labels={"trips_per_hour": "Trips/Hour", "time_period": "Time Period"},
            color="time_period",
            category_orders={"time_period": ["morning peak", "midday", "evening peak", "night"]}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Scheduled vs actual trips
        fig = px.box(
            filtered, x="time_period", y="scheduled_trips",
            title="Scheduled Trips by Time Period",
            labels={"scheduled_trips": "Scheduled Trips", "time_period": "Time Period"},
            color="time_period",
            category_orders={"time_period": ["morning peak", "midday", "evening peak", "night"]}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Headway reliability
    st.markdown('<div class="sub-header">Headway Reliability by Time Period</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(
            filtered, x="time_period", y="headway_reliability",
            title="Headway Reliability (within 20% of scheduled)",
            labels={"headway_reliability": "Reliability", "time_period": "Time Period"},
            color="time_period",
            category_orders={"time_period": ["morning peak", "midday", "evening peak", "night"]}
        )
        fig.update_yaxes(range=[0, 1])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Service gap
        fig = px.box(
            filtered, x="time_period", y="service_gap_min",
            title="Service Gap (Actual - Scheduled Headway)",
            labels={"service_gap_min": "Gap (min)", "time_period": "Time Period"},
            color="time_period",
            category_orders={"time_period": ["morning peak", "midday", "evening peak", "night"]}
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Frequency table
    st.markdown('<div class="sub-header">Frequency Details</div>', unsafe_allow_html=True)
    display_cols = ["route_id", "date", "time_period", "scheduled_trips", "trips_per_hour",
                    "avg_scheduled_headway_min", "headway_reliability", "service_gap_min",
                    "headway_cv", "is_weekend"]
    available = [c for c in display_cols if c in filtered.columns]

    df_display = filtered[available].copy()
    if "headway_reliability" in df_display.columns:
        df_display["headway_reliability"] = (df_display["headway_reliability"] * 100).round(1)
    if "is_weekend" in df_display.columns:
        df_display["day_type"] = df_display["is_weekend"].map({True: "Weekend", False: "Weekday"})

    st.dataframe(df_display.sort_values(["route_id", "date", "time_period"]).head(200),
                 use_container_width=True, height=400)

    # Daily frequency trend
    st.markdown('<div class="sub-header">Daily Frequency Trend (Selected Routes)</div>', unsafe_allow_html=True)
    daily_freq = filtered.groupby(["date", "time_period"]).agg(
        avg_trips_per_hour=("trips_per_hour", "mean"),
        avg_headway_reliability=("headway_reliability", "mean"),
        total_trips=("scheduled_trips", "sum"),
    ).reset_index()

    fig = px.line(
        daily_freq, x="date", y="avg_trips_per_hour", color="time_period",
        title="Average Trips per Hour Over Time",
        labels={"avg_trips_per_hour": "Avg Trips/Hour", "date": "Date", "time_period": "Time Period"},
        category_orders={"time_period": ["morning peak", "midday", "evening peak", "night"]}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_headway_analysis(headway: pd.DataFrame, route_m: pd.DataFrame):
    """Render headway analysis."""
    if headway.empty:
        st.warning("No headway analysis data available.")
        return

    st.markdown('<div class="sub-header">Headway Distribution by Route & Time Period</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(
            headway, x="time_period", y="scheduled_mean",
            title="Scheduled Headway by Time Period",
            labels={"scheduled_mean": "Scheduled Headway (min)", "time_period": "Time Period"},
            color="time_period",
            category_orders={"time_period": ["morning peak", "midday", "evening peak", "night"]}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            headway, x="time_period", y="actual_mean",
            title="Actual Headway by Time Period",
            labels={"actual_mean": "Actual Headway (min)", "time_period": "Time Period"},
            color="time_period",
            category_orders={"time_period": ["morning peak", "midday", "evening peak", "night"]}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Headway CV and reliability
    st.markdown('<div class="sub-header">Headway Variability & Reliability</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(
            headway, x="time_period", y="headway_cv",
            title="Headway Coefficient of Variation",
            labels={"headway_cv": "Headway CV", "time_period": "Time Period"},
            color="time_period",
            category_orders={"time_period": ["morning peak", "midday", "evening peak", "night"]}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            headway, x="time_period", y="headway_reliability",
            title="Headway Reliability",
            labels={"headway_reliability": "Reliability", "time_period": "Time Period"},
            color="time_period",
            category_orders={"time_period": ["morning peak", "midday", "evening peak", "night"]}
        )
        fig.update_yaxes(range=[0, 1])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Headway table
    st.markdown('<div class="sub-header">Headway Details by Route</div>', unsafe_allow_html=True)
    display_cols = ["route_id", "time_period", "scheduled_mean", "scheduled_std",
                    "actual_mean", "actual_std", "headway_cv", "headway_reliability", "trips"]
    available = [c for c in display_cols if c in headway.columns]

    df_display = headway[available].copy()
    if "headway_reliability" in df_display.columns:
        df_display["headway_reliability"] = (df_display["headway_reliability"] * 100).round(1)
    if "headway_cv" in df_display.columns:
        df_display["headway_cv"] = df_display["headway_cv"].round(3)

    st.dataframe(df_display.sort_values(["route_id", "time_period"]), use_container_width=True, height=500)


def render_service_level(service: pd.DataFrame, route_m: pd.DataFrame):
    """Render service level analysis."""
    if service.empty:
        st.warning("No service level data available.")
        return

    st.markdown('<div class="sub-header">Service Level Metrics by Route & Time Period</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(
            service, x="time_period", y="service_regularity",
            title="Service Regularity (Inverse Headway CV)",
            labels={"service_regularity": "Regularity Score", "time_period": "Time Period"},
            color="time_period",
            category_orders={"time_period": ["morning peak", "midday", "evening peak", "night"]}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            service, x="time_period", y="trips_per_hour",
            title="Service Availability (Trips/Hour)",
            labels={"trips_per_hour": "Trips/Hour", "time_period": "Time Period"},
            color="time_period",
            category_orders={"time_period": ["morning peak", "midday", "evening peak", "night"]}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Service reliability composite
    st.markdown('<div class="sub-header">Composite Service Reliability</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(
            service, x="time_period", y="service_reliability",
            title="Service Reliability Score (Composite)",
            labels={"service_reliability": "Reliability Score", "time_period": "Time Period"},
            color="time_period",
            category_orders={"time_period": ["morning peak", "midday", "evening peak", "night"]}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            service, x="time_period", y="on_time_rate",
            title="On-Time Rate",
            labels={"on_time_rate": "On-Time Rate", "time_period": "Time Period"},
            color="time_period",
            category_orders={"time_period": ["morning peak", "midday", "evening peak", "night"]}
        )
        fig.update_yaxes(range=[0, 1])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Service level table
    st.markdown('<div class="sub-header">Service Level Details</div>', unsafe_allow_html=True)
    display_cols = ["route_id", "date", "time_period", "service_regularity", "trips_per_hour",
                    "on_time_rate", "headway_reliability", "service_reliability"]
    available = [c for c in display_cols if c in service.columns]

    df_display = service[available].copy()
    if "on_time_rate" in df_display.columns:
        df_display["on_time_rate"] = (df_display["on_time_rate"] * 100).round(1)
    if "headway_reliability" in df_display.columns:
        df_display["headway_reliability"] = (df_display["headway_reliability"] * 100).round(1)

    st.dataframe(df_display.sort_values(["route_id", "date", "time_period"]).head(200),
                 use_container_width=True, height=400)


def render_peak_analysis(peak: pd.DataFrame, route_m: pd.DataFrame):
    """Render peak/off-peak frequency analysis."""
    if peak.empty:
        st.warning("No peak analysis data available.")
        return

    st.markdown('<div class="sub-header">Peak vs Off-Peak Frequency Analysis</div>', unsafe_allow_html=True)

    # Merge with route metrics for context
    merged = peak.merge(route_m[["route_id", "route_name", "category", "route_score"]], on="route_id", how="left")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(
            merged, x="offpeak_freq", y="peak_freq",
            size="route_score", color="category",
            hover_data=["route_id", "route_name", "peak_ratio"],
            title="Peak vs Off-Peak Frequency",
            labels={"peak_freq": "Peak Frequency (trips)", "offpeak_freq": "Off-Peak Frequency (trips)"}
        )
        # Add diagonal line
        max_val = max(merged["peak_freq"].max(), merged["offpeak_freq"].max())
        fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines",
                                 line=dict(dash="dash", color="gray"), name="Equal"))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            merged, x="peak_ratio", nbins=20,
            title="Peak/Off-Peak Ratio Distribution",
            labels={"peak_ratio": "Peak Ratio", "count": "Routes"},
            color_discrete_sequence=["#1f77b4"]
        )
        fig.add_vline(x=1, line_dash="dash", line_color="gray", annotation_text="Ratio = 1")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Peak ratio by category
    st.markdown('<div class="sub-header">Peak Ratio by Route Category</div>', unsafe_allow_html=True)
    fig = px.box(
        merged, x="category", y="peak_ratio",
        title="Peak/Off-Peak Ratio by Category",
        labels={"peak_ratio": "Peak Ratio", "category": "Category"},
        color="category"
    )
    fig.add_hline(y=1, line_dash="dash", line_color="gray")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Peak analysis table
    st.markdown('<div class="sub-header">Peak Analysis Details</div>', unsafe_allow_html=True)
    display_cols = ["route_id", "route_name", "category", "peak_freq", "offpeak_freq", "peak_ratio", "route_score"]
    available = [c for c in display_cols if c in merged.columns]

    df_display = merged[available].copy()
    if "peak_ratio" in df_display.columns:
        df_display["peak_ratio"] = df_display["peak_ratio"].round(2)

    st.dataframe(df_display.sort_values("peak_ratio", ascending=False), use_container_width=True, height=500)


if __name__ == "__main__":
    render()