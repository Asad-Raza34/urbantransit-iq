"""Executive Overview Dashboard Page."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.services.data_access import calculate_kpis, load_route_metrics, load_crowding_analytics, load_alerts


def render():
    st.markdown('<div class="main-header">📊 Executive Overview</div>', unsafe_allow_html=True)

    # Calculate KPIs
    kpis = calculate_kpis()

    # Top-level KPI cards
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Total Routes",
            f"{kpis['total_routes']}",
            help="Number of active routes in the network"
        )

    with col2:
        st.metric(
            "Daily Passengers",
            f"{kpis['total_passengers_daily']:,.0f}",
            help="Average daily passenger boardings"
        )

    with col3:
        st.metric(
            "Avg Delay",
            f"{kpis['avg_delay_min']:.1f} min",
            delta=f"{kpis['avg_delay_min'] - 5.0:.1f} vs target",
            delta_color="inverse",
            help="Average departure delay across all routes"
        )

    with col4:
        st.metric(
            "On-Time %",
            f"{kpis['on_time_pct']:.1f}%",
            delta=f"{kpis['on_time_pct'] - 75:.1f}% vs target",
            help="Share of trips departing within 5 minutes of schedule"
        )

    with col5:
        st.metric(
            "Avg Occupancy",
            f"{kpis['avg_occupancy_pct']:.1f}%",
            help="Average occupancy across all trips"
        )

    st.markdown("---")

    # Second row of KPIs
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Overcrowding Events",
            kpis['overcrowding_events'],
            help="Route-date-timeband combinations with >20% crowded trips"
        )

    with col2:
        st.metric(
            "Persistent Overcrowding",
            kpis['persistent_overcrowding_routes'],
            help="Routes with overcrowding on 15+ days"
        )

    with col3:
        st.metric(
            "Underutilized Routes",
            kpis['underutilized_routes'],
            help="Routes with >50% trips underutilized in any time band"
        )

    with col4:
        st.metric(
            "Active Alerts",
            kpis['active_alerts'],
            delta=f"{kpis['critical_alerts']} critical",
            delta_color="inverse",
            help="New alerts requiring attention"
        )

    with col5:
        st.metric(
            "Bottleneck Stops",
            kpis['bottleneck_stops'],
            help="Stops with high delay/crowding/volume scores"
        )

    st.markdown("---")

    # Charts row 1: Route score distribution and top/bottom routes
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="sub-header">Route Score Distribution</div>', unsafe_allow_html=True)
        route_m = load_route_metrics()
        if not route_m.empty:
            fig = px.histogram(
                route_m, x="route_score", nbins=20,
                title="Route Score Distribution (0-100)",
                labels={"route_score": "Route Score", "count": "Number of Routes"},
                color_discrete_sequence=["#1f77b4"]
            )
            fig.add_vline(x=route_m["route_score"].mean(), line_dash="dash",
                         annotation_text=f"Mean: {route_m['route_score'].mean():.1f}")
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="sub-header">Top & Bottom Routes by Score</div>', unsafe_allow_html=True)
        if not route_m.empty:
            top5 = route_m.nlargest(5, "route_score")[["route_id", "route_name", "route_score", "route_class"]]
            bottom5 = route_m.nsmallest(5, "route_score")[["route_id", "route_name", "route_score", "route_class"]]

            combined = pd.concat([top5.assign(rank="Top 5"), bottom5.assign(rank="Bottom 5")])

            fig = px.bar(
                combined, x="route_score", y="route_id", color="rank",
                orientation="h", title="Route Scores",
                labels={"route_score": "Score", "route_id": "Route"},
                color_discrete_map={"Top 5": "#28a745", "Bottom 5": "#dc3545"}
            )
            fig.update_layout(height=350, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

    # Charts row 2: Delay and occupancy trends
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="sub-header">Delay vs On-Time Performance</div>', unsafe_allow_html=True)
        if not route_m.empty:
            fig = px.scatter(
                route_m, x="avg_delay_min", y="on_time_share",
                size="passengers", color="route_class",
                hover_data=["route_id", "route_name", "route_score"],
                labels={"avg_delay_min": "Avg Delay (min)", "on_time_share": "On-Time Share"},
                title="Delay vs Reliability by Route"
            )
            fig.add_hline(y=0.75, line_dash="dash", annotation_text="Degraded Threshold")
            fig.add_hline(y=0.55, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="sub-header">Occupancy vs Crowding</div>', unsafe_allow_html=True)
        if not route_m.empty:
            fig = px.scatter(
                route_m, x="occupancy_avg_pct", y="crowding_share",
                size="passengers", color="route_class",
                hover_data=["route_id", "route_name"],
                labels={"occupancy_avg_pct": "Avg Occupancy %", "crowding_share": "Crowding Share"},
                title="Occupancy vs Crowding by Route"
            )
            fig.add_vline(x=85, line_dash="dash", annotation_text="High Crowding Threshold")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    # Recent alerts summary
    st.markdown('<div class="sub-header">Recent Critical Alerts</div>', unsafe_allow_html=True)
    alerts = load_alerts()
    if not alerts.empty:
        critical_alerts = alerts[(alerts["severity"] == "critical") & (alerts["status"] == "new")].head(5)
        if not critical_alerts.empty:
            for _, alert in critical_alerts.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div class="metric-card alert-critical">
                        <strong>🚨 {alert['title']}</strong><br>
                        {alert['description']}<br>
                        <small>Route: {alert['affected_route'] or 'N/A'} | Type: {alert['alert_type']}</small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No critical alerts at this time.")

    # Route classification summary
    st.markdown('<div class="sub-header">Route Classification Summary</div>', unsafe_allow_html=True)
    if not route_m.empty:
        class_counts = route_m["route_class"].value_counts().reset_index()
        class_counts.columns = ["Classification", "Count"]

        fig = px.pie(
            class_counts, values="Count", names="Classification",
            title="Routes by Classification",
            color="Classification",
            color_discrete_map={
                "Top Performer": "#28a745",
                "Solid": "#1f77b4",
                "Needs Attention": "#ffc107",
                "Action Required": "#dc3545",
            }
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)