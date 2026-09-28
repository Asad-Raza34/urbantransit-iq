"""Map Dashboard Page."""

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

from src.map.network_map import create_network_map, get_map_metric_options, render_map_html
from src.services.data_access import load_route_metrics, load_stop_metrics


def render():
    st.markdown('<div class="main-header">🗺️ Network Map</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background: #e7f3ff; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <strong>🗺️ Interactive Network Map:</strong> Visualize the transit network with real stop coordinates and route geometries.
    Select a metric mode to color routes/stops by performance indicators. Heatmap modes show stop-level intensity.
    </div>
    """, unsafe_allow_html=True)

    # Load route metrics for route selection
    route_m = load_route_metrics()
    all_route_ids = sorted(route_m["route_id"].unique().tolist())

    # Controls
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        metric_options = get_map_metric_options()
        metric_labels = {opt["value"]: opt["label"] for opt in metric_options}
        # ``.get`` rather than ``[]``: the widget value persists in session state
        # keyed by "map_metric_select", so a stale value from an older option set
        # must degrade to showing the raw value instead of raising KeyError.
        selected_metric = st.selectbox(
            "Metric Mode",
            options=[opt["value"] for opt in metric_options],
            format_func=lambda x: metric_labels.get(x, str(x)),
            index=0,
            key="map_metric_select",
        )

    with col2:
        route_ids = st.multiselect(
            "Filter Routes",
            all_route_ids,
            default=st.session_state.filters.get("route_ids", []),
            key="map_route_filter",
        )

    with col3:
        show_stops = st.checkbox("Show Stops", value=True, key="map_show_stops")

    with col4:
        show_routes = st.checkbox("Show Routes", value=True, key="map_show_routes")

    with col5:
        show_heatmap = st.checkbox("Show Heatmap", value=False, key="map_show_heatmap",
                                   help="Enable heatmap layer for stop-level metrics (available in heatmap modes)")

    show_bottlenecks = st.checkbox("Show Bottlenecks", value=False, key="map_show_bottlenecks")

    # Generate map
    if st.button("🗺️ Generate Map", type="primary", key="generate_map_btn"):
        with st.spinner("Generating network map..."):
            map_obj = create_network_map(
                metric_mode=selected_metric,
                route_filter=route_ids if route_ids else None,
                show_bottlenecks=selected_metric == "bottleneck" and show_bottlenecks,
                show_stops=show_stops,
                show_routes=show_routes,
                show_heatmap=show_heatmap,
            )

            # Render in memory. This used to write and re-read an ~800 KB
            # "temp_map.html" in the working directory on every click, which
            # polluted the project folder and let concurrent sessions clash.
            html_content = render_map_html(map_obj)

            st.components.v1.html(html_content, height=700, scrolling=True)
            st.success("Map generated successfully!")

    # Metric explanation
    st.markdown('<div class="sub-header">Metric Modes</div>', unsafe_allow_html=True)

    metric_info = {
        "default": "Default view showing routes colored by category (Core/Feeder/Express). Stops shown in blue.",
        "demand": "Routes/stops colored by passenger demand (total boardings). Red = high demand.",
        "delay": "Routes colored by average delay in minutes. Red = higher delays.",
        "crowding": "Routes colored by crowding share (% trips over 85% capacity). Red = more crowding.",
        "occupancy": "Routes colored by maximum occupancy percentage. Red = high occupancy.",
        "reliability": "Routes colored by on-time share. Green = higher reliability.",
        "bottleneck": "Shows stop bottlenecks as red warning markers. Routes in category colors.",
        "heatmap_demand": "Heatmap showing passenger demand intensity at stops. Red = high demand concentration.",
        "heatmap_crowding": "Heatmap showing crowding intensity at stops. Red = high crowding concentration.",
        "heatmap_delay": "Heatmap showing delay intensity at stops. Red = high delay concentration.",
        "heatmap_occupancy": "Heatmap showing occupancy intensity at stops. Red = high occupancy concentration.",
    }

    for key, desc in metric_info.items():
        with st.expander(f"{metric_labels.get(key, key)}"):
            st.write(desc)


if __name__ == "__main__":
    render()