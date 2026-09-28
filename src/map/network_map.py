"""Interactive network map for UrbanTransit IQ.

Generates Folium-based maps with stops, routes, and metric overlays
using actual project data (stops with lat/lon, route geometries from stop sequences).
Supports heatmap layers for demand, crowding, and other metrics.
"""

import json
from pathlib import Path
from typing import Any

import folium
import pandas as pd
from folium.plugins import HeatMap, HeatMapWithTime

from config import settings
from src.analytics.core import read_processed
from src.log import get_logger
from src.paths import DATA_CLEANED
from src.storage import read_dataset

logger = get_logger(__name__)


# Color schemes for different metrics
METRIC_COLORS = {
    "demand": "YlOrRd",
    "delay": "Reds",
    "crowding": "OrRd",
    "reliability": "RdYlGn_r",
    "bottleneck": "PuRd",
    "occupancy": "YlOrBr",
}


def _load_stops() -> pd.DataFrame:
    """Load stops with coordinates."""
    return read_dataset(DATA_CLEANED / "stops.parquet")


def _load_routes() -> pd.DataFrame:
    """Load routes reference."""
    return read_dataset(DATA_CLEANED / "routes.parquet")


def _load_route_stops() -> pd.DataFrame:
    """Load route-stop sequences."""
    return read_dataset(DATA_CLEANED / "route_stops.parquet")


def _build_route_geometries(route_stops: pd.DataFrame, stops: pd.DataFrame) -> dict[str, list[tuple[float, float]]]:
    """Build line geometries for each route from stop sequences."""
    stops_indexed = stops.set_index("stop_id")[["lat", "lon"]]
    geometries = {}

    for route_id, group in route_stops.groupby("route_id"):
        group = group.sort_values("seq")
        coords = []
        for _, row in group.iterrows():
            stop_id = row["stop_id"]
            if stop_id in stops_indexed.index:
                lat = stops_indexed.at[stop_id, "lat"]
                lon = stops_indexed.at[stop_id, "lon"]
                coords.append((lat, lon))
        if len(coords) >= 2:
            geometries[route_id] = coords
    return geometries


def _get_route_metrics() -> pd.DataFrame:
    """Load route metrics for metric overlays."""
    try:
        return read_processed("route_metrics")
    except FileNotFoundError:
        return pd.DataFrame()


def _get_stop_metrics() -> pd.DataFrame:
    """Load stop metrics for metric overlays."""
    try:
        return read_processed("stop_metrics")
    except FileNotFoundError:
        return pd.DataFrame()


def _get_crowding_data() -> dict[str, pd.DataFrame]:
    """Load crowding analytics."""
    from src.services.data_access import load_crowding_analytics
    return load_crowding_analytics()


def create_base_map(center: tuple[float, float] = (33.68, 73.05), zoom: int = 11) -> folium.Map:
    """Create a base Folium map centered on the network."""
    return folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="OpenStreetMap",
        control_scale=True,
    )


def add_stops_layer(
    m: folium.Map,
    stops: pd.DataFrame,
    metric: str = None,
    metric_data: pd.DataFrame = None,
    radius_scale: float = 5.0,
) -> folium.FeatureGroup:
    """Add stops as circle markers, optionally colored by a metric."""
    fg = folium.FeatureGroup(name="Stops", show=True)

    if metric and metric_data is not None and not metric_data.empty:
        # Merge metric data with stops
        stops_merged = stops.merge(metric_data, on="stop_id", how="left")
        max_val = stops_merged[metric].max() if metric in stops_merged.columns else 1
        min_val = stops_merged[metric].min() if metric in stops_merged.columns else 0
    else:
        stops_merged = stops
        max_val = 1
        min_val = 0

    for _, row in stops_merged.iterrows():
        lat, lon = row["lat"], row["lon"]
        stop_id = row["stop_id"]
        stop_name = row.get("stop_name", stop_id)
        zone = row.get("zone", "unknown")

        if metric and metric in row and pd.notna(row[metric]):
            val = row[metric]
            # Normalize to 0-1 for color
            if max_val > min_val:
                norm = (val - min_val) / (max_val - min_val)
            else:
                norm = 0.5
            # Color from blue (low) to red (high)
            color = _interpolate_color(norm)
            radius = max(3, radius_scale * (0.5 + norm))
            fill_opacity = 0.7
        else:
            color = "#3388ff"
            radius = 4
            fill_opacity = 0.6

        popup_text = f"<b>{stop_name}</b> ({stop_id})<br>Zone: {zone}"
        if metric and metric in row and pd.notna(row[metric]):
            popup_text += f"<br>{metric}: {row[metric]:.2f}"

        folium.CircleMarker(
            location=(lat, lon),
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=fill_opacity,
            popup=folium.Popup(popup_text, max_width=250),
            tooltip=f"{stop_name} ({stop_id})",
        ).add_to(fg)

    return fg


def _interpolate_color(norm: float) -> str:
    """Interpolate from blue (0) to red (1)."""
    # Blue -> Cyan -> Yellow -> Red
    if norm < 0.25:
        r, g, b = 0, int(255 * norm * 4), 255
    elif norm < 0.5:
        r, g, b = 0, 255, int(255 * (1 - (norm - 0.25) * 4))
    elif norm < 0.75:
        r, g, b = int(255 * (norm - 0.5) * 4), 255, 0
    else:
        r, g, b = 255, int(255 * (1 - (norm - 0.75) * 4)), 0
    return f"#{r:02x}{g:02x}{b:02x}"


def add_routes_layer(
    m: folium.Map,
    route_geometries: dict[str, list[tuple[float, float]]],
    routes: pd.DataFrame,
    metric: str = None,
    metric_data: pd.DataFrame = None,
    weight_scale: float = 3.0,
) -> folium.FeatureGroup:
    """Add route lines, optionally colored by a metric."""
    fg = folium.FeatureGroup(name="Routes", show=True)

    if metric and metric_data is not None and not metric_data.empty:
        max_val = metric_data[metric].max() if metric in metric_data.columns else 1
        min_val = metric_data[metric].min() if metric in metric_data.columns else 0
    else:
        max_val = 1
        min_val = 0

    for route_id, coords in route_geometries.items():
        route_info = routes[routes["route_id"] == route_id]
        route_name = route_info["route_name"].iloc[0] if not route_info.empty else route_id
        category = route_info["category"].iloc[0] if not route_info.empty else "unknown"

        if metric and metric_data is not None and not metric_data.empty:
            row = metric_data[metric_data["route_id"] == route_id]
            if not row.empty and metric in row.columns and pd.notna(row[metric].iloc[0]):
                val = row[metric].iloc[0]
                if max_val > min_val:
                    norm = (val - min_val) / (max_val - min_val)
                else:
                    norm = 0.5
                color = _interpolate_color(norm)
                weight = max(2, weight_scale * (0.5 + norm))
            else:
                color = _category_color(category)
                weight = 3
        else:
            color = _category_color(category)
            weight = 3

        popup_text = f"<b>{route_name}</b> ({route_id})<br>Category: {category}"
        if metric and metric_data is not None and not metric_data.empty:
            row = metric_data[metric_data["route_id"] == route_id]
            if not row.empty and metric in row.columns and pd.notna(row[metric].iloc[0]):
                popup_text += f"<br>{metric}: {row[metric].iloc[0]:.2f}"

        folium.PolyLine(
            locations=coords,
            color=color,
            weight=weight,
            opacity=0.8,
            popup=folium.Popup(popup_text, max_width=250),
            tooltip=f"{route_name} ({route_id})",
        ).add_to(fg)

    return fg


def _category_color(category: str) -> str:
    """Get color for route category."""
    colors = {
        "core": "#1f77b4",
        "feeder": "#2ca02c",
        "express": "#ff7f0e",
    }
    return colors.get(category, "#7f7f7f")


def add_bottleneck_layer(
    m: folium.Map,
    stops: pd.DataFrame,
    stop_metrics: pd.DataFrame,
) -> folium.FeatureGroup:
    """Add bottleneck stops as special markers."""
    fg = folium.FeatureGroup(name="Bottlenecks", show=False)

    if stop_metrics.empty or "is_bottleneck" not in stop_metrics.columns:
        return fg

    bottlenecks = stop_metrics[stop_metrics["is_bottleneck"] == True]
    stops_merged = stops.merge(bottlenecks[["stop_id", "bottleneck_score"]], on="stop_id", how="inner")

    for _, row in stops_merged.iterrows():
        folium.Marker(
            location=(row["lat"], row["lon"]),
            icon=folium.Icon(color="red", icon="exclamation-triangle", prefix="fa"),
            popup=folium.Popup(
                f"<b>{row.get('stop_name', row['stop_id'])}</b> ({row['stop_id']})<br>"
                f"Bottleneck Score: {row['bottleneck_score']:.1f}<br>"
                f"Zone: {row.get('zone', 'unknown')}",
                max_width=250,
            ),
            tooltip=f"Bottleneck: {row.get('stop_name', row['stop_id'])}",
        ).add_to(fg)

    return fg


def add_overcrowding_layer(
    m: folium.Map,
    stops: pd.DataFrame,
    crowding_events: pd.DataFrame,
) -> folium.FeatureGroup:
    """Add stops with overcrowding events as markers."""
    fg = folium.FeatureGroup(name="Overcrowding Events", show=False)

    if crowding_events.empty:
        return fg

    # Aggregate overcrowding by stop
    # We need to join through stop_facts to get stop-level crowding
    # For now, aggregate by route and show on route midpoints
    # This is a simplified version - could be enhanced with stop-level data
    return fg


def add_heatmap_layer(
    m: folium.Map,
    stops: pd.DataFrame,
    metric_data: pd.DataFrame,
    metric_column: str,
    radius: int = 15,
    blur: int = 10,
    max_zoom: int = 13,
) -> folium.FeatureGroup:
    """Add a heatmap layer for stop-level metrics.
    
    Args:
        m: Folium map object
        stops: Stops DataFrame with lat/lon
        metric_data: DataFrame with metric values per stop
        metric_column: Column name for the metric to visualize
        radius: Heatmap radius in pixels
        blur: Heatmap blur in pixels
        max_zoom: Maximum zoom level for heatmap
        
    Returns:
        FeatureGroup containing the heatmap
    """
    fg = folium.FeatureGroup(name=f"Heatmap: {metric_column}", show=False)
    
    if metric_data.empty or metric_column not in metric_data.columns:
        logger.warning(f"Heatmap metric '{metric_column}' not found in data")
        return fg
    
    # Merge stops with metric data
    merged = stops.merge(metric_data[["stop_id", metric_column]], on="stop_id", how="inner")
    merged = merged.dropna(subset=[metric_column, "lat", "lon"])
    
    if merged.empty:
        logger.warning(f"No valid data for heatmap metric '{metric_column}'")
        return fg
    
    # Prepare heatmap data: [[lat, lon, weight], ...]
    # Normalize weights to 0-1 range for better visualization
    max_val = merged[metric_column].max()
    min_val = merged[metric_column].min()
    
    if max_val > min_val:
        merged["weight"] = (merged[metric_column] - min_val) / (max_val - min_val)
    else:
        merged["weight"] = 0.5
    
    heat_data = merged[["lat", "lon", "weight"]].values.tolist()
    
    HeatMap(
        heat_data,
        radius=radius,
        blur=blur,
        max_zoom=max_zoom,
        gradient={0.0: 'blue', 0.25: 'cyan', 0.5: 'yellow', 0.75: 'orange', 1.0: 'red'},
    ).add_to(fg)
    
    return fg


def add_heatmap_with_time_layer(
    m: folium.Map,
    stops: pd.DataFrame,
    metric_data: pd.DataFrame,
    metric_column: str,
    time_column: str = "date",
    radius: int = 15,
    blur: int = 10,
) -> folium.FeatureGroup:
    """Add a time-animated heatmap layer (requires time series data).
    
    Note: This requires metric_data to have a time dimension (date column)
    with multiple time points per stop.
    """
    fg = folium.FeatureGroup(name=f"Heatmap (Time): {metric_column}", show=False)
    
    if metric_data.empty or metric_column not in metric_data.columns:
        logger.warning(f"Time heatmap metric '{metric_column}' not found in data")
        return fg
    
    if time_column not in metric_data.columns:
        logger.warning(f"Time column '{time_column}' not found in data")
        return fg
    
    # Merge stops with metric data
    merged = stops.merge(
        metric_data[["stop_id", time_column, metric_column]], 
        on="stop_id", how="inner"
    )
    merged = merged.dropna(subset=[metric_column, "lat", "lon", time_column])
    
    if merged.empty:
        return fg
    
    # Convert time to string for grouping
    merged[time_column] = pd.to_datetime(merged[time_column])
    merged["time_str"] = merged[time_column].dt.strftime("%Y-%m-%d")
    
    # Normalize weights
    max_val = merged[metric_column].max()
    min_val = merged[metric_column].min()
    if max_val > min_val:
        merged["weight"] = (merged[metric_column] - min_val) / (max_val - min_val)
    else:
        merged["weight"] = 0.5
    
    # Group by time and create heatmap data for each time step
    time_data = []
    for time_str, group in merged.groupby("time_str"):
        heat_points = group[["lat", "lon", "weight"]].values.tolist()
        time_data.append(heat_points)
    
    if time_data:
        HeatMapWithTime(
            time_data,
            radius=radius,
            blur=blur,
            gradient={0.0: 'blue', 0.25: 'cyan', 0.5: 'yellow', 0.75: 'orange', 1.0: 'red'},
            auto_play=False,
            display_index=True,
            index=merged["time_str"].unique().tolist(),
        ).add_to(fg)
    
    return fg


def create_network_map(
    metric_mode: str = "default",
    route_filter: list[str] = None,
    stop_filter: list[str] = None,
    show_bottlenecks: bool = False,
    show_stops: bool = True,
    show_routes: bool = True,
    show_heatmap: bool = False,
) -> folium.Map:
    """Create the main network map with optional metric overlays.

    metric_mode: "default", "demand", "delay", "crowding", "reliability", "bottleneck", "occupancy",
                 "heatmap_demand", "heatmap_crowding", "heatmap_delay", "heatmap_occupancy"
    """
    logger.info(f"Creating network map with metric_mode={metric_mode}")

    # Load base data
    stops = _load_stops()
    routes = _load_routes()
    route_stops = _load_route_stops()

    # Apply filters
    if stop_filter:
        stops = stops[stops["stop_id"].isin(stop_filter)]
    if route_filter:
        routes = routes[routes["route_id"].isin(route_filter)]
        route_stops = route_stops[route_stops["route_id"].isin(route_filter)]

    # Build route geometries
    route_geometries = _build_route_geometries(route_stops, stops)

    # Compute center from stops
    center_lat = stops["lat"].mean()
    center_lon = stops["lon"].mean()

    # Create base map
    m = create_base_map(center=(center_lat, center_lon))

    # Load metric data based on mode
    route_metric_data = None
    stop_metric_data = None
    metric_column = None
    is_heatmap = metric_mode.startswith("heatmap_")
    
    # Extract base metric from heatmap mode
    base_metric = metric_mode.replace("heatmap_", "") if is_heatmap else metric_mode

    if base_metric == "demand":
        route_metrics = _get_route_metrics()
        if not route_metrics.empty:
            route_metric_data = route_metrics
            metric_column = "passengers"
    elif base_metric == "delay":
        route_metrics = _get_route_metrics()
        if not route_metrics.empty:
            route_metric_data = route_metrics
            metric_column = "avg_delay_min"
    elif base_metric == "crowding":
        route_metrics = _get_route_metrics()
        if not route_metrics.empty:
            route_metric_data = route_metrics
            metric_column = "crowding_share"
    elif base_metric == "occupancy":
        route_metrics = _get_route_metrics()
        if not route_metrics.empty:
            route_metric_data = route_metrics
            metric_column = "occupancy_max_pct"
    elif base_metric == "reliability":
        route_metrics = _get_route_metrics()
        if not route_metrics.empty:
            route_metric_data = route_metrics
            metric_column = "on_time_share"
    elif base_metric == "bottleneck":
        stop_metrics = _get_stop_metrics()

    # Load stop-level metrics for heatmaps
    if is_heatmap:
        try:
            stop_metric_data = read_processed("stop_metrics")
        except FileNotFoundError:
            stop_metric_data = pd.DataFrame()

    # Add layers
    if show_routes:
        routes_fg = add_routes_layer(
            m, route_geometries, routes,
            metric=metric_column, metric_data=route_metric_data
        )
        routes_fg.add_to(m)

    if show_stops and not is_heatmap:
        stops_fg = add_stops_layer(
            m, stops,
            metric=metric_column, metric_data=stop_metric_data
        )
        stops_fg.add_to(m)

    if show_heatmap or is_heatmap:
        # Determine heatmap metric column
        heatmap_metric_map = {
            "heatmap_demand": "passengers",
            "heatmap_crowding": "crowding_share",
            "heatmap_delay": "avg_arr_delay_min",
            "heatmap_occupancy": "occupancy_max_pct",
        }
        heatmap_metric = heatmap_metric_map.get(metric_mode, "passengers")
        
        if not stop_metric_data.empty and heatmap_metric in stop_metric_data.columns:
            heatmap_fg = add_heatmap_layer(
                m, stops, stop_metric_data, heatmap_metric,
                radius=18, blur=12
            )
            heatmap_fg.add_to(m)
        else:
            logger.warning(f"Heatmap metric '{heatmap_metric}' not available in stop_metrics")

    if show_bottlenecks and base_metric == "bottleneck":
        bottleneck_fg = add_bottleneck_layer(m, stops, stop_metrics)
        bottleneck_fg.add_to(m)

    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Add title
    display_mode = metric_mode.replace("heatmap_", "Heatmap: ") if is_heatmap else metric_mode.title()
    title_html = f'''
    <div style="position: fixed; top: 10px; left: 50px; width: 350px; height: 40px;
                background-color: white; border: 2px solid grey; border-radius: 5px;
                z-index: 9999; font-size: 14px; font-weight: bold; padding: 8px;">
        UrbanTransit IQ — Network Map ({display_mode})
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    return m


def save_map(map_obj: folium.Map, path: Path) -> None:
    """Save map to HTML file. Use :func:`render_map_html` for the dashboard."""
    map_obj.save(str(path))
    logger.info(f"Map saved to {path}")


def render_map_html(map_obj: folium.Map) -> str:
    """Return the map as an HTML string, without touching the filesystem.

    The dashboard used to call ``save_map(map_obj, "temp_map.html")`` and then read
    the file straight back. That wrote an ~800 KB file into the process's working
    directory on every "Generate Map" click (relative to the CWD, so it broke when
    Streamlit was started from elsewhere), never cleaned it up, and let two
    concurrent sessions overwrite each other's map mid-read.
    """
    return map_obj.get_root().render()


def get_map_metric_options() -> list[dict]:
    """Get available metric modes for the map."""
    return [
        {"value": "default", "label": "Default (Category Colors)"},
        {"value": "demand", "label": "Passenger Demand"},
        {"value": "delay", "label": "Average Delay"},
        {"value": "crowding", "label": "Crowding Share"},
        {"value": "occupancy", "label": "Max Occupancy"},
        {"value": "reliability", "label": "On-Time Reliability"},
        {"value": "bottleneck", "label": "Stop Bottlenecks"},
        {"value": "heatmap_demand", "label": "🔥 Heatmap: Passenger Demand"},
        {"value": "heatmap_crowding", "label": "🔥 Heatmap: Crowding"},
        {"value": "heatmap_delay", "label": "🔥 Heatmap: Delay"},
        {"value": "heatmap_occupancy", "label": "🔥 Heatmap: Occupancy"},
    ]


if __name__ == "__main__":
    # Test map generation
    m = create_network_map(metric_mode="demand")
    save_map(m, Path("test_map.html"))
    print("Test map saved to test_map.html")