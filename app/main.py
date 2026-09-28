"""UrbanTransit IQ - Main Streamlit Application."""

import sys
from pathlib import Path

# Add project root to Python path so 'src' imports work when running from app/
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from src.services.data_access import (
    load_route_metrics, load_demand_analytics, load_crowding_analytics,
    load_alerts, load_stop_metrics, load_recommendations, calculate_kpis
)
from src.auth import init_auth, get_current_user, require_auth, show_user_menu, is_auth_enabled

# Page configuration
st.set_page_config(
    page_title="UrbanTransit IQ",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize authentication
init_auth()

# Initialize session state for filters
if "filters" not in st.session_state:
    st.session_state.filters = {
        "route_ids": [],
        "date_range": None,
        "severity": [],
        "priority": [],
    }

# Cache data loading functions for performance
@st.cache_data(ttl=3600, show_spinner=False)
def _cached_route_metrics():
    return load_route_metrics()

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_demand_analytics():
    return load_demand_analytics()

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_crowding_analytics():
    return load_crowding_analytics()

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_alerts():
    return load_alerts()

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_stop_metrics():
    return load_stop_metrics()

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_recommendations():
    return load_recommendations()

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_kpis():
    return calculate_kpis()

# Require authentication if enabled
if is_auth_enabled():
    require_auth()


# ---------------------------------------------------------------------------
# Role-based page access (SRS 44).
#
# ``require_role`` existed but was never called, so every role could open every
# page. Pages not listed default to "viewer"; the mapping below is the single
# place that defines the access boundary.
# ---------------------------------------------------------------------------
ROLE_LEVEL = {"viewer": 0, "analyst": 1, "admin": 2}

PAGE_MIN_ROLE = {
    "⚡ Spark Monitoring": "admin",
    "🏥 Health Monitoring": "admin",
    "📋 Audit Timeline": "admin",
    "🔮 What-If Simulator": "analyst",
    "📄 Reports & Export": "analyst",
}

ALL_PAGES = [
    "📊 Executive Overview",
    "👥 Passenger Flow",
    "🛣️ Route Analytics",
    "⏱️ Delay & Reliability",
    "📈 Crowding & Capacity",
    "🎯 Recommendations",
    "🚨 Smart Alerts",
    "🔮 What-If Simulator",
    "🗺️ Network Map",
    "🔬 Route Clustering",
    "📈 Occupancy Forecasting",
    "🕐 Frequency Analysis",
    "🎯 Delay Severity Classification",
    "⚡ Spark Monitoring",
    "🏥 Health Monitoring",
    "📋 Audit Timeline",
    "📄 Reports & Export",
]

_current_user = get_current_user()
_user_level = ROLE_LEVEL.get(_current_user.role, 0)
_visible_pages = [
    p for p in ALL_PAGES
    if _user_level >= ROLE_LEVEL.get(PAGE_MIN_ROLE.get(p, "viewer"), 0)
]
_restricted_pages = [p for p in ALL_PAGES if p not in _visible_pages]
if not _visible_pages:
    st.error("Your role has no accessible pages. Contact an administrator.")
    st.stop()

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #ffffff;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: #000000;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-critical { border-left-color: #dc3545; }
    .alert-high { border-left-color: #fd7e14; }
    .alert-medium { border-left-color: #ffc107; }
    .alert-low { border-left-color: #28a745; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.markdown('<div class="main-header">🚌 UrbanTransit IQ</div>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.selectbox(
        "Navigate",
        _visible_pages,
        index=0,
    )
    if _restricted_pages:
        st.caption(
            f"Role **{_current_user.role}** — {len(_restricted_pages)} page(s) hidden: "
            + ", ".join(p.split(" ", 1)[1] for p in _restricted_pages)
        )

    st.markdown("---")

    # Global filters
    st.markdown("### Filters")
    routes_df = _cached_route_metrics()
    all_route_ids = sorted(routes_df["route_id"].unique().tolist())

    selected_routes = st.multiselect(
        "Routes",
        all_route_ids,
        default=st.session_state.filters["route_ids"],
        key="global_route_filter",
    )
    st.session_state.filters["route_ids"] = selected_routes

    # Date range
    import pandas as pd
    daily = _cached_demand_analytics().get("daily_demand", pd.DataFrame())
    if not daily.empty:
        min_date = daily["date"].min().date()
        max_date = daily["date"].max().date()
        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="global_date_filter",
        )
        if len(date_range) == 2:
            st.session_state.filters["date_range"] = date_range

    # Severity filter
    severity_options = ["critical", "high", "medium", "low"]
    selected_severity = st.multiselect(
        "Alert Severity",
        severity_options,
        default=st.session_state.filters["severity"],
        key="global_severity_filter",
    )
    st.session_state.filters["severity"] = selected_severity

    # Priority filter
    priority_options = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    selected_priority = st.multiselect(
        "Recommendation Priority",
        priority_options,
        default=st.session_state.filters["priority"],
        key="global_priority_filter",
    )
    st.session_state.filters["priority"] = selected_priority

    st.markdown("---")
    
    # Show user menu if authenticated
    show_user_menu()
    
    st.caption("UrbanTransit IQ v1.0.0")
    st.caption("Smart Public Transport Analytics & Decision Support")

# Main content routing
if page == "📊 Executive Overview":
    from app.pages.executive import render
    render()
elif page == "👥 Passenger Flow":
    from app.pages.passenger_flow import render
    render()
elif page == "🛣️ Route Analytics":
    from app.pages.route_analytics import render
    render()
elif page == "⏱️ Delay & Reliability":
    from app.pages.delay_reliability import render
    render()
elif page == "📈 Crowding & Capacity":
    from app.pages.crowding_capacity import render
    render()
elif page == "🎯 Recommendations":
    from app.pages.recommendations import render
    render()
elif page == "🚨 Smart Alerts":
    from app.pages.smart_alerts import render
    render()
elif page == "🔮 What-If Simulator":
    from app.pages.whatif import render
    render()
elif page == "🗺️ Network Map":
    from app.pages.map import render
    render()
elif page == "🔬 Route Clustering":
    from app.pages.clustering import render
    render()
elif page == "📈 Occupancy Forecasting":
    from app.pages.forecasting import render
    render()
elif page == "🕐 Frequency Analysis":
    from app.pages.frequency import render
    render()
elif page == "🎯 Delay Severity Classification":
    from app.pages.delay_classification import render
    render()
elif page == "⚡ Spark Monitoring":
    from app.pages.spark_monitoring import render
    render()
elif page == "🏥 Health Monitoring":
    from app.pages.health_monitoring import render
    render()
elif page == "📋 Audit Timeline":
    from app.pages.audit_timeline import render
    render()
elif page == "📄 Reports & Export":
    from app.pages.reports_export import render
    render()