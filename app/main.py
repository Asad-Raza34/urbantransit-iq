import sys
from pathlib import Path
import hashlib
import hmac
import time
import os
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

from src.services.data_access import (
    load_route_metrics,
    load_demand_analytics,
    load_crowding_analytics,
    load_alerts,
    load_stop_metrics,
    load_recommendations,
    calculate_kpis,
)


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="UrbanTransit IQ",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===========================================================================
# AUTHENTICATION
# ===========================================================================


@dataclass
class User:
    username: str
    role: str = "viewer"
    permissions: list[str] = None

    def __post_init__(self):
        if self.permissions is None:
            if self.role == "viewer":
                self.permissions = ["read"]
            else:
                self.permissions = ["read", "write", "admin"]


# ---------------------------------------------------------------------------
# Default users
# ---------------------------------------------------------------------------

DEFAULT_USERS = {
    "admin": {
        "password_hash": hashlib.sha256(
            "changeme123".encode()
        ).hexdigest(),
        "role": "admin",
    },
    "analyst": {
        "password_hash": hashlib.sha256(
            "analyst123".encode()
        ).hexdigest(),
        "role": "analyst",
    },
    "viewer": {
        "password_hash": hashlib.sha256(
            "viewer123".encode()
        ).hexdigest(),
        "role": "viewer",
    },
}


# ---------------------------------------------------------------------------
# Authentication configuration
# ---------------------------------------------------------------------------

def _get_secret_value(name: str, default=None):
    """Read configuration from Streamlit Secrets or environment variables."""

    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    value = os.getenv(name)

    if value is not None:
        return value

    return default


def _get_bool_secret(
    name: str,
    default: bool = False,
) -> bool:
    """Read a boolean configuration value safely."""

    value = _get_secret_value(
        name,
        default,
    )

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {
            "true",
            "1",
            "yes",
            "on",
        }

    return bool(value)


def _get_auth_config() -> dict:
    """Get authentication configuration."""

    return {
        "enabled": _get_bool_secret(
            "UTIQ_AUTH_ENABLED",
            False,
        ),
        "secret_key": _get_secret_value(
            "UTIQ_AUTH_SECRET_KEY",
            None,
        ),
        "cookie_name": _get_secret_value(
            "UTIQ_AUTH_COOKIE_NAME",
            "urbantransit_auth",
        ),
        "cookie_expiry_days": int(
            _get_secret_value(
                "UTIQ_AUTH_COOKIE_EXPIRY_DAYS",
                30,
            )
        ),
        "max_failed_attempts": int(
            _get_secret_value(
                "UTIQ_AUTH_MAX_FAILED_ATTEMPTS",
                5,
            )
        ),
        "lockout_seconds": int(
            _get_secret_value(
                "UTIQ_AUTH_LOCKOUT_SECONDS",
                300,
            )
        ),
    }


def init_auth() -> None:
    """Initialize authentication session state."""

    if "auth_user" not in st.session_state:
        st.session_state.auth_user = None

    if "auth_token" not in st.session_state:
        st.session_state.auth_token = None

    if "auth_attempts" not in st.session_state:
        st.session_state.auth_attempts = 0

    if "auth_locked_until" not in st.session_state:
        st.session_state.auth_locked_until = 0


def is_auth_enabled() -> bool:
    """Check whether authentication is enabled."""

    config = _get_auth_config()

    return config.get(
        "enabled",
        False,
    )


def _hash_password(password: str) -> str:
    """Hash a password using SHA-256."""

    return hashlib.sha256(
        password.encode()
    ).hexdigest()


def _verify_password(
    password: str,
    password_hash: str,
) -> bool:
    """Verify a password against its hash."""

    return hmac.compare_digest(
        _hash_password(password),
        password_hash,
    )


def _create_session_token(
    username: str,
    secret_key: str,
) -> str:
    """Create a signed session token."""

    timestamp = str(int(time.time()))

    message = f"{username}:{timestamp}".encode()

    signature = hmac.new(
        secret_key.encode(),
        message,
        hashlib.sha256,
    ).hexdigest()

    return f"{username}:{timestamp}:{signature}"


def _verify_session_token(
    token: str,
    secret_key: str,
    max_age_days: int = 30,
) -> Optional[str]:
    """Verify a signed session token."""

    try:
        parts = token.split(":")

        if len(parts) != 3:
            return None

        username, timestamp_str, signature = parts

        timestamp = int(timestamp_str)

        if time.time() - timestamp > max_age_days * 86400:
            return None

        message = f"{username}:{timestamp}".encode()

        expected_signature = hmac.new(
            secret_key.encode(),
            message,
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(
            signature,
            expected_signature,
        ):
            return None

        return username

    except Exception:
        return None


def get_current_user() -> Optional[User]:
    """Return the currently authenticated user."""

    init_auth()

    if not is_auth_enabled():
        return User(
            username="anonymous",
            role="admin",
        )

    user_data = st.session_state.get(
        "auth_user"
    )

    if user_data:
        return User(**user_data)

    return None


def _record_failed_attempt() -> None:
    """Record failed login attempts and apply lockout."""

    config = _get_auth_config()

    max_attempts = config.get(
        "max_failed_attempts",
        5,
    )

    lockout_seconds = config.get(
        "lockout_seconds",
        300,
    )

    st.session_state.auth_attempts = (
        st.session_state.get(
            "auth_attempts",
            0,
        )
        + 1
    )

    remaining = (
        max_attempts
        - st.session_state.auth_attempts
    )

    if remaining > 0:
        st.error(
            f"Invalid username or password. "
            f"{remaining} attempt(s) remaining."
        )

    if st.session_state.auth_attempts >= max_attempts:

        st.session_state.auth_locked_until = (
            time.time() + lockout_seconds
        )

        st.error(
            "Too many failed attempts. "
            f"Account locked for "
            f"{int(lockout_seconds // 60)} minute(s)."
        )


def authenticate(
    username: str,
    password: str,
) -> Optional[User]:
    """Authenticate a user."""

    config = _get_auth_config()

    secret_key = config.get(
        "secret_key"
    )

    if not secret_key:
        st.error(
            "Authentication secret key is not configured."
        )
        return None

    if time.time() < st.session_state.get(
        "auth_locked_until",
        0,
    ):
        remaining_seconds = int(
            st.session_state.auth_locked_until
            - time.time()
        )

        st.error(
            "Too many failed attempts. "
            f"Please wait {max(1, remaining_seconds)} "
            "second(s) before trying again."
        )

        return None

    username = username.strip().lower()

    if username not in DEFAULT_USERS:
        _record_failed_attempt()
        return None

    user_record = DEFAULT_USERS[username]

    if not _verify_password(
        password,
        user_record["password_hash"],
    ):
        _record_failed_attempt()
        return None

    token = _create_session_token(
        username,
        secret_key,
    )

    user = User(
        username=username,
        role=user_record["role"],
    )

    st.session_state.auth_user = {
        "username": user.username,
        "role": user.role,
        "permissions": user.permissions,
    }

    st.session_state.auth_token = token

    st.session_state.auth_attempts = 0
    st.session_state.auth_locked_until = 0

    return user


def logout() -> None:
    """Log out current user."""

    st.session_state.auth_user = None
    st.session_state.auth_token = None
    st.session_state.auth_attempts = 0
    st.session_state.auth_locked_until = 0

    st.session_state.pop(
        "authenticated",
        None,
    )

    st.rerun()


def require_auth() -> User:
    """Require authentication before showing the dashboard."""

    user = get_current_user()

    if user is None:
        show_login()
        st.stop()

    return user


def require_role(
    required_role: str,
) -> User:
    """Require a specific role."""

    user = require_auth()

    role_hierarchy = {
        "viewer": 0,
        "analyst": 1,
        "admin": 2,
    }

    user_level = role_hierarchy.get(
        user.role,
        0,
    )

    required_level = role_hierarchy.get(
        required_role,
        0,
    )

    if user_level < required_level:
        st.error(
            f"Access denied. Required role: {required_role}"
        )
        st.stop()

    return user


def show_login() -> None:
    """Display the UrbanTransit IQ login page."""

    st.markdown(
        """
        <style>

        [data-testid="stSidebar"] {
            display: none;
        }

        .login-wrapper {
            max-width: 460px;
            margin: 7rem auto 0 auto;
        }

        .login-card {
            padding: 2.5rem;
            border-radius: 18px;
            border: 1px solid rgba(128, 128, 128, 0.25);
            background: rgba(255, 255, 255, 0.04);
            box-shadow: 0 10px 35px rgba(0, 0, 0, 0.15);
        }

        .login-title {
            text-align: center;
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }

        .login-subtitle {
            text-align: center;
            color: #888;
            margin-bottom: 2rem;
        }

        .login-brand {
            text-align: center;
            font-size: 3.5rem;
            margin-bottom: 0.5rem;
        }

        .login-footer {
            text-align: center;
            color: #888;
            font-size: 0.85rem;
            margin-top: 1.5rem;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="login-wrapper">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="login-card">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="login-brand">🚌</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="login-title">UrbanTransit IQ</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="login-subtitle">'
        'Smart Public Transport Analytics & Decision Support'
        '</div>',
        unsafe_allow_html=True,
    )

    if not is_auth_enabled():

        st.warning(
            "Authentication is currently disabled."
        )

        st.info(
            "Enable authentication using "
            "`UTIQ_AUTH_ENABLED = true` "
            "in Streamlit Cloud Secrets."
        )

        st.markdown(
            '</div></div>',
            unsafe_allow_html=True,
        )

        return

    config = _get_auth_config()

    secret_key = config.get(
        "secret_key"
    )

    if not secret_key:

        st.error(
            "Authentication is enabled but "
            "`UTIQ_AUTH_SECRET_KEY` is not configured."
        )

        st.info(
            "Open Streamlit Cloud → Settings → Secrets "
            "and add a secret key of at least 32 characters."
        )

        st.markdown(
            '</div></div>',
            unsafe_allow_html=True,
        )

        return

    with st.form("login_form"):

        username = st.text_input(
            "Username",
            placeholder="Enter your username",
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
        )

        submit = st.form_submit_button(
            "Login",
            type="primary",
            use_container_width=True,
        )

        if submit:

            if not username or not password:

                st.error(
                    "Please enter both username and password."
                )

            else:

                user = authenticate(
                    username,
                    password,
                )

                if user:

                    st.success(
                        f"Welcome, {user.username}!"
                    )

                    st.rerun()

    st.markdown(
        '<div class="login-footer">'
        'UrbanTransit IQ v1.0.0'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '</div></div>',
        unsafe_allow_html=True,
    )


def show_user_menu() -> None:
    """Show authenticated user menu in sidebar."""

    user = get_current_user()

    if not user:
        return

    with st.sidebar:

        st.markdown("---")

        st.markdown(
            f"**Logged in as:** "
            f"{user.username} ({user.role})"
        )

        if st.button(
            "Logout",
            use_container_width=True,
        ):
            logout()


# ===========================================================================
# INITIALIZE AUTHENTICATION
# ===========================================================================

init_auth()


# ===========================================================================
# REQUIRE LOGIN BEFORE DASHBOARD
# ===========================================================================

if is_auth_enabled():
    _current_user = require_auth()
else:
    _current_user = get_current_user()


# ===========================================================================
# SESSION FILTER STATE
# ===========================================================================

if "filters" not in st.session_state:

    st.session_state.filters = {
        "route_ids": [],
        "date_range": None,
        "severity": [],
        "priority": [],
    }


# ===========================================================================
# CACHE DATA LOADING FUNCTIONS
# ===========================================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def _cached_route_metrics():
    return load_route_metrics()


@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def _cached_demand_analytics():
    return load_demand_analytics()


@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def _cached_crowding_analytics():
    return load_crowding_analytics()


@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def _cached_alerts():
    return load_alerts()


@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def _cached_stop_metrics():
    return load_stop_metrics()


@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def _cached_recommendations():
    return load_recommendations()


@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def _cached_kpis():
    return calculate_kpis()


# ===========================================================================
# ROLE-BASED PAGE ACCESS
# ===========================================================================

ROLE_LEVEL = {
    "viewer": 0,
    "analyst": 1,
    "admin": 2,
}


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


_user_level = ROLE_LEVEL.get(
    _current_user.role,
    0,
)


_visible_pages = [
    p
    for p in ALL_PAGES
    if _user_level
    >= ROLE_LEVEL.get(
        PAGE_MIN_ROLE.get(
            p,
            "viewer",
        ),
        0,
    )
]


_restricted_pages = [
    p
    for p in ALL_PAGES
    if p not in _visible_pages
]


if not _visible_pages:

    st.error(
        "Your role has no accessible pages. "
        "Contact an administrator."
    )

    st.stop()


# ===========================================================================
# CUSTOM CSS
# ===========================================================================

st.markdown(
    """
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

    .alert-critical {
        border-left-color: #dc3545;
    }

    .alert-high {
        border-left-color: #fd7e14;
    }

    .alert-medium {
        border-left-color: #ffc107;
    }

    .alert-low {
        border-left-color: #28a745;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# SIDEBAR NAVIGATION
# ===========================================================================

with st.sidebar:

    st.markdown(
        '<div class="main-header">'
        '🚌 UrbanTransit IQ'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    page = st.selectbox(
        "Navigate",
        _visible_pages,
        index=0,
    )

    if _restricted_pages:

        st.caption(
            f"Role **{_current_user.role}** — "
            f"{len(_restricted_pages)} page(s) hidden: "
            + ", ".join(
                p.split(" ", 1)[1]
                for p in _restricted_pages
            )
        )

    st.markdown("---")

    st.markdown("### Filters")

    routes_df = _cached_route_metrics()

    all_route_ids = sorted(
        routes_df["route_id"]
        .unique()
        .tolist()
    )

    selected_routes = st.multiselect(
        "Routes",
        all_route_ids,
        default=st.session_state.filters[
            "route_ids"
        ],
        key="global_route_filter",
    )

    st.session_state.filters[
        "route_ids"
    ] = selected_routes

    daily = _cached_demand_analytics().get(
        "daily_demand",
        pd.DataFrame(),
    )

    if not daily.empty:

        min_date = (
            daily["date"]
            .min()
            .date()
        )

        max_date = (
            daily["date"]
            .max()
            .date()
        )

        date_range = st.date_input(
            "Date Range",
            value=(
                min_date,
                max_date,
            ),
            min_value=min_date,
            max_value=max_date,
            key="global_date_filter",
        )

        if len(date_range) == 2:

            st.session_state.filters[
                "date_range"
            ] = date_range

    severity_options = [
        "critical",
        "high",
        "medium",
        "low",
    ]

    selected_severity = st.multiselect(
        "Alert Severity",
        severity_options,
        default=st.session_state.filters[
            "severity"
        ],
        key="global_severity_filter",
    )

    st.session_state.filters[
        "severity"
    ] = selected_severity

    priority_options = [
        "CRITICAL",
        "HIGH",
        "MEDIUM",
        "LOW",
    ]

    selected_priority = st.multiselect(
        "Recommendation Priority",
        priority_options,
        default=st.session_state.filters[
            "priority"
        ],
        key="global_priority_filter",
    )

    st.session_state.filters[
        "priority"
    ] = selected_priority

    st.markdown("---")

    show_user_menu()

    st.caption(
        "UrbanTransit IQ v1.0.0"
    )

    st.caption(
        "Smart Public Transport Analytics & Decision Support"
    )


# ===========================================================================
# MAIN CONTENT ROUTING
# ===========================================================================

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