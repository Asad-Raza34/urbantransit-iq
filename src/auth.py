"""Lightweight authentication for UrbanTransit IQ Streamlit dashboard.

Provides simple session-based authentication with optional enablement.
"""

import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass
from typing import Optional

import streamlit as st

from config import settings
from src.log import get_logger

logger = get_logger(__name__)


@dataclass
class User:
    """Authenticated user."""
    username: str
    role: str = "viewer"
    permissions: list[str] = None

    def __post_init__(self):
        if self.permissions is None:
            self.permissions = ["read"] if self.role == "viewer" else ["read", "write", "admin"]


# Default users (in production, use proper user store)
DEFAULT_USERS = {
    "admin": {
        "password_hash": hashlib.sha256("changeme123".encode()).hexdigest(),
        "role": "admin",
    },
    "analyst": {
        "password_hash": hashlib.sha256("analyst123".encode()).hexdigest(),
        "role": "analyst",
    },
    "viewer": {
        "password_hash": hashlib.sha256("viewer123".encode()).hexdigest(),
        "role": "viewer",
    },
}


def _get_auth_config() -> dict:
    """Get authentication configuration from settings."""
    return {
        "enabled": getattr(settings, "AUTH_ENABLED", False),
        "secret_key": getattr(settings, "AUTH_SECRET_KEY", None),
        "cookie_name": getattr(settings, "AUTH_COOKIE_NAME", "urbantransit_auth"),
        "cookie_expiry_days": getattr(settings, "AUTH_COOKIE_EXPIRY_DAYS", 30),
    }


def _hash_password(password: str) -> str:
    """Hash a password with SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def _verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    return hmac.compare_digest(_hash_password(password), password_hash)


def _create_session_token(username: str, secret_key: str) -> str:
    """Create a secure session token."""
    timestamp = str(int(time.time()))
    message = f"{username}:{timestamp}".encode()
    signature = hmac.new(secret_key.encode(), message, hashlib.sha256).hexdigest()
    return f"{username}:{timestamp}:{signature}"


def _verify_session_token(token: str, secret_key: str, max_age_days: int = 30) -> Optional[str]:
    """Verify a session token and return username if valid."""
    try:
        parts = token.split(":")
        if len(parts) != 3:
            return None
        username, timestamp_str, signature = parts
        timestamp = int(timestamp_str)
        
        # Check expiry
        if time.time() - timestamp > max_age_days * 86400:
            return None
        
        # Verify signature
        message = f"{username}:{timestamp}".encode()
        expected_signature = hmac.new(secret_key.encode(), message, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected_signature):
            return None
        
        return username
    except Exception:
        return None


def init_auth() -> None:
    """Initialize authentication state."""
    if "auth_user" not in st.session_state:
        st.session_state.auth_user = None
    if "auth_attempts" not in st.session_state:
        st.session_state.auth_attempts = 0
    if "auth_locked_until" not in st.session_state:
        st.session_state.auth_locked_until = 0


def is_auth_enabled() -> bool:
    """Check if authentication is enabled."""
    config = _get_auth_config()
    return config.get("enabled", False)


def get_current_user() -> Optional[User]:
    """Get the currently authenticated user."""
    init_auth()

    if not is_auth_enabled():
        # Return default user when auth disabled
        return User(username="anonymous", role="admin")

    
    user_data = st.session_state.get("auth_user")
    if user_data:
        return User(**user_data)
    return None


def require_auth() -> User:
    """Require authentication, show login if not authenticated."""
    user = get_current_user()
    if user is None:
        show_login()
        st.stop()
    return user


def require_role(required_role: str) -> User:
    """Require a specific role."""
    user = require_auth()
    role_hierarchy = {"viewer": 0, "analyst": 1, "admin": 2}
    user_level = role_hierarchy.get(user.role, 0)
    required_level = role_hierarchy.get(required_role, 0)
    
    if user_level < required_level:
        st.error(f"Access denied. Required role: {required_role}")
        st.stop()
    return user


def authenticate(username: str, password: str) -> Optional[User]:
    """Authenticate a user with username and password."""
    config = _get_auth_config()
    secret_key = config.get("secret_key")
    
    if not secret_key:
        logger.error("Authentication secret key not configured")
        return None
    
    # Check lockout
    if time.time() < st.session_state.get("auth_locked_until", 0):
        st.error("Too many failed attempts. Please wait before trying again.")
        return None
    
    # In production, fetch from secure user store
    users = DEFAULT_USERS
    
    if username not in users:
        _record_failed_attempt()
        return None
    
    user_record = users[username]
    if not _verify_password(password, user_record["password_hash"]):
        _record_failed_attempt()
        return None
    
    # Success - create session
    token = _create_session_token(username, secret_key)
    user = User(username=username, role=user_record["role"])
    
    st.session_state.auth_user = {
        "username": user.username,
        "role": user.role,
        "permissions": user.permissions,
    }
    st.session_state.auth_token = token
    st.session_state.auth_attempts = 0
    
    logger.info(f"User {username} authenticated successfully")
    return user


def _record_failed_attempt() -> None:
    """Record a failed authentication attempt and apply the lockout policy."""
    max_attempts = getattr(settings, "AUTH_MAX_FAILED_ATTEMPTS", 5)
    lockout_seconds = getattr(settings, "AUTH_LOCKOUT_SECONDS", 300)
    st.session_state.auth_attempts = st.session_state.get("auth_attempts", 0) + 1
    remaining = max_attempts - st.session_state.auth_attempts
    if remaining > 0:
        st.error(f"Invalid username or password. {remaining} attempt(s) remaining.")
    if st.session_state.auth_attempts >= max_attempts:
        st.session_state.auth_locked_until = time.time() + lockout_seconds
        logger.warning("Auth lockout engaged for %s seconds", lockout_seconds)
        st.error(
            f"Too many failed attempts. Account locked for "
            f"{int(lockout_seconds // 60)} minute(s)."
        )


def logout() -> None:
    """Log out the current user."""
    st.session_state.auth_user = None
    st.session_state.auth_token = None
    st.session_state.auth_attempts = 0
    st.session_state.auth_locked_until = 0
    logger.info("User logged out")


def show_login() -> None:
    """Display the login form."""
    st.markdown("""
    <style>
    .login-container {
        max-width: 400px;
        margin: 2rem auto;
        padding: 2rem;
        border: 1px solid #ddd;
        border-radius: 8px;
        background: #fafafa;
    }
    .login-title {
        text-align: center;
        color: #1f4e79;
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="login-title">🚌 UrbanTransit IQ - Login</h2>', unsafe_allow_html=True)
    
    if not is_auth_enabled():
        st.info("Authentication is disabled. Running in open mode.")
        if st.button("Continue without authentication"):
            st.session_state.auth_user = {"username": "anonymous", "role": "admin", "permissions": ["read", "write", "admin"]}
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    if not _get_auth_config().get("secret_key"):
        # Without a secret key every login attempt fails silently, which reads as
        # a broken app. Fail loudly with the exact variable to set instead.
        st.error(
            "Authentication is enabled but `UTIQ_AUTH_SECRET_KEY` is not set, so no "
            "session can be signed and login cannot succeed. Set a secret of 32+ "
            "characters (in `.env` or the environment) and restart the app."
        )
        st.markdown('</div>', unsafe_allow_html=True)
        return

    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        submit = st.form_submit_button("Login", type="primary", use_container_width=True)
        
        if submit:
            if username and password:
                user = authenticate(username, password)
                if user:
                    st.success(f"Welcome, {user.username}!")
                    st.rerun()
            else:
                st.error("Please enter both username and password")
    
    st.markdown("---")
    st.caption("Default users: admin/changeme123, analyst/analyst123, viewer/viewer123")
    st.markdown('</div>', unsafe_allow_html=True)


def show_user_menu() -> None:
    """Show user menu in sidebar."""
    user = get_current_user()
    if not user:
        return
    
    with st.sidebar:
        st.markdown("---")
        st.markdown(f"**Logged in as:** {user.username} ({user.role})")
        
        if st.button("Logout", use_container_width=True):
            logout()
            st.rerun()


if __name__ == "__main__":
    # Test authentication functions
    print("Testing authentication...")
    print(f"Auth enabled: {is_auth_enabled()}")
    print(f"Default users: {list(DEFAULT_USERS.keys())}")