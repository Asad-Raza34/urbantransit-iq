"""Authentication / authorisation verification (SRS section 44).

Runs the real app with authentication enabled and exercises:

* the login form appearing instead of the dashboard,
* wrong credentials and the failed-attempt counter,
* account lockout after the configured number of failures,
* successful login for admin / analyst / viewer,
* role-based page visibility,
* logout returning to the login screen.

Requires ``UTIQ_AUTH_ENABLED=true`` and ``UTIQ_AUTH_SECRET_KEY`` to be set in the
environment (the script checks this and exits with a clear message otherwise).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(PROJECT_ROOT / "app" / "main.py")
CREDENTIALS = {
    "admin": "changeme123",
    "analyst": "analyst123",
    "viewer": "viewer123",
}
results: list[tuple[bool, str]] = []


def check(ok: bool, label: str) -> None:
    results.append((ok, label))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def fresh_app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    return at


def login(at: AppTest, username: str, password: str) -> AppTest:
    at.text_input[0].set_value(username)
    at.text_input[1].set_value(password)
    at.button[0].click().run()
    return at


def page_labels(at: AppTest) -> list[str]:
    for sb in at.selectbox:
        if sb.label == "Navigate":
            return list(sb.options)
    return []


def main() -> int:
    from src.config import settings

    print("Auth configuration:")
    print(f"  AUTH_ENABLED = {settings.AUTH_ENABLED}")
    secret = getattr(settings, "AUTH_SECRET_KEY", "")
    print(f"  secret length = {len(secret)}")
    if not settings.AUTH_ENABLED or len(secret) < 16:
        print("\nERROR: run with UTIQ_AUTH_ENABLED=true and a 32+ char "
              "UTIQ_AUTH_SECRET_KEY to verify authentication.")
        return 2
    print(f"  max attempts = {getattr(settings, 'AUTH_MAX_FAILED_ATTEMPTS', 5)}")
    print(f"  lockout secs = {getattr(settings, 'AUTH_LOCKOUT_SECONDS', 300)}\n")

    print("1. Dashboard is protected when auth is enabled")
    at = fresh_app()
    check(at.text_input and at.text_input[0].label == "Username",
          "login form is shown instead of the dashboard")
    check(not page_labels(at), "sidebar navigation is hidden before login")
    check(not at.exception, "no exception on the login screen")

    print("\n2. Wrong password is rejected and counted")
    at = fresh_app()
    login(at, "admin", "definitely-wrong")
    errs = " ".join(str(e.value) for e in at.error)
    check("Invalid username or password" in errs, f"error shown: {errs[:80]!r}")
    check("remaining" in errs.lower(), "remaining-attempt counter is reported")

    print("\n3. Unknown user is rejected")
    at = fresh_app()
    login(at, "nobody", "whatever")
    check(any("Invalid username or password" in str(e.value) for e in at.error),
          "unknown username rejected")
    check(not page_labels(at), "still not logged in after unknown user")

    print("\n4. Account lockout after repeated failures")
    at = fresh_app()
    max_attempts = int(getattr(settings, "AUTH_MAX_FAILED_ATTEMPTS", 5))
    for _ in range(max_attempts):
        login(at, "admin", "wrong-password")
    errs = " ".join(str(e.value) for e in at.error)
    check("locked" in errs.lower(), f"lockout message shown: {errs[:100]!r}")
    login(at, "admin", CREDENTIALS["admin"])
    errs_after = " ".join(str(e.value) for e in at.error)
    check("Too many failed attempts" in errs_after,
          "correct password is refused while locked out")
    check(not page_labels(at), "dashboard stays blocked during lockout")

    print("\n5. Each role logs in and sees the expected pages")
    expected_hidden = {
        "admin": [],
        "analyst": ["Health Monitoring", "Spark Monitoring", "Audit Timeline"],
        "viewer": ["Health Monitoring", "Spark Monitoring", "Audit Timeline",
                   "What-If Simulator", "Reports & Export"],
    }
    for role, password in CREDENTIALS.items():
        at = fresh_app()
        login(at, role, password)
        check(not at.exception, f"{role}: login produced no exception")
        labels = page_labels(at)
        check(len(labels) > 0, f"{role}: navigation available after login "
                               f"({len(labels)} pages)")
        for hidden in expected_hidden[role]:
            visible = any(hidden in label for label in labels)
            check(not visible, f"{role}: '{hidden}' is hidden")
        if role == "admin":
            check(len(labels) == 16, f"admin sees all 16 pages (saw {len(labels)})")
        if role == "viewer":
            check(any("Executive" in label for label in labels),
                  "viewer can still read the Executive Overview")

    print("\n6. Logout returns to the login screen")
    at = fresh_app()
    login(at, "admin", CREDENTIALS["admin"])
    page_labels(at)
    logout_buttons = [b for b in at.button if b.label == "Logout"]
    if logout_buttons:
        logout_buttons[0].click().run()
        check(at.text_input and at.text_input[0].label == "Username",
              "logout shows the login form again")
        # AppTest keeps the previous execution's elements in its tree, so run the
        # script once more to observe the settled post-logout state.
        at.run()
        check(not page_labels(at), "navigation is gone after logout")
        check(at.text_input and at.text_input[0].label == "Username",
              "login form persists after logout settles")
    else:
        check(False, "Logout button was found")

    print("\n" + "=" * 70)
    failed = [label for ok, label in results if not ok]
    print(f"AUTH CHECKS: {len(results) - len(failed)}/{len(results)} passed")
    for label in failed:
        print("  FAILED:", label)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
