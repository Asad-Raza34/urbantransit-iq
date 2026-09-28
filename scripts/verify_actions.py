"""Prove every action-gated button in the dashboard actually produces a result.

``verify_features.py`` proves each page renders and that every *widget* can be
driven to another value. It cannot cover buttons that gate output: on those
pages nothing interesting renders until the button is pressed.

This harness drives the app the way a real user does -- it loads ``app/main.py``
and uses the sidebar **Navigate** selectbox. Loading a page file directly is not
equivalent: ``main.py`` initialises ``st.session_state.filters`` and the auth
session, and real users never bypass it.

For each button it presses, it reports how the chart/table/KPI/download counts
changed, plus every status box the app raised. A red ``st.error`` box is a
user-visible failure even when no exception was raised, so it is treated as a
failure, not an "OK".
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from verify_features import load_page  # noqa: E402

# Every action button in the app, in sidebar order. ``select_all`` fills every
# multiselect on the page first, because some buttons act on the selection rather
# than on fixed input.
TARGETS: list[dict] = [
    {"page": "🛣️ Route Analytics", "buttons": ["Load Profile"]},
    {"page": "⏱️ Delay & Reliability", "buttons": ["🔮 Generate Prediction"]},
    {"page": "⏱️ Delay & Reliability", "buttons": ["📊 Evaluate on Unseen Cases"]},
    {"page": "🎯 Recommendations", "buttons": ["🔄 Regenerate Recommendations"]},
    {
        "page": "🚨 Smart Alerts",
        "buttons": [
            "🔄 Regenerate Alerts",
            "✅ Acknowledge Selected",
            "✅ Resolve Selected",
            "📋 Export Selected",
        ],
        "preselect": ["Select Alerts for Bulk Action"],
    },
    {"page": "🔮 What-If Simulator", "buttons": ["▶️ Run Selected Scenarios"]},
    {"page": "🔮 What-If Simulator", "buttons": ["▶️ Run Custom Scenario"]},
    {"page": "🗺️ Network Map", "buttons": ["🗺️ Generate Map"]},
    {"page": "🔬 Route Clustering", "buttons": ["🔄 Rebuild Clusters"]},
    {
        "page": "📈 Occupancy Forecasting",
        "buttons": ["🔮 Generate Forecast", "📊 Evaluate on Unseen Cases"],
    },
    {"page": "⚡ Spark Monitoring", "buttons": ["Show Job Details"]},
    {"page": "🏥 Health Monitoring", "buttons": ["📸 Record Sample"]},
    {"page": "📋 Audit Timeline", "buttons": ["🔄 Refresh"]},
    {
        "page": "📄 Reports & Export",
        "buttons": [
            "📄 Generate PDF Report",
            "📦 Build Export",
            "🗜️ Build Full Report Package",
        ],
    },
]


def _n(at, kind: str) -> int:
    """Element count for ``kind``, tolerating names AppTest does not know."""
    try:
        return len(at.get(kind))
    except Exception:
        return 0


def counts(at) -> dict:
    """Visible output counts, including non-Plotly renders.

    The Network Map draws with folium through ``st.components.v1.html``, so a
    Plotly-only count reported 0 charts even when the map rendered fine.
    """
    return {
        "charts": _n(at, "plotly_chart"),
        "tables": _n(at, "dataframe") + _n(at, "table") + _n(at, "arrow_data_frame"),
        "metrics": _n(at, "metric"),
        "downloads": _n(at, "download_button"),
        "html": _n(at, "html") + _n(at, "component_instance") + _n(at, "iframe"),
        "json": _n(at, "json"),
    }


def _safe_list(at, kind: str) -> list:
    try:
        return list(at.get(kind))
    except Exception:
        return []


def messages(at) -> list[str]:
    """Streamlit status boxes.

    A red ``st.error`` box is a user-visible failure but is *not* an exception,
    so checking only ``at.exception`` would have called a failed action "OK".
    """
    out: list[str] = []
    for kind in ("error", "warning", "success", "info"):
        for el in _safe_list(at, kind):
            body = str(getattr(el, "value", "") or getattr(el, "body", "") or "")
            first = body.strip().splitlines()[0] if body.strip() else ""
            out.append(f"{kind}: {first[:200]}")
    return out


def all_buttons(at) -> list:
    """Every button in the tree, including those nested inside tabs.

    ``AppTest.button`` is a flat, top-level-only view, so a button rendered
    inside ``st.tabs`` is invisible to it. ``at.get()`` walks the whole tree.
    """
    return list(at.get("button"))


def button_labels(at) -> list[str]:
    return [(getattr(b, "label", "") or "").strip() for b in all_buttons(at)]


def find_button(at, label: str):
    """Find a button by exact label first, then by substring.

    Exact-first matters: "✅ Acknowledge" is a substring of "✅ Acknowledge
    Selected", so a substring-only search would press the bulk button and
    report it as the per-alert one.
    """
    needle = label.lower()
    buttons = all_buttons(at)
    for btn in buttons:
        if (getattr(btn, "label", "") or "").strip().lower() == needle:
            return btn
    for btn in buttons:
        if needle in (getattr(btn, "label", "") or "").lower():
            return btn
    return None


def fmt(d: dict) -> str:
    return "  ".join(f"{k}={v}" for k, v in d.items() if v)


def error_of(at) -> str | None:
    if at.exception:
        node = at.exception[0]
        msg = getattr(node, "message", "") or ""
        first = msg.strip().splitlines()[0] if msg.strip() else ""
        return f"{node.type or 'Exception'}: {first}"
    return None


def has_error_box(at) -> bool:
    return any(m.startswith("error:") for m in messages(at))


def preselect(at, labels: list[str]) -> None:
    """Tick the first few options of named multiselects, then settle the rerun.

    Uses ``set_value``: ``Multiselect.select()`` stores the argument itself, so
    passing a list makes ``format_func`` receive a list and the page fails with
    an error a real user cannot produce. ``verify_features`` uses ``set_value``
    for the same reason. Only the bulk-action box is ticked -- ticking every
    filter too would change the dataset under test at the same time.
    """
    changed = False
    for label in labels:
        idx = -1
        for i, ms in enumerate(_safe_list(at, "multiselect")):
            if ms.label == label:
                idx = i
                break
        if idx < 0:
            print(f"      preselect -> widget {label!r} not found")
            continue
        ms = at.multiselect[idx]
        try:
            picks = list(ms.options)[:3]
        except (AttributeError, TypeError):
            picks = []
        if not picks:
            print(f"      preselect -> {label!r} has no options to tick")
            continue
        ms.set_value(picks)
        changed = True
        print(f"      preselect -> {label!r} set to {len(picks)} option(s)")
    if changed:
        at.run(timeout=900)


def drive(page: str, labels: list[str], preselect_labels: list[str] | None = None) -> None:
    print(f"\n{'=' * 72}")
    print(f"PAGE  {page}")
    print("=" * 72)
    try:
        at = load_page(page, timeout=900)
    except Exception:
        print("  LOAD ERROR:")
        print("   ", traceback.format_exc().splitlines()[-1])
        return

    err = error_of(at)
    print(f"  navigate         : {'OK' if err is None else 'FAILED -> ' + err}")
    if err:
        return

    if preselect_labels:
        try:
            preselect(at, preselect_labels)
            err = error_of(at)
            if err:
                print(f"  preselect        : FAILED -> {err}")
                return
            print("  preselect        : OK")
        except Exception:
            print("  preselect        : ERROR")
            print("   ", traceback.format_exc().splitlines()[-1])

    print(f"  before           : {fmt(counts(at))}")
    print(f"  buttons available: {button_labels(at)}")
    for m in messages(at):
        print(f"      on-load message -> {m}")

    for label in labels:
        # Re-tick the selection before each action: a bulk action consumes the
        # selection it acted on, so a real user re-selects before the next one.
        if preselect_labels:
            try:
                preselect(at, preselect_labels)
                err = error_of(at)
                if err:
                    print(f"  [{label!r}] PRESELECT RENDER ERROR -> {err}")
                    continue
            except Exception:
                print(f"  [{label!r}] PRESELECT ERROR")
                print("   ", traceback.format_exc().splitlines()[-1])
                continue

        target = find_button(at, label)
        if target is None:
            print(f"  [{label!r}] NOT FOUND on this page")
            continue
        before = counts(at)
        try:
            target.click().run(timeout=900)
        except Exception:
            print(f"  [{label!r}] CLICK ERROR:")
            print("   ", traceback.format_exc().splitlines()[-1])
            continue

        err = error_of(at)
        if err:
            print(f"  [{label!r}] RENDER ERROR -> {err}")
            continue

        after = counts(at)
        delta = {k: after[k] - before[k] for k in after}
        status = "SHOWED ERROR BOX" if has_error_box(at) else "OK"
        print(f"  [{label!r}] {status}  after: {fmt(after)}  delta: {fmt(delta)}")
        for m in messages(at):
            print(f"      message -> {m}")

        # A failed action often leaves the page unchanged; make that explicit.
        if status == "OK" and not any(v for v in delta.values()):
            print("      note -> no visible change in charts/tables/metrics/downloads")


def main() -> int:
    for target in TARGETS:
        drive(target["page"], target["buttons"], target.get("preselect"))
    print("\nDONE")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
