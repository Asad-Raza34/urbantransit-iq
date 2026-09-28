"""Feature-level verification harness for the UrbanTransit IQ dashboard.

Complements ``verify_dashboard.py`` (which only proves a page *renders*). This
harness proves more:

* every Plotly chart actually contains data (trace count + total plotted points),
* KPI cards are populated and contain no ``nan``/``inf``,
* switching every dropdown / multiselect / slider / radio option does not raise.

Evidence is printed per page so the results can be quoted directly.

Usage::

    python scripts/verify_features.py                  # all pages
    python scripts/verify_features.py "Network Map"     # pages matching substring
    python scripts/verify_features.py --budget 120      # seconds of widget sweeps per page
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from streamlit.testing.v1 import AppTest  # noqa: E402

# Reuse the navigation reader from the sibling harness (not a package, so add
# the scripts directory itself to the import path).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_dashboard import nav_pages  # noqa: E402

APP = str(PROJECT_ROOT / "app" / "main.py")


def chart_stats(at) -> list[dict]:
    """Trace count and total plotted points per Plotly chart."""
    out = []
    for el in at.get("plotly_chart"):
        try:
            spec = json.loads(el.proto.spec) if el.proto.spec else {}
        except json.JSONDecodeError:
            spec = {}
        traces = spec.get("data", []) or []
        points = 0
        scalar_value = False
        for tr in traces:
            for axis in ("x", "y", "values", "r", "z", "lat", "lon"):
                val = tr.get(axis)
                if isinstance(val, list):
                    points += len(val)
            # Indicator/gauge traces carry a scalar ``value`` and no arrays.
            if isinstance(tr.get("value"), (int, float)):
                scalar_value = True
        out.append({"traces": len(traces), "points": points,
                    "scalar_value": scalar_value})
    return out


def metric_issues(at) -> list[str]:
    """KPI cards whose value looks unpopulated."""
    issues = []
    for m in at.metric:
        text = f"{m.value}"
        if any(tok in text.lower() for tok in ("nan", "inf", "none")):
            issues.append(f"{m.label}={text}")
    return issues


def widget_index(at, kind: str, label: str) -> int:
    """Index of the widget with ``label`` in ``at.<kind>``, or -1 when absent.

    Streamlit re-orders the element tree between reruns, so widget positions must
    be re-resolved by label rather than cached as a numeric index.
    """
    for i, widget in enumerate(getattr(at, kind)):
        if widget.label == label:
            return i
    return -1


def go_to_page(at: AppTest, page: str) -> AppTest:
    """Select ``page`` in the sidebar navigation, resolving it by label."""
    idx = widget_index(at, "selectbox", "Navigate")
    if idx < 0:
        raise RuntimeError("sidebar 'Navigate' selectbox not found")
    at.selectbox[idx].select(page).run()
    return at


def load_page(page: str, timeout: int = 300) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=timeout)
    at.run()
    return go_to_page(at, page)


def safe_options(widget) -> list:
    """Candidate alternative values for a widget, excluding the current one."""
    try:
        options = list(widget.options)
    except (AttributeError, TypeError):
        return []
    return [o for o in options if o != widget.value]


def _formatted_widget_raw_values() -> dict[tuple[str, str], list]:
    """Raw option values for widgets that use a ``format_func``.

    ``AppTest`` exposes ``options`` as the *formatted* strings, so for a formatted
    widget the underlying raw values cannot be recovered from the element. They
    are listed explicitly here so those dropdowns are still exercised (notably
    every Network Map metric and heatmap mode, and the scenario templates).
    """
    from src.map.network_map import get_map_metric_options
    from src.simulator.whatif import SCENARIO_TEMPLATES

    return {
        ("Network Map", "Metric Mode"): [o["value"] for o in get_map_metric_options()],
        ("Audit Timeline", "Time Range"): [6, 24, 72, 168, 720],
        ("What-If Simulator", "Select Scenarios to Run"): list(SCENARIO_TEMPLATES.keys()),
    }


def raw_options(widget, page: str) -> list:
    """Alternative *raw* values for a selectbox/multiselect, excluding current."""
    exposed = list(widget.options)
    current = widget.value
    multi = isinstance(current, (list, tuple))
    selected = list(current) if multi else [current]

    # Without a format_func the exposed options ARE the raw values.
    if all(v in exposed for v in selected):
        return [o for o in exposed if o not in selected]

    for (page_key, label), values in _formatted_widget_raw_values().items():
        if page_key in page and label == widget.label:
            return [v for v in values if v not in selected]
    return []


def sweep_widgets(at, budget_s: float, page: str) -> tuple[int, list[str]]:
    """Switch widget options one at a time, returning (attempts, failures)."""
    attempts = 0
    failures: list[str] = []
    started = time.time()

    def out_of_time() -> bool:
        return time.time() - started > budget_s

    # Dropdowns (skip the sidebar navigation widget).
    dropdown_labels = [sb.label for sb in at.selectbox if sb.label != "Navigate"]
    for label in dropdown_labels:
        idx = widget_index(at, "selectbox", label)
        if idx < 0 or out_of_time():
            continue
        for option in raw_options(at.selectbox[idx], page):
            idx = widget_index(at, "selectbox", label)
            if idx < 0 or out_of_time():
                break
            attempts += 1
            try:
                at.selectbox[idx].select(option).run()
                for exc in at.exception:
                    failures.append(f"selectbox[{label}={option}]: {exc.value}")
            except Exception as exc:  # noqa: BLE001
                failures.append(f"selectbox[{label}={option}]: {type(exc).__name__}: {exc}")

    # Multiselects (pick up to the first three options).
    for label in [ms.label for ms in at.multiselect]:
        idx = widget_index(at, "multiselect", label)
        if idx < 0 or out_of_time():
            continue
        widget = at.multiselect[idx]
        if not widget.options:
            continue
        picks = raw_options(widget, page)[:3]
        if not picks:
            continue
        attempts += 1
        try:
            at.multiselect[idx].set_value(picks).run()
            for exc in at.exception:
                failures.append(f"multiselect[{label}]: {exc.value}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"multiselect[{label}]: {type(exc).__name__}: {exc}")

    # Sliders (min, max, midpoint).
    for label in [sl.label for sl in at.slider]:
        idx = widget_index(at, "slider", label)
        if idx < 0 or out_of_time():
            continue
        widget = at.slider[idx]
        try:
            lo, hi = widget.min, widget.max
        except (AttributeError, TypeError):
            continue
        span = hi - lo
        if isinstance(widget.value, (list, tuple)):
            # Range slider: the value must stay a 2-tuple.
            probes = [(lo, hi), (lo + span * 0.25, lo + span * 0.75), (lo + span * 0.5, hi)]
        else:
            probes = [lo, hi, lo + span / 2]
        for value in probes:
            idx = widget_index(at, "slider", label)
            if idx < 0 or out_of_time():
                break
            attempts += 1
            try:
                at.slider[idx].set_value(value).run()
                for exc in at.exception:
                    failures.append(f"slider[{label}={value}]: {exc.value}")
            except Exception as exc:  # noqa: BLE001
                failures.append(f"slider[{label}={value}]: {type(exc).__name__}: {exc}")

    # Radios.
    for label in [rd.label for rd in at.radio]:
        idx = widget_index(at, "radio", label)
        if idx < 0 or out_of_time():
            continue
        for option in safe_options(at.radio[idx]):
            idx = widget_index(at, "radio", label)
            if idx < 0 or out_of_time():
                break
            attempts += 1
            try:
                at.radio[idx].set_value(option).run()
                for exc in at.exception:
                    failures.append(f"radio[{label}={option}]: {exc.value}")
            except Exception as exc:  # noqa: BLE001
                failures.append(f"radio[{label}={option}]: {type(exc).__name__}: {exc}")

    return attempts, failures


def main() -> int:
    args = [a for a in sys.argv[1:]]
    budget = 180.0
    if "--budget" in args:
        i = args.index("--budget")
        budget = float(args[i + 1])
        del args[i:i + 2]
    wanted = args[0].lower() if args else None

    pages = nav_pages()
    if wanted:
        pages = [p for p in pages if wanted in p.lower()]

    print(f"Feature verification for {len(pages)} page(s), widget budget {budget:.0f}s/page")
    print("=" * 78)
    total_failures = 0
    empty_charts: list[str] = []
    for page in pages:
        t0 = time.time()
        at = load_page(page)
        charts = chart_stats(at)
        issues = metric_issues(at)
        render_fail = [str(e.value) for e in at.exception] + [str(e.value) for e in at.error]

        attempts, widget_failures = sweep_widgets(at, budget, page)
        # Re-read the page after the sweep to confirm the final state is still clean.
        go_to_page(at, page)
        post_fail = [str(e.value) for e in at.exception]

        empty = [c for c in charts if c["points"] == 0 and not c["scalar_value"]]
        if empty:
            empty_charts.append(f"{page}: {len(empty)} empty chart(s)")
        total_failures += len(render_fail) + len(widget_failures) + len(post_fail) + len(issues)

        points = sum(c["points"] for c in charts)
        print(f"\nPAGE {page}")
        print(f"  charts={len(charts)} total_points={points} "
              f"empty_charts={len(empty)} kpi_cards={len(at.metric)}")
        print(f"  widget_switches_attempted={attempts} widget_failures={len(widget_failures)}")
        print(f"  elapsed={time.time() - t0:.1f}s")
        for msg in render_fail:
            print(f"  RENDER ERROR: {msg[:300]}")
        for msg in issues:
            print(f"  BAD KPI:      {msg[:300]}")
        for msg in widget_failures[:12]:
            print(f"  WIDGET ERROR: {msg[:300]}")
        for msg in post_fail:
            print(f"  POST-SWEEP ERROR: {msg[:300]}")

    print("\n" + "=" * 78)
    print(f"TOTAL PROBLEMS: {total_failures}")
    if empty_charts:
        print("PAGES WITH EMPTY CHARTS:")
        for line in empty_charts:
            print("  -", line)
    return 1 if total_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
