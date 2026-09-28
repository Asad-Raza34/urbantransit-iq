"""Dump the real UI of every dashboard page.

Writes an inventory of what each page actually shows -- every widget with its
exact label, its options, every chart title, and element counts -- so
documentation can be written from observed behaviour instead of intent.

Usage::

    python scripts/describe_ui.py                 # all pages
    python scripts/describe_ui.py --pages 3       # first 3 pages only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from verify_dashboard import nav_pages  # noqa: E402
from verify_features import load_page  # noqa: E402

KINDS = [
    "selectbox", "multiselect", "slider", "checkbox", "radio", "button",
    "download_button", "text_input", "text_area", "number_input", "date_input",
    "color_picker", "toggle", "metric", "markdown", "caption", "tab", "expander",
    "dataframe", "table", "json",
]


def _safe(at, kind: str) -> list:
    try:
        return list(at.get(kind))
    except Exception:  # noqa: BLE001
        return []


def chart_titles(at) -> list[str]:
    titles = []
    for el in _safe(at, "plotly_chart"):
        try:
            spec = json.loads(el.proto.spec) if el.proto.spec else {}
            title = spec.get("layout", {}).get("title", {})
            text = title.get("text") if isinstance(title, dict) else title
            titles.append(str(text)[:110] if text else "(no title)")
        except Exception:  # noqa: BLE001
            titles.append("(unreadable spec)")
    return titles


def summarise(widget, kind: str) -> str:
    label = getattr(widget, "label", "") or ""
    try:
        value = widget.value
    except Exception:  # noqa: BLE001
        value = "?"
    extra = ""
    if kind == "selectbox":
        try:
            extra = f" options={list(widget.options)[:8]}"
        except Exception:  # noqa: BLE001
            pass
    elif kind in {"slider", "number_input"}:
        extra = f" range=({getattr(widget, 'min', '?')}..{getattr(widget, 'max', '?')})"
    elif kind == "multiselect":
        try:
            extra = f" n_options={len(widget.options)}"
        except Exception:  # noqa: BLE001
            pass
    shown = str(value)
    return f"{label!r} = {shown}{extra}"


def describe(at, page: str) -> None:
    print(f"\n{'=' * 78}")
    print(f"PAGE: {page}")
    print("=" * 78)
    print(f"  elements: charts={len(chart_titles(at))} "
          f"tables={len(_safe(at, 'dataframe'))} metrics={len(_safe(at, 'metric'))} "
          f"downloads={len(_safe(at, 'download_button'))} json={len(_safe(at, 'json'))}")

    for kind in KINDS:
        items = _safe(at, kind)
        if not items:
            continue
        if kind == "markdown":
            continue
        if kind == "metric":
            for m in items:
                print(f"  KPI      : {m.label!r} = {m.value}")
            continue
        if kind in {"dataframe", "table"}:
            for d in items:
                try:
                    print(f"  TABLE    : {len(d.value)} rows x {len(d.value.columns)} cols "
                          f"{list(d.value.columns)[:10]}")
                except Exception:  # noqa: BLE001
                    print("  TABLE    : (shape unreadable)")
            continue
        for item in items:
            print(f"  {kind.upper():9}: {summarise(item, kind)}")

    titles = chart_titles(at)
    for t in titles:
        print(f"  CHART    : {t}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=0, help="limit to the first N pages")
    args = ap.parse_args()

    pages = nav_pages()
    if args.pages:
        pages = pages[: args.pages]
    print(f"Describing {len(pages)} page(s)")
    for page in pages:
        try:
            at = load_page(page, timeout=900)
        except Exception as exc:  # noqa: BLE001
            print(f"\nPAGE: {page}\n  LOAD ERROR: {exc}")
            continue
        if at.exception:
            print(f"\nPAGE: {page}\n  EXCEPTION: {at.exception[0].value}")
            continue
        describe(at, page)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
