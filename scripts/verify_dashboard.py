"""Headless dashboard verification harness.

Runs the real Streamlit app through ``streamlit.testing.v1.AppTest`` and walks
every entry in the sidebar navigation, reporting any exception or error that the
page raised. This is first-hand evidence that a page renders, not a claim.

Usage::

    python scripts/verify_dashboard.py            # every page
    python scripts/verify_dashboard.py "Network Map"   # one page (substring match)
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Windows consoles default to cp1252 and cannot encode the emoji in the nav
# labels; force UTF-8 so the report prints instead of crashing.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(PROJECT_ROOT / "app" / "main.py")


def nav_pages() -> list[str]:
    """Read the real navigation list out of app/main.py (single source of truth).

    The list is the module-level ``ALL_PAGES`` assignment; the sidebar selectbox
    then filters it by the current user's role.
    """
    import ast

    tree = ast.parse((PROJECT_ROOT / "app" / "main.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "ALL_PAGES":
                    if isinstance(node.value, ast.List):
                        return [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
    raise RuntimeError("could not find ALL_PAGES in app/main.py")


def run_page(page: str, timeout: int = 300) -> dict:
    result = {"page": page, "ok": False, "exceptions": [], "errors": [], "seconds": 0.0}
    start = time.time()
    try:
        at = AppTest.from_file(APP, default_timeout=timeout)
        at.run()
        if at.exception:
            result["exceptions"].append(
                f"[startup] {at.exception[0].value or at.exception[0].message}"
            )
        # Select the page in the sidebar navigation and re-run.
        at.selectbox[0].select(page).run()
        for exc in at.exception:
            result["exceptions"].append(str(exc.value or exc.message))
        for err in at.error:
            result["errors"].append(str(err.value))
    except Exception as exc:  # noqa: BLE001
        result["exceptions"].append(f"{type(exc).__name__}: {exc}")
        result["traceback"] = traceback.format_exc()
    result["seconds"] = round(time.time() - start, 1)
    result["ok"] = not result["exceptions"] and not result["errors"]
    return result


def main() -> int:
    pages = nav_pages()
    wanted = sys.argv[1].lower() if len(sys.argv) > 1 else None
    if wanted:
        pages = [p for p in pages if wanted in p.lower()]
    print(f"Verifying {len(pages)} dashboard page(s)\n" + "=" * 72)
    failures = 0
    for page in pages:
        res = run_page(page)
        flag = "PASS" if res["ok"] else "FAIL"
        if not res["ok"]:
            failures += 1
        print(f"[{flag}] {page}  ({res['seconds']}s)")
        for e in res["exceptions"]:
            print(f"        EXCEPTION: {e[:400]}")
        for e in res["errors"]:
            print(f"        ERROR:     {e[:400]}")
        if not res["ok"] and res.get("traceback"):
            print(res["traceback"][-1500:])
    print("=" * 72)
    print(f"RESULT: {len(pages) - failures}/{len(pages)} pages rendered without exceptions/errors")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
