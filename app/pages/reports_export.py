"""Reports & Export Dashboard Page.

Implements the reporting and export requirements: PDF analytical reports with
embedded charts, and CSV / JSON / Excel / Parquet export of the analytical
results, plus a bundled ZIP package.
"""

import io
import json
import zipfile
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from src.audit import record
from src.paths import REPORTS_ROOT
from src.reports.generator import ReportGenerator, SUPPORTED_REPORT_TYPES
from src.services.data_access import (
    load_alerts,
    load_audit_summary,
    load_audit_timeline,
    load_crowding_analytics,
    load_demand_analytics,
    load_frequency_analytics,
    load_recommendations,
    load_route_metrics,
    load_segmentation,
    load_stop_metrics,
    calculate_kpis,
)
from src.services.export import ExportService

# Datasets offered for export: label -> callable returning a DataFrame.
EXPORTABLE = {
    "Route Metrics": load_route_metrics,
    "Stop Metrics": load_stop_metrics,
    "Alerts": load_alerts,
    "Recommendations": load_recommendations,
    "Audit Timeline": lambda: load_audit_timeline(limit=5000),
    "Audit Summary": load_audit_summary,
    "Daily Demand": lambda: load_demand_analytics()["daily_demand"],
    "Peak Patterns": lambda: load_demand_analytics()["peak_patterns"],
    "Overcrowding Events": lambda: load_crowding_analytics()["overcrowding_events"],
    "Persistent Overcrowding": lambda: load_crowding_analytics()["persistent_overcrowding"],
    "Underutilization": lambda: load_crowding_analytics()["underutilization"],
    "Frequency Analysis": lambda: load_frequency_analytics()["frequency_analysis"],
    "Headway Analysis": lambda: load_frequency_analytics()["headway_analysis"],
    "Service Level": lambda: load_frequency_analytics()["service_level"],
    "Segmented Passengers": lambda: load_segmentation()["segmented_passengers"],
    "Segment Summary": lambda: load_segmentation()["segment_summary"],
}

EXPORT_FORMATS = ["CSV", "JSON", "Excel", "Parquet"]


def _df_to_bytes(df: pd.DataFrame, fmt: str) -> bytes:
    """Serialise a DataFrame to the requested export format."""
    if fmt == "CSV":
        return df.to_csv(index=False).encode("utf-8")
    if fmt == "JSON":
        return df.to_json(orient="records", date_format="iso", indent=2).encode("utf-8")
    if fmt == "Excel":
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.head(100_000).to_excel(writer, sheet_name="data", index=False)
        return buf.getvalue()
    if fmt == "Parquet":
        buf = io.BytesIO()
        # Parquet needs string column names and no duplicate names.
        df.to_parquet(buf, index=False)
        return buf.getvalue()
    raise ValueError(f"Unsupported format: {fmt}")


def _extension(fmt: str) -> str:
    return {"CSV": "csv", "JSON": "json", "Excel": "xlsx", "Parquet": "parquet"}[fmt]


def render():
    st.markdown('<div class="main-header">📄 Reports & Export</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background: #e7f3ff; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <strong>📄 Reports & Export:</strong> Generate a PDF analytical report with embedded charts,
    or export any analytical result as CSV, JSON, Excel or Parquet.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📄 PDF Reports", "📦 Data Export"])

    with tab1:
        render_pdf_reports()
    with tab2:
        render_data_export()


def render_pdf_reports():
    """PDF report generation with embedded Plotly charts."""
    st.markdown('<div class="sub-header">Generate PDF Report</div>', unsafe_allow_html=True)

    kpis = calculate_kpis()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Routes", kpis.get("total_routes", 0))
    with col2:
        st.metric("On-Time %", f"{kpis.get('on_time_pct', 0):.1f}%")
    with col3:
        st.metric("Active Alerts", kpis.get("active_alerts", 0))
    with col4:
        st.metric("Avg Route Score", f"{kpis.get('route_score_avg', 0):.1f}")

    col1, col2 = st.columns([2, 1])
    with col1:
        report_type = st.selectbox(
            "Report Type", list(SUPPORTED_REPORT_TYPES), key="report_type_select"
        )
    with col2:
        include_charts = st.checkbox(
            "Embed charts", value=True, key="report_include_charts",
            help="Render the analytical charts into the PDF (requires Kaleido).",
        )

    if st.button("📄 Generate PDF Report", type="primary", key="generate_pdf_btn"):
        with st.spinner("Building PDF report (rendering charts can take a moment)..."):
            try:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                out_path = REPORTS_ROOT / f"report_{report_type}_{stamp}.pdf"
                generator = ReportGenerator()
                if include_charts:
                    generator.generate_executive_report(out_path)
                else:
                    # Temporarily disable chart embedding for a lightweight report.
                    import src.reports.generator as report_module
                    original = report_module.KALEIDO_AVAILABLE
                    report_module.KALEIDO_AVAILABLE = False
                    try:
                        generator.generate_executive_report(out_path)
                    finally:
                        report_module.KALEIDO_AVAILABLE = original

                st.session_state["pdf_bytes"] = out_path.read_bytes()
                st.session_state["pdf_name"] = out_path.name
                st.session_state["pdf_size"] = len(st.session_state["pdf_bytes"])
                record(action="report:generate", component="reports",
                       details=json.dumps({"report_type": report_type,
                                           "charts": include_charts,
                                           "path": str(out_path)}))
                st.success(f"Report generated: {out_path.name}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Report generation failed: {type(exc).__name__}: {exc}")

    pdf_bytes = st.session_state.get("pdf_bytes")
    if pdf_bytes:
        st.download_button(
            "📥 Download PDF Report",
            pdf_bytes,
            st.session_state.get("pdf_name", "urbantransit_report.pdf"),
            "application/pdf",
        )
        st.caption(
            f"Loaded size: {st.session_state.get('pdf_size', len(pdf_bytes)):,} bytes — "
            "report generated in this session and saved under `reports/`."
        )

    st.markdown("---")
    st.markdown('<div class="sub-header">Existing Reports on Disk</div>', unsafe_allow_html=True)
    existing = sorted(REPORTS_ROOT.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    if existing:
        st.dataframe(
            pd.DataFrame([
                {"File": p.name, "Size (KB)": round(p.stat().st_size / 1024, 1),
                 "Modified (UTC)": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
                                   .strftime("%Y-%m-%d %H:%M")}
                for p in existing[:25]
            ]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No PDF reports generated yet.")


def render_data_export():
    """Export analytical results in CSV / JSON / Excel / Parquet."""
    st.markdown('<div class="sub-header">Export Analytical Results</div>', unsafe_allow_html=True)

    available = [name for name in EXPORTABLE]
    selected = st.multiselect(
        "Datasets to export",
        available,
        default=["Route Metrics"],
        key="export_datasets",
    )
    fmt = st.selectbox("Format", EXPORT_FORMATS, key="export_format")

    if st.button("📦 Build Export", type="primary", key="build_export_btn"):
        if not selected:
            st.warning("Select at least one dataset.")
        else:
            try:
                frames = {name: EXPORTABLE[name]() for name in selected}
                frames = {k: v for k, v in frames.items() if isinstance(v, pd.DataFrame)}
                empty = [k for k, v in frames.items() if v.empty]
                if len(frames) == 1 and not empty:
                    name, df = next(iter(frames.items()))
                    payload = _df_to_bytes(df, fmt)
                    ext = _extension(fmt)
                    st.session_state["export_bytes"] = payload
                    st.session_state["export_name"] = (
                        f"{name.lower().replace(' ', '_')}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.{ext}"
                    )
                    st.session_state["export_mime"] = "application/octet-stream"
                    st.session_state["export_note"] = f"{name}: {len(df):,} rows"
                else:
                    # Multiple datasets: always bundle as a ZIP of CSVs so every
                    # selected frame survives regardless of the chosen format.
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        for name, df in frames.items():
                            zf.writestr(f"{name.lower().replace(' ', '_')}.csv", df.to_csv(index=False))
                    st.session_state["export_bytes"] = buf.getvalue()
                    st.session_state["export_name"] = (
                        f"urbantransit_export_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.zip"
                    )
                    st.session_state["export_mime"] = "application/zip"
                    st.session_state["export_note"] = (
                        f"{len(frames)} datasets bundled as ZIP of CSVs: "
                        + ", ".join(f"{k} ({len(v):,} rows)" for k, v in frames.items())
                    )
                if empty:
                    st.warning(f"No rows available for: {', '.join(empty)}")
                record(action="export:build", component="reports",
                       details=json.dumps({"datasets": selected, "format": fmt}))
                st.success("Export built.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Export failed: {type(exc).__name__}: {exc}")

    export_bytes = st.session_state.get("export_bytes")
    if export_bytes:
        st.download_button(
            "📥 Download Export",
            export_bytes,
            st.session_state.get("export_name", "urbantransit_export.bin"),
            st.session_state.get("export_mime", "application/octet-stream"),
        )
        st.caption(f"{st.session_state.get('export_note', '')} — "
                   f"size {len(export_bytes):,} bytes")

    st.markdown("---")
    st.markdown('<div class="sub-header">Full Report Package</div>', unsafe_allow_html=True)
    st.caption("Bundles KPIs, route/stop metrics, crowding and demand analytics, "
               "recommendations, alerts and the audit summary into one ZIP.")

    if st.button("🗜️ Build Full Report Package (ZIP)", key="build_zip_pkg"):
        with st.spinner("Assembling report package..."):
            try:
                service = ExportService()
                zip_path = service.create_full_report_package()
                st.session_state["pkg_bytes"] = zip_path.read_bytes()
                st.session_state["pkg_name"] = zip_path.name
                record(action="export:package", component="reports",
                       details=json.dumps({"path": str(zip_path)}))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Package build failed: {type(exc).__name__}: {exc}")

    pkg_bytes = st.session_state.get("pkg_bytes")
    if pkg_bytes:
        st.download_button(
            "📥 Download Report Package",
            pkg_bytes,
            st.session_state.get("pkg_name", "urbantransit_report.zip"),
            "application/zip",
        )
        st.caption(f"Package size {len(pkg_bytes):,} bytes")


if __name__ == "__main__":
    render()
