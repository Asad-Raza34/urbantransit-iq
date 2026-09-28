"""Export service for reports and data exports."""

import json
import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src import paths
from src.paths import DATA_ANALYTICS
from src.storage import write_dataset
from src.log import get_logger

logger = get_logger(__name__)


def _reports_root() -> Path:
    """Resolve the export directory on every call.

    Binding ``REPORTS_ROOT`` at import time froze the value into this module, so
    anything that redirected ``src.paths.REPORTS_ROOT`` -- notably
    ``tests/test_export_registry.py`` -- still wrote into the real ``reports/``
    folder, leaving ``test_export.csv`` and ``test_json.json`` in the project's
    output directory after every test run. ``src.paths.report_path`` already
    resolves at call time, so this matches the existing convention.
    """
    return paths.REPORTS_ROOT


class ExportService:
    """Handles export of analytics, recommendations, alerts, and reports."""

    def __init__(self):
        _reports_root().mkdir(parents=True, exist_ok=True)

    def export_to_csv(self, df: pd.DataFrame, name: str) -> Path:
        """Export DataFrame to CSV."""
        path = _reports_root() / f"{name}.csv"
        write_dataset(df, path)
        return path

    def export_to_json(self, data: Any, name: str) -> Path:
        """Export data to JSON."""
        path = _reports_root() / f"{name}.json"
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return path

    def export_to_excel(self, sheets: dict[str, pd.DataFrame], name: str) -> Path:
        """Export multiple DataFrames to Excel workbook."""
        path = _reports_root() / f"{name}.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for sheet_name, df in sheets.items():
                # Excel sheet names max 31 chars
                safe_name = sheet_name[:31]
                df.to_excel(writer, sheet_name=safe_name, index=False)
        return path

    def export_recommendations(self, format: str = "csv") -> Path:
        """Export recommendations."""
        from src.services.data_access import load_recommendations
        df = load_recommendations()
        name = f"recommendations_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        if format == "csv":
            return self.export_to_csv(df, name)
        elif format == "json":
            return self.export_to_json(df.to_dict("records"), name)
        elif format == "excel":
            return self.export_to_excel({"recommendations": df}, name)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def export_alerts(self, format: str = "csv", status: list[str] = None) -> Path:
        """Export alerts."""
        from src.services.data_access import load_alerts
        df = load_alerts()
        if status:
            df = df[df["status"].isin(status)]
        name = f"alerts_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        if format == "csv":
            return self.export_to_csv(df, name)
        elif format == "json":
            return self.export_to_json(df.to_dict("records"), name)
        elif format == "excel":
            return self.export_to_excel({"alerts": df}, name)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def export_route_comparison(self, route_ids: list[str], format: str = "csv") -> Path:
        """Export route comparison."""
        from src.comparison.routes import compare_routes
        cmp = compare_routes(route_ids, include_all_metrics=True)
        df = pd.DataFrame(cmp.metrics).T  # transpose for readability
        df.index.name = "route_id"
        name = f"route_comparison_{'_'.join(route_ids)}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        if format == "csv":
            return self.export_to_csv(df.reset_index(), name)
        elif format == "json":
            return self.export_to_json(cmp.to_dict(), name)
        elif format == "excel":
            return self.export_to_excel({"comparison": df.reset_index()}, name)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def export_audit_timeline(self, format: str = "csv", limit: int = 1000) -> Path:
        """Export audit timeline."""
        from src.services.data_access import load_audit_timeline
        df = load_audit_timeline(limit=limit)
        name = f"audit_timeline_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        if format == "csv":
            return self.export_to_csv(df, name)
        elif format == "json":
            return self.export_to_json(df.to_dict("records"), name)
        elif format == "excel":
            return self.export_to_excel({"audit_timeline": df}, name)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def export_scenario_results(self, scenario_results: list[dict], format: str = "json") -> Path:
        """Export what-if scenario results."""
        name = f"scenario_results_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        if format == "json":
            return self.export_to_json(scenario_results, name)
        elif format == "excel":
            # Flatten for Excel
            sheets = {}
            for i, res in enumerate(scenario_results):
                sheets[f"scenario_{i}"] = pd.DataFrame([res])
            return self.export_to_excel(sheets, name)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def create_full_report_package(self) -> Path:
        """Create a comprehensive ZIP package with all reports."""
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        zip_path = _reports_root() / f"urbantransit_report_{timestamp}.zip"

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # KPIs
            from src.services.data_access import calculate_kpis
            kpis = calculate_kpis()
            zf.writestr("kpis.json", json.dumps(kpis, indent=2, default=str))

            # Route metrics
            from src.services.data_access import load_route_metrics
            route_m = load_route_metrics()
            zf.writestr("route_metrics.csv", route_m.to_csv(index=False))

            # Stop metrics
            from src.services.data_access import load_stop_metrics
            stop_m = load_stop_metrics()
            zf.writestr("stop_metrics.csv", stop_m.to_csv(index=False))

            # Crowding
            from src.services.data_access import load_crowding_analytics
            crowding = load_crowding_analytics()
            for name, df in crowding.items():
                if not df.empty:
                    zf.writestr(f"crowding_{name}.csv", df.to_csv(index=False))

            # Demand
            from src.services.data_access import load_demand_analytics
            demand = load_demand_analytics()
            for name, df in demand.items():
                if not df.empty:
                    zf.writestr(f"demand_{name}.csv", df.to_csv(index=False))

            # Recommendations
            from src.services.data_access import load_recommendations
            recs = load_recommendations()
            if not recs.empty:
                zf.writestr("recommendations.csv", recs.to_csv(index=False))

            # Alerts
            from src.services.data_access import load_alerts
            alerts = load_alerts()
            if not alerts.empty:
                zf.writestr("alerts.csv", alerts.to_csv(index=False))

            # Segmentation
            from src.services.data_access import load_segmentation
            seg = load_segmentation()
            for name, df in seg.items():
                if not df.empty:
                    zf.writestr(f"segmentation_{name}.csv", df.to_csv(index=False))

            # Audit summary
            from src.services.data_access import load_audit_summary
            audit_sum = load_audit_summary()
            zf.writestr("audit_summary.csv", audit_sum.to_csv(index=False))

            # Analytics summary
            from src.services.data_access import load_analytics_summary
            ana_sum = load_analytics_summary()
            zf.writestr("analytics_summary.json", json.dumps(ana_sum, indent=2, default=str))

        return zip_path


if __name__ == "__main__":
    svc = ExportService()
    # Test
    path = svc.export_recommendations("csv")
    print(f"Exported to: {path}")