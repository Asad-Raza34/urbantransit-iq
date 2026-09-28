"""Production monitoring for UrbanTransit IQ.

Provides health checks, metrics, and Prometheus integration.
"""

import os
import time
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import psutil

from config import settings
from src.log import get_logger
from src.paths import DATA_ANALYTICS, DATA_PROCESSED, MODELS_PYTHON, DATA_METADATA

logger = get_logger(__name__)


@dataclass
class HealthCheck:
    """Health check result."""
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    timestamp: str
    latency_ms: float = 0.0


@dataclass
class SystemMetrics:
    """System-level metrics."""
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_percent: float
    disk_free_gb: float
    process_count: int
    uptime_seconds: float


class HealthChecker:
    """Performs health checks on system components."""
    
    def __init__(self):
        self.start_time = time.time()
    
    def check_data_files(self) -> HealthCheck:
        """Check if critical data files exist."""
        start = time.time()
        critical_files = [
            DATA_ANALYTICS / "route_metrics.parquet",
            DATA_ANALYTICS / "stop_metrics.parquet",
            DATA_PROCESSED / "trip_facts.parquet",
        ]
        
        missing = [f for f in critical_files if not f.exists()]
        latency = (time.time() - start) * 1000
        
        if missing:
            return HealthCheck(
                component="data_files",
                status="unhealthy",
                message=f"Missing critical files: {[str(f) for f in missing]}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                latency_ms=latency,
            )
        return HealthCheck(
            component="data_files",
            status="healthy",
            message="All critical data files present",
            timestamp=datetime.now(timezone.utc).isoformat(),
            latency_ms=latency,
        )
    
    def check_model_files(self) -> HealthCheck:
        """Check if model files exist."""
        start = time.time()
        model_files = [
            MODELS_PYTHON / "occupancy_forecast_random_forest.joblib",
            MODELS_PYTHON / "occupancy_forecast_linear.joblib",
        ]
        
        missing = [f for f in model_files if not f.exists()]
        latency = (time.time() - start) * 1000
        
        if missing:
            return HealthCheck(
                component="model_files",
                status="degraded",
                message=f"Some model files missing: {[str(f) for f in missing]}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                latency_ms=latency,
            )
        return HealthCheck(
            component="model_files",
            status="healthy",
            message="Model files present",
            timestamp=datetime.now(timezone.utc).isoformat(),
            latency_ms=latency,
        )
    
    def check_spark_availability(self) -> HealthCheck:
        """Check Spark availability."""
        start = time.time()
        try:
            from src.spark_context import spark_available
            ok, msg = spark_available()
            latency = (time.time() - start) * 1000
            
            if ok:
                return HealthCheck(
                    component="spark",
                    status="healthy",
                    message=msg,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    latency_ms=latency,
                )
            else:
                return HealthCheck(
                    component="spark",
                    status="degraded",
                    message=msg,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    latency_ms=latency,
                )
        except Exception as e:
            latency = (time.time() - start) * 1000
            return HealthCheck(
                component="spark",
                status="unhealthy",
                message=f"Spark check failed: {e}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                latency_ms=latency,
            )
    
    def check_hdfs(self) -> HealthCheck:
        """Check HDFS/storage availability."""
        start = time.time()
        try:
            from src.hdfs_store import test_hdfs_connection
            result = test_hdfs_connection()
            latency = (time.time() - start) * 1000
            
            if result.get("connection_ok", False):
                return HealthCheck(
                    component="hdfs",
                    status="healthy",
                    message=f"HDFS mode: {result.get('mode')}, connection OK",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    latency_ms=latency,
                )
            elif result.get("mode") == "local":
                return HealthCheck(
                    component="hdfs",
                    status="healthy",
                    message="Local mirror mode active",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    latency_ms=latency,
                )
            else:
                return HealthCheck(
                    component="hdfs",
                    status="degraded",
                    message=result.get("error", "HDFS connection failed"),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    latency_ms=latency,
                )
        except Exception as e:
            latency = (time.time() - start) * 1000
            return HealthCheck(
                component="hdfs",
                status="unhealthy",
                message=f"HDFS check failed: {e}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                latency_ms=latency,
            )
    
    def check_disk_space(self) -> HealthCheck:
        """Check disk space."""
        start = time.time()
        try:
            disk = psutil.disk_usage("/")
            free_gb = disk.free / (1024**3)
            percent_used = (disk.used / disk.total) * 100
            latency = (time.time() - start) * 1000
            
            if percent_used > 95:
                status = "unhealthy"
                msg = f"Critical: Disk {percent_used:.1f}% used, {free_gb:.1f} GB free"
            elif percent_used > 85:
                status = "degraded"
                msg = f"Warning: Disk {percent_used:.1f}% used, {free_gb:.1f} GB free"
            else:
                status = "healthy"
                msg = f"Disk {percent_used:.1f}% used, {free_gb:.1f} GB free"
            
            return HealthCheck(
                component="disk_space",
                status=status,
                message=msg,
                timestamp=datetime.now(timezone.utc).isoformat(),
                latency_ms=latency,
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            return HealthCheck(
                component="disk_space",
                status="unhealthy",
                message=f"Disk check failed: {e}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                latency_ms=latency,
            )
    
    def run_all_checks(self) -> list[HealthCheck]:
        """Run all health checks."""
        checks = [
            self.check_data_files(),
            self.check_model_files(),
            self.check_spark_availability(),
            self.check_hdfs(),
            self.check_disk_space(),
        ]
        return checks


def get_system_metrics() -> SystemMetrics:
    """Get current system metrics."""
    cpu = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    
    return SystemMetrics(
        cpu_percent=cpu,
        memory_percent=memory.percent,
        memory_available_gb=memory.available / (1024**3),
        disk_percent=(disk.used / disk.total) * 100,
        disk_free_gb=disk.free / (1024**3),
        process_count=len(psutil.pids()),
        uptime_seconds=time.time() - psutil.boot_time(),
    )


def run_health_checks() -> dict[str, Any]:
    """Run all health checks and return summary."""
    checker = HealthChecker()
    checks = checker.run_all_checks()
    
    overall_status = "healthy"
    for check in checks:
        if check.status == "unhealthy":
            overall_status = "unhealthy"
            break
        elif check.status == "degraded" and overall_status == "healthy":
            overall_status = "degraded"
    
    return {
        "status": overall_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": [asdict(c) for c in checks],
        "uptime_seconds": time.time() - checker.start_time,
    }


def record_metrics_sample(path: Path = None) -> dict:
    """Append one system-metrics sample to the on-disk history.

    Health Monitoring previously had no way to show trends because nothing kept a
    time series. Each sample is one small JSON object per line, which lets the
    dashboard plot real CPU/memory/disk history instead of a placeholder.
    """
    if path is None:
        path = DATA_METADATA / "metrics_history.jsonl"
    metrics = get_system_metrics()
    sample = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cpu_percent": round(metrics.cpu_percent, 2),
        "memory_percent": round(metrics.memory_percent, 2),
        "disk_percent": round(metrics.disk_percent, 2),
        "process_count": int(metrics.process_count),
        "uptime_seconds": round(metrics.uptime_seconds, 1),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(sample) + "\n")
    return sample


def load_metrics_history(limit: int = 500, path: Path = None) -> "pd.DataFrame":
    """Load recorded system-metrics samples oldest-first."""
    import pandas as pd

    if path is None:
        path = DATA_METADATA / "metrics_history.jsonl"
    if not path.exists():
        return pd.DataFrame(
            columns=["ts", "cpu_percent", "memory_percent", "disk_percent",
                     "process_count", "uptime_seconds"]
        )
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts"]).sort_values("ts")
    return df.tail(limit).reset_index(drop=True)


def get_prometheus_metrics() -> str:
    """Generate Prometheus-format metrics."""
    metrics = get_system_metrics()
    checker = HealthChecker()
    checks = checker.run_all_checks()
    
    lines = [
        "# HELP urbantransit_cpu_percent CPU usage percentage",
        "# TYPE urbantransit_cpu_percent gauge",
        f"urbantransit_cpu_percent {metrics.cpu_percent}",
        "",
        "# HELP urbantransit_memory_percent Memory usage percentage",
        "# TYPE urbantransit_memory_percent gauge",
        f"urbantransit_memory_percent {metrics.memory_percent}",
        "",
        "# HELP urbantransit_disk_percent Disk usage percentage",
        "# TYPE urbantransit_disk_percent gauge",
        f"urbantransit_disk_percent {metrics.disk_percent}",
        "",
        "# HELP urbantransit_uptime_seconds Application uptime in seconds",
        "# TYPE urbantransit_uptime_seconds counter",
        f"urbantransit_uptime_seconds {metrics.uptime_seconds:.0f}",
        "",
        "# HELP urbantransit_health_status Component health status (1=healthy, 0.5=degraded, 0=unhealthy)",
        "# TYPE urbantransit_health_status gauge",
    ]
    
    status_map = {"healthy": 1, "degraded": 0.5, "unhealthy": 0}
    for check in checks:
        status_val = status_map.get(check.status, 0)
        lines.append(f'urbantransit_health_status{{component="{check.component}"}} {status_val}')
    
    return "\n".join(lines)


def write_health_report(output_path: Path = None) -> Path:
    """Write health check report to file."""
    if output_path is None:
        output_path = DATA_METADATA / "health_report.json"
    
    report = run_health_checks()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


# Streamlit integration for health dashboard
def render_health_dashboard():
    """Render health monitoring dashboard in Streamlit."""
    import streamlit as st
    import plotly.express as px
    import pandas as pd
    
    st.markdown('<div class="main-header">🏥 System Health Monitoring</div>', unsafe_allow_html=True)
    
    # Run checks
    checker = HealthChecker()
    checks = checker.run_all_checks()
    metrics = get_system_metrics()
    
    # Overall status
    status_colors = {"healthy": "🟢", "degraded": "🟡", "unhealthy": "🔴"}
    overall = "healthy"
    for c in checks:
        if c.status == "unhealthy":
            overall = "unhealthy"
            break
        elif c.status == "degraded" and overall == "healthy":
            overall = "degraded"
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Status", f"{status_colors.get(overall, '⚪')} {overall.title()}")
    with col2:
        st.metric("CPU Usage", f"{metrics.cpu_percent:.1f}%")
    with col3:
        st.metric("Memory Usage", f"{metrics.memory_percent:.1f}%")
    with col4:
        st.metric("Disk Usage", f"{metrics.disk_percent:.1f}%")
    
    st.markdown("---")
    
    # Component checks
    st.markdown('<div class="sub-header">Component Health Checks</div>', unsafe_allow_html=True)
    
    check_data = []
    for check in checks:
        check_data.append({
            "Component": check.component,
            "Status": f"{status_colors.get(check.status, '⚪')} {check.status.title()}",
            "Message": check.message,
            "Latency (ms)": f"{check.latency_ms:.1f}",
        })
    
    df = pd.DataFrame(check_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # System metrics chart
    st.markdown('<div class="sub-header">System Metrics</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            x=["CPU", "Memory", "Disk"],
            y=[metrics.cpu_percent, metrics.memory_percent, metrics.disk_percent],
            labels={"x": "Resource", "y": "Usage %"},
            title="Resource Utilization",
            color=["CPU", "Memory", "Disk"],
            color_discrete_map={"CPU": "#1f77b4", "Memory": "#ff7f0e", "Disk": "#2ca02c"}
        )
        fig.add_hline(y=85, line_dash="dash", line_color="orange", annotation_text="Warning (85%)")
        fig.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="Critical (95%)")
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Health check timeline
        timestamps = [c.timestamp for c in checks]
        statuses = [c.status for c in checks]
        # Create a simple timeline
        check_df = pd.DataFrame(check_data)
        st.dataframe(check_df, use_container_width=True)
    
    # Prometheus metrics endpoint info
    st.markdown('<div class="sub-header">Prometheus Metrics</div>', unsafe_allow_html=True)
    st.code(get_prometheus_metrics(), language="text")
    st.caption("Metrics available at /metrics endpoint (when running with Prometheus exporter)")


if __name__ == "__main__":
    # Run health checks and print report
    report = run_health_checks()
    print(json.dumps(report, indent=2))
    print("\nPrometheus metrics:")
    print(get_prometheus_metrics())