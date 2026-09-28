"""System Health Monitoring Dashboard Page."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.monitoring import (
    HealthChecker, get_system_metrics, get_prometheus_metrics,
    record_metrics_sample, load_metrics_history,
)


def render():
    st.markdown('<div class="main-header">🏥 System Health Monitoring</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background: #e7f3ff; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <strong>🏥 System Health Monitoring:</strong> Real-time health checks for data, models, Spark, HDFS, and system resources.
    Includes Prometheus-compatible metrics export.
    </div>
    """, unsafe_allow_html=True)

    # Run health checks
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

    # Status indicator
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

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Component Health", "📊 System Metrics", "📈 Resource Trends", "📋 Prometheus Metrics"])

    with tab1:
        render_component_health(checks)

    with tab2:
        render_system_metrics(metrics)

    with tab3:
        render_resource_trends()

    with tab4:
        render_prometheus_metrics()


def render_component_health(checks):
    """Render component health checks."""
    st.markdown('<div class="sub-header">Component Health Checks</div>', unsafe_allow_html=True)

    status_colors = {"healthy": "🟢", "degraded": "🟡", "unhealthy": "🔴"}

    check_data = []
    for check in checks:
        check_data.append({
            "Component": check.component,
            "Status": f"{status_colors.get(check.status, '⚪')} {check.status.title()}",
            "Message": check.message,
            "Latency (ms)": f"{check.latency_ms:.1f}",
            "Timestamp": check.timestamp,
        })

    df = pd.DataFrame(check_data)
    
    # Color-code the dataframe
    def color_status(val):
        if "🟢" in val:
            return 'background-color: #d4edda'
        elif "🟡" in val:
            return 'background-color: #fff3cd'
        elif "🔴" in val:
            return 'background-color: #f8d7da'
        return ''

    styled_df = df.style.map(color_status, subset=['Status'])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # Summary counts
    st.markdown('<div class="sub-header">Health Summary</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        healthy_count = sum(1 for c in checks if c.status == "healthy")
        st.metric("Healthy", healthy_count)
    with col2:
        degraded_count = sum(1 for c in checks if c.status == "degraded")
        st.metric("Degraded", degraded_count)
    with col3:
        unhealthy_count = sum(1 for c in checks if c.status == "unhealthy")
        st.metric("Unhealthy", unhealthy_count)


def render_system_metrics(metrics):
    """Render system metrics charts."""
    st.markdown('<div class="sub-header">System Resource Utilization</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Resource utilization bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["CPU", "Memory", "Disk"],
            y=[metrics.cpu_percent, metrics.memory_percent, metrics.disk_percent],
            marker_color=["#1f77b4", "#ff7f0e", "#2ca02c"],
            text=[f"{metrics.cpu_percent:.1f}%", f"{metrics.memory_percent:.1f}%", f"{metrics.disk_percent:.1f}%"],
            textposition='auto',
        ))
        fig.add_hline(y=85, line_dash="dash", line_color="orange", annotation_text="Warning (85%)")
        fig.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="Critical (95%)")
        fig.update_layout(
            title="Resource Utilization",
            yaxis_title="Usage %",
            yaxis_range=[0, 100],
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Detailed metrics
        st.markdown('<div class="sub-header">Detailed Metrics</div>', unsafe_allow_html=True)
        
        metric_data = {
            "Metric": [
                "CPU Usage (%)",
                "Memory Usage (%)",
                "Available Memory (GB)",
                "Disk Usage (%)",
                "Free Disk (GB)",
                "Process Count",
                "Uptime (hours)"
            ],
            "Value": [
                f"{metrics.cpu_percent:.1f}",
                f"{metrics.memory_percent:.1f}",
                f"{metrics.memory_available_gb:.2f}",
                f"{metrics.disk_percent:.1f}",
                f"{metrics.disk_free_gb:.2f}",
                str(metrics.process_count),
                f"{metrics.uptime_seconds / 3600:.1f}"
            ]
        }
        
        df = pd.DataFrame(metric_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Gauge charts
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics.cpu_percent,
            title={'text': "CPU %"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#1f77b4"},
                   'steps': [
                       {'range': [0, 85], 'color': "lightgreen"},
                       {'range': [85, 95], 'color': "orange"},
                       {'range': [95, 100], 'color': "red"}],
                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 95}}
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


def render_resource_trends():
    """Render resource trend monitoring from the collected metrics history."""
    st.markdown('<div class="sub-header">Resource Trends</div>', unsafe_allow_html=True)

    # Record this page view as a sample, then plot the accumulated history.
    # The time series is built from real measurements rather than a placeholder.
    record_metrics_sample()
    history = load_metrics_history(limit=500)

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("📸 Record Sample", key="health_record_sample"):
            record_metrics_sample()
            st.rerun()
    with col1:
        st.caption(
            f"{len(history)} sample(s) collected. A sample is appended every time this tab "
            "is opened (or when you press Record Sample), so the trend fills in as the "
            "application is used."
        )

    if history.empty:
        st.info("No samples recorded yet. Reload this tab to collect the first sample.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["ts"], y=history["cpu_percent"],
                             mode="lines+markers", name="CPU %"))
    fig.add_trace(go.Scatter(x=history["ts"], y=history["memory_percent"],
                             mode="lines+markers", name="Memory %"))
    fig.add_trace(go.Scatter(x=history["ts"], y=history["disk_percent"],
                             mode="lines+markers", name="Disk %"))
    fig.add_hline(y=85, line_dash="dash", line_color="orange", annotation_text="Warning (85%)")
    fig.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="Critical (95%)")
    fig.update_layout(
        title="Resource Utilisation Over Time",
        yaxis_title="Usage %",
        yaxis_range=[0, 100],
        height=400,
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Recent samples**")
    st.dataframe(history.tail(20).iloc[::-1], use_container_width=True, hide_index=True)

    st.markdown("""
    **Recommended Production Setup:**
    - Deploy Prometheus node exporter on all hosts
    - Configure Prometheus to scrape metrics every 15s
    - Use Grafana for dashboard visualization
    - Set up alerting rules for:
      - CPU > 85% for 5min
      - Memory > 85% for 5min  
      - Disk > 90%
      - Health check failures
    """)


def render_prometheus_metrics():
    """Render Prometheus metrics export."""
    st.markdown('<div class="sub-header">Prometheus Metrics Export</div>', unsafe_allow_html=True)
    
    st.markdown("""
    The following metrics are available in Prometheus format for scraping:
    """)
    
    prometheus_output = get_prometheus_metrics()
    st.code(prometheus_output, language="text")
    
    st.markdown("---")
    st.markdown("""
    **Integration with Prometheus:**
    
    1. Add this endpoint to your `prometheus.yml`:
    ```yaml
    scrape_configs:
      - job_name: 'urbantransit-iq'
        static_configs:
          - targets: ['localhost:8501']
    ```
    
    2. Or run a dedicated metrics exporter:
    ```bash
    python -m src.monitoring  # Outputs metrics to stdout
    ```
    
    **Available Metrics:**
    - `urbantransit_cpu_percent` - CPU usage percentage
    - `urbantransit_memory_percent` - Memory usage percentage
    - `urbantransit_disk_percent` - Disk usage percentage
    - `urbantransit_uptime_seconds` - Application uptime
    - `urbantransit_health_status{component="..."}` - Component health (1=healthy, 0.5=degraded, 0=unhealthy)
    
    **Grafana Dashboard Recommendations:**
    - Resource utilization panels (CPU, Memory, Disk)
    - Health status table with color coding
    - Uptime and availability tracking
    - Alert rules for degraded/unhealthy components
    """)


if __name__ == "__main__":
    render()