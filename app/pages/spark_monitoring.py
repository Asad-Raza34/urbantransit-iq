"""Spark Job Monitoring Dashboard Page."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta


def render():
    st.markdown('<div class="main-header">⚡ Spark Job Monitoring</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background: #e7f3ff; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <strong>⚡ Spark Job Monitoring:</strong> Real-time and historical monitoring of Spark job execution.
    Shows job status, duration, stages, and task metrics from the JVM status tracker.
    <br><strong>Note:</strong> On Windows, only JVM-only operations (Spark SQL, LinearRegression) are reliable.
    Python worker bridge is unreliable - tree models (GBT, RF) cannot be scored via Spark MLlib.
    </div>
    """, unsafe_allow_html=True)

    # Check Spark availability
    from src.spark_context import spark_available, spark_status, JobMonitor, spark_session
    
    ok, msg = spark_available()
    status = spark_status()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if ok:
            st.metric("Spark Status", "✅ Available")
        else:
            st.metric("Spark Status", "❌ Unavailable")
    with col2:
        st.metric("Master", status.get("master", "N/A"))
    with col3:
        st.metric("Java Home", "✅" if status.get("java_home") else "❌")
    with col4:
        st.metric("Hadoop Home", "✅" if status.get("hadoop_home") else "❌")

    if not ok:
        st.error(f"Spark unavailable: {msg}")
        st.info("Spark requires Java 17 and Hadoop winutils. Check DEPLOYMENT.md for setup.")
        return

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Job History", "🔍 Job Details", "⚙️ Spark Configuration", "⚠️ Limitations"])

    with tab1:
        render_job_history()

    with tab2:
        render_job_details()

    with tab3:
        render_spark_config(status)

    with tab4:
        render_limitations()


def render_job_history():
    """Render Spark job history from audit trail and status tracker."""
    st.markdown('<div class="sub-header">Recent Spark Jobs</div>', unsafe_allow_html=True)

    # Try to get jobs from status tracker
    try:
        from src.spark_context import spark_session, JobMonitor
        spark = spark_session()
        monitor = JobMonitor(spark)
        jobs = monitor.recent(limit=50)
        
        if jobs:
            # Convert to DataFrame for display
            job_data = []
            for job in jobs:
                job_data.append({
                    "Job ID": job["job_id"],
                    "Status": job["status"],
                    "Stages": job["num_stages"],
                    "Total Tasks": sum(s["tasks"] for s in job["stages"]),
                    "Completed": sum(s["completed"] for s in job["stages"]),
                    "Failed": sum(s["failed"] for s in job["stages"]),
                })
            
            df = pd.DataFrame(job_data)
            
            # Status color coding
            def status_color(status):
                colors = {
                    "SUCCEEDED": "🟢",
                    "FAILED": "🔴",
                    "RUNNING": "🟡",
                    "UNKNOWN": "⚪"
                }
                return colors.get(status, status)
            
            df["Status"] = df["Status"].apply(status_color)
            
            st.dataframe(df, use_container_width=True, height=400)
            
            # Job status pie chart
            if not df.empty:
                status_counts = df["Status"].apply(lambda x: x[1:] if x.startswith(("🟢", "🔴", "🟡", "⚪")) else x).value_counts().reset_index()
                status_counts.columns = ["Status", "Count"]
                
                fig = px.pie(
                    status_counts, values="Count", names="Status",
                    title="Job Status Distribution",
                    color="Status",
                    color_discrete_map={
                        "SUCCEEDED": "#28a745",
                        "FAILED": "#dc3545",
                        "RUNNING": "#ffc107",
                        "UNKNOWN": "#6c757d",
                    }
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No recent Spark jobs found in status tracker.")
    except Exception as e:
        st.warning(f"Could not retrieve job history from Spark status tracker: {e}")
        st.info("Job history requires an active Spark session with completed jobs.")

    # Also show audit trail for Spark jobs
    st.markdown('<div class="sub-header">Spark Jobs from Audit Trail</div>', unsafe_allow_html=True)
    try:
        from src.audit import timeline
        audit_df = timeline(limit=100, component="spark")
        if not audit_df.empty:
            # Filter for Spark-related actions
            spark_audit = audit_df[audit_df["action"].str.contains("spark", case=False, na=False)]
            if not spark_audit.empty:
                display_cols = ["ts", "action", "status", "details", "duration_ms"]
                available = [c for c in display_cols if c in spark_audit.columns]
                df_display = spark_audit[available].head(50).copy()
                if "duration_ms" in df_display.columns:
                    df_display["duration_ms"] = df_display["duration_ms"].round(0).astype(int)
                st.dataframe(df_display, use_container_width=True, height=300)
            else:
                st.info("No Spark job audit entries found.")
        else:
            st.info("No audit trail data available.")
    except Exception as e:
        st.warning(f"Could not load audit trail: {e}")


def render_job_details():
    """Render detailed job information."""
    st.markdown('<div class="sub-header">Job Detail View</div>', unsafe_allow_html=True)
    
    try:
        from src.spark_context import spark_session, JobMonitor
        spark = spark_session()
        monitor = JobMonitor(spark)
        jobs = monitor.recent(limit=50)
        
        if jobs:
            job_ids = [str(job["job_id"]) for job in jobs]
            selected = st.selectbox("Select Job ID", job_ids, key="spark_job_select")
            
            if st.button("Show Job Details", key="show_job_detail"):
                selected_job = next((j for j in jobs if str(j["job_id"]) == selected), None)
                if selected_job:
                    st.markdown(f"### Job {selected_job['job_id']} Details")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Job ID", selected_job["job_id"])
                    with col2:
                        st.metric("Status", selected_job["status"])
                    with col3:
                        st.metric("Number of Stages", selected_job["num_stages"])
                    
                    # Stage details
                    st.markdown("#### Stages")
                    if selected_job["stages"]:
                        stage_data = []
                        for stage in selected_job["stages"]:
                            stage_data.append({
                                "Stage ID": stage["stage_id"],
                                "Tasks": stage["tasks"],
                                "Completed": stage["completed"],
                                "Failed": stage["failed"],
                                "Active": stage["active"],
                                "Success Rate": f"{stage['completed']/stage['tasks']*100:.1f}%" if stage["tasks"] > 0 else "N/A"
                            })
                        stage_df = pd.DataFrame(stage_data)
                        st.dataframe(stage_df, use_container_width=True)
                        
                        # Stage visualization
                        fig = px.bar(
                            stage_df, x="Stage ID", y=["Completed", "Failed", "Active"],
                            title="Stage Task Status",
                            barmode="stack",
                            labels={"value": "Tasks", "variable": "Status"}
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No stage information available.")
        else:
            st.info("No jobs available for detail view.")
    except Exception as e:
        st.error(f"Could not load job details: {e}")


def render_spark_config(status: dict):
    """Render Spark configuration details."""
    st.markdown('<div class="sub-header">Spark Configuration</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Runtime Configuration**")
        config_items = [
            ("Master", status.get("master", "N/A")),
            ("Java Home", status.get("java_home", "N/A")),
            ("Hadoop Home", status.get("hadoop_home", "N/A")),
        ]
        for key, value in config_items:
            st.text(f"{key}: {value}")
    
    with col2:
        st.markdown("**Key Settings**")
        from config import settings
        settings_items = [
            ("Driver Memory", settings.SPARK_DRIVER_MEMORY),
            ("Shuffle Partitions", str(settings.SPARK_SHUFFLE_PARTITIONS)),
            ("App Name", settings.SPARK_APP_NAME),
        ]
        for key, value in settings_items:
            st.text(f"{key}: {value}")

    st.markdown("---")
    st.markdown("**Environment Variables**")
    import os
    env_vars = [
        "JAVA_HOME",
        "HADOOP_HOME",
        "UTIQ_HADOOP_MODE",
        "UTIQ_HADOOP_NAMENODE",
        "UTIQ_SPARK_MASTER",
    ]
    for var in env_vars:
        value = os.environ.get(var, "Not set")
        st.text(f"{var}: {value}")


def render_limitations():
    """Render Spark limitations on Windows."""
    st.markdown('<div class="sub-header">Known Limitations on Windows</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Python Worker Bridge Unreliability
    
    The PySpark Python worker bridge is **unreliable on Windows** due to:
    - `pyspark.daemon` imports POSIX-only signals that crash on Windows
    - Worker processes fail to connect back to the driver
    - Timeouts and connection failures during model scoring
    
    ### What Works Reliably (JVM-Only Operations)
    
    ✅ **Spark SQL** - All SQL queries, aggregations, joins
    ✅ **DataFrame operations** - filter, select, groupBy, agg (no UDFs)
    ✅ **Parquet read/write** - Using `make_spark_safe()` for timestamp conversion
    ✅ **LinearRegression (MLlib)** - Pure JVM transform, no Python worker needed
    ✅ **Spark job monitoring** - Status tracker runs on JVM
    
    ### What Does NOT Work Reliably
    
    ❌ **Tree-based MLlib models** - GBT, RandomForest, DecisionTree (require Python worker for scoring)
    ❌ **Python UDFs** - Any `.udf` or `pandas_udf` usage
    ❌ **`toPandas()` on large DataFrames** - Triggers worker communication
    ❌ **Model persistence/loading** for tree models - Requires Python worker
    
    ### Workarounds Implemented
    
    1. **JVM-Only Policy** - All Spark MLlib code uses LinearRegression only
    2. **Python ML** - scikit-learn RandomForest used for tree-based models
    3. **Dual-Pipeline Parity** - Both engines trained on identical data/split for comparison
    4. **Spark-Safe Parquet** - Timestamps converted to epoch microseconds via `make_spark_safe()`
    4. **Worker Reuse Disabled** - `spark.python.worker.reuse=false` in config
    
    ### Platform Constraint (Not a Bug)
    
    This is a **known platform constraint**, not a code bug. The PySpark team documents
    Windows as having limited support for the Python worker bridge. For production
    deployments requiring Spark MLlib tree models, use Linux-based infrastructure.
    """)


if __name__ == "__main__":
    render()