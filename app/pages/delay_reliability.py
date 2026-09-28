"""Delay & Reliability Dashboard Page."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.services.data_access import load_route_metrics, load_crowding_analytics, load_demand_analytics
from src.comparison.routes import get_route_profile


def render():
    st.markdown('<div class="main-header">⏱️ Delay & Reliability</div>', unsafe_allow_html=True)

    route_m = load_route_metrics()
    crowding = load_crowding_analytics()
    quality_days = crowding.get("quality_days", pd.DataFrame())
    daily_demand = load_demand_analytics().get("daily_demand", pd.DataFrame())

    if route_m.empty:
        st.warning("No route metrics data available.")
        return

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Network On-Time %", f"{route_m['on_time_share'].mean()*100:.1f}%")
    with col2:
        st.metric("Avg Delay", f"{route_m['avg_delay_min'].mean():.1f} min")
    with col3:
        st.metric("P90 Delay", f"{route_m['p90_delay_min'].mean():.1f} min")
    with col4:
        severe = route_m["severe_critical_share"].mean() * 100
        st.metric("Severe+Critical %", f"{severe:.1f}%")

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Delay Analysis", "🎯 Reliability", "📋 Schedule Adherence", "🚌 Route Detail", "🔮 Delay Prediction"])

    with tab1:
        render_delay_analysis(route_m, quality_days, daily_demand)

    with tab2:
        render_reliability(route_m, quality_days)

    with tab3:
        render_adherence(route_m)

    with tab4:
        render_route_detail(route_m)

    with tab5:
        render_delay_prediction(route_m)


def render_delay_analysis(route_m: pd.DataFrame, quality_days: pd.DataFrame, daily_demand: pd.DataFrame):
    """Render delay analysis charts."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="sub-header">Delay Distribution by Route</div>', unsafe_allow_html=True)
        fig = px.box(
            route_m, y="avg_delay_min", x="category",
            title="Average Delay by Route Category",
            labels={"avg_delay_min": "Avg Delay (min)", "category": "Category"},
            color="category"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="sub-header">Delay vs On-Time Scatter</div>', unsafe_allow_html=True)
        fig = px.scatter(
            route_m, x="avg_delay_min", y="on_time_share",
            size="passengers", color="category",
            hover_data=["route_id", "route_name", "route_class"],
            labels={"avg_delay_min": "Avg Delay (min)", "on_time_share": "On-Time Share"},
            title="Delay vs Reliability"
        )
        fig.add_hline(y=0.75, line_dash="dash", annotation_text="Degraded Threshold")
        fig.add_hline(y=0.55, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Delay severity distribution
    st.markdown('<div class="sub-header">Delay Severity Distribution</div>', unsafe_allow_html=True)
    sev_cols = ["on_time_share", "adherence_share"]
    # We need delay_severity breakdown - use route_metrics
    # Create severity categories from route_m
    sev_data = route_m[["route_id", "route_name", "category", "avg_delay_min",
                         "on_time_share", "adherence_share", "severe_critical_share"]].copy()

    # Categorize routes by severity profile
    def categorize_severity(row):
        if row["severe_critical_share"] > 0.3:
            return "Critical"
        elif row["severe_critical_share"] > 0.15:
            return "High"
        elif row["avg_delay_min"] > 10:
            return "Moderate"
        elif row["on_time_share"] < 0.75:
            return "Degraded"
        else:
            return "Good"

    sev_data["severity_profile"] = sev_data.apply(categorize_severity, axis=1)
    sev_counts = sev_data["severity_profile"].value_counts().reset_index()
    sev_counts.columns = ["Severity Profile", "Count"]

    fig = px.pie(
        sev_counts, values="Count", names="Severity Profile",
        title="Routes by Delay Severity Profile",
        color="Severity Profile",
        color_discrete_map={
            "Good": "#28a745",
            "Degraded": "#ffc107",
            "Moderate": "#fd7e14",
            "High": "#dc3545",
            "Critical": "#8b0000",
        }
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Daily delay trend (city-wide)
    if not quality_days.empty:
        st.markdown('<div class="sub-header">Daily Quality Trend (City-Wide)</div>', unsafe_allow_html=True)
        daily_quality = quality_days.groupby("date").agg(
            avg_quality=("quality_score", "mean"),
            avg_on_time=("on_time_share", "mean"),
            avg_delay=("avg_delay_min", "mean"),
        ).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_quality["date"], y=daily_quality["avg_quality"],
            name="Quality Score", line=dict(color="#1f77b4")
        ))
        fig.add_trace(go.Scatter(
            x=daily_quality["date"], y=daily_quality["avg_on_time"]*100,
            name="On-Time %", line=dict(color="#28a745"), yaxis="y2"
        ))
        fig.add_trace(go.Scatter(
            x=daily_quality["date"], y=daily_quality["avg_delay"],
            name="Avg Delay (min)", line=dict(color="#dc3545"), yaxis="y3"
        ))

        fig.update_layout(
            title="City-Wide Daily Quality Trends",
            height=500,
            yaxis=dict(title="Quality Score", side="left"),
            yaxis2=dict(title="On-Time %", side="right", overlaying="y"),
            yaxis3=dict(title="Avg Delay (min)", side="right", overlaying="y", anchor="free", position=0.95),
        )
        st.plotly_chart(fig, use_container_width=True)


def render_reliability(route_m: pd.DataFrame, quality_days: pd.DataFrame):
    """Render reliability analysis."""
    st.markdown('<div class="sub-header">Route Reliability Classification</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Reliability classification pie
        rel_counts = route_m["route_class"].value_counts().reset_index()
        rel_counts.columns = ["Classification", "Count"]

        fig = px.pie(
            rel_counts, values="Count", names="Classification",
            title="Routes by Reliability Classification",
            color="Classification",
            color_discrete_map={
                "Top Performer": "#28a745",
                "Solid": "#1f77b4",
                "Needs Attention": "#ffc107",
                "Action Required": "#dc3545",
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # On-time share by category
        fig = px.box(
            route_m, x="category", y="on_time_share",
            title="On-Time Share by Route Category",
            labels={"on_time_share": "On-Time Share", "category": "Category"},
            color="category"
        )
        fig.add_hline(y=0.75, line_dash="dash", annotation_text="Degraded Threshold")
        fig.add_hline(y=0.55, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Quality days heatmap
    if not quality_days.empty:
        st.markdown('<div class="sub-header">Daily Quality Status Heatmap</div>', unsafe_allow_html=True)

        # Sample top routes for readability
        top_routes = route_m.nlargest(20, "route_score")["route_id"].tolist()
        qd_sample = quality_days[quality_days["route_id"].isin(top_routes)]

        if not qd_sample.empty:
            # Pivot for heatmap
            pivot = qd_sample.pivot_table(
                index="route_id", columns="date", values="quality_score", aggfunc="mean"
            ).fillna(0)

            fig = px.imshow(
                pivot, aspect="auto",
                title="Daily Quality Score by Route (Top 20)",
                labels=dict(x="Date", y="Route", color="Quality Score"),
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=50,
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

    # Reliability table
    st.markdown('<div class="sub-header">Route Reliability Details</div>', unsafe_allow_html=True)

    rel_cols = ["route_id", "route_name", "category", "route_class", "route_score",
                "on_time_share", "adherence_share", "avg_delay_min", "severe_critical_share",
                "bunching_share", "headway_cv"]
    available = [c for c in rel_cols if c in route_m.columns]

    df = route_m[available].copy()
    if "on_time_share" in df.columns:
        df["on_time_share"] = (df["on_time_share"] * 100).round(1)
    if "adherence_share" in df.columns:
        df["adherence_share"] = (df["adherence_share"] * 100).round(1)
    if "severe_critical_share" in df.columns:
        df["severe_critical_share"] = (df["severe_critical_share"] * 100).round(1)
    if "bunching_share" in df.columns:
        df["bunching_share"] = (df["bunching_share"] * 100).round(1)

    st.dataframe(df.sort_values("route_score"), use_container_width=True, height=500)


def render_adherence(route_m: pd.DataFrame):
    """Render schedule adherence analysis."""
    st.markdown('<div class="sub-header">Schedule Adherence Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Adherence distribution
        fig = px.histogram(
            route_m, x="adherence_share", nbins=20,
            title="Schedule Adherence Distribution (±2 min)",
            labels={"adherence_share": "Adherence Share", "count": "Routes"},
            color_discrete_sequence=["#1f77b4"]
        )
        fig.add_vline(x=0.8, line_dash="dash", annotation_text="Good Target (80%)")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Adherence vs Delay
        fig = px.scatter(
            route_m, x="adherence_share", y="avg_delay_min",
            size="passengers", color="category",
            hover_data=["route_id", "route_name"],
            labels={"adherence_share": "Adherence (±2 min)", "avg_delay_min": "Avg Delay (min)"},
            title="Adherence vs Delay"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Headway analysis
    st.markdown('<div class="sub-header">Headway Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            route_m, x="avg_headway_min", y="headway_cv",
            size="passengers", color="category",
            hover_data=["route_id", "route_name", "bunching_share"],
            labels={"avg_headway_min": "Avg Headway (min)", "headway_cv": "Headway CV"},
            title="Headway vs Variability"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Bunching vs Headway
        fig = px.scatter(
            route_m, x="avg_headway_min", y="bunching_share",
            size="passengers", color="category",
            hover_data=["route_id", "route_name", "gapping_share"],
            labels={"avg_headway_min": "Avg Headway (min)", "bunching_share": "Bunching Share"},
            title="Headway vs Bunching"
        )
        fig.add_vline(x=5, line_dash="dash", annotation_text="5-min threshold")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Detailed headway table
    st.markdown('<div class="sub-header">Headway Details</div>', unsafe_allow_html=True)

    hw_cols = ["route_id", "route_name", "category", "avg_headway_min", "headway_std_min",
               "headway_cv", "bunching_share", "gapping_share", "on_time_share"]
    available = [c for c in hw_cols if c in route_m.columns]

    df = route_m[available].copy()
    if "on_time_share" in df.columns:
        df["on_time_share"] = (df["on_time_share"] * 100).round(1)
    if "bunching_share" in df.columns:
        df["bunching_share"] = (df["bunching_share"] * 100).round(1)
    if "gapping_share" in df.columns:
        df["gapping_share"] = (df["gapping_share"] * 100).round(1)

    st.dataframe(df.sort_values("bunching_share", ascending=False), use_container_width=True, height=400)


def render_route_detail(route_m: pd.DataFrame):
    """Render detailed route delay/reliability profile."""
    from src.comparison.routes import get_route_profile

    route_ids = sorted(route_m["route_id"].unique().tolist())
    selected = st.selectbox("Select Route", route_ids, key="delay_route_detail_select")

    if st.button("Load Route Detail", key="load_delay_detail"):
        profile = get_route_profile(selected)
        if "error" in profile:
            st.error(profile["error"])
        else:
            st.markdown(f'<div class="main-header">Route Detail: {profile["route_id"]}</div>', unsafe_allow_html=True)

            basic = profile["basic_info"]
            perf = profile["performance_summary"]

            # KPIs
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Delay", f"{perf['avg_delay_min']:.1f} min")
            with col2:
                st.metric("P90 Delay", f"{basic.get('p90_delay_min', 0):.1f} min")
            with col3:
                st.metric("On-Time %", f"{perf['on_time_pct']:.1f}%")
            with col4:
                st.metric("Adherence %", f"{perf.get('adherence_share', 0)*100:.1f}%")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Route Score", f"{perf['route_score']:.1f}")
            with col2:
                st.metric("Route Class", perf["route_class"])
            with col3:
                st.metric("Severe+Critical %", f"{basic.get('severe_critical_share', 0)*100:.1f}%")
            with col4:
                st.metric("Headway CV", f"{basic.get('headway_cv', 0):.3f}")

            # Delay severity
            st.markdown('<div class="sub-header">Delay Severity Breakdown</div>', unsafe_allow_html=True)
            sev = basic.get("severe_critical_share", 0) * 100
            st.write(f"**Severe + Critical Delay Share:** {sev:.1f}%")

            # Headway details
            st.markdown('<div class="sub-header">Headway Details</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Headway", f"{basic.get('avg_headway_min', 0):.1f} min")
            with col2:
                st.metric("Headway Std Dev", f"{basic.get('headway_std_min', 0):.1f} min")
            with col3:
                st.metric("Headway CV", f"{basic.get('headway_cv', 0):.3f}")

            # Bunching
            st.metric("Bunching Share", f"{perf['bunching_share']:.1f}%")
            st.metric("Gapping Share", f"{basic.get('gapping_share', 0)*100:.1f}%")

            if st.button("← Back"):
                st.rerun()


def render_delay_prediction(route_m: pd.DataFrame):
    """Render delay prediction analysis."""
    st.markdown('<div class="main-header">🔮 Delay Prediction</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background: rgb(0 0 0); padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <strong>🔮 Delay Prediction:</strong> Predicts route-level average departure delay using historical delay patterns,
    headway, occupancy, and demand features. Models: Random Forest, Gradient Boosting, Linear Regression.
    </div>
    """, unsafe_allow_html=True)

    # Model training section
    st.markdown('<div class="sub-header">Model Training</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.write("Train delay prediction models on historical trip data.")
    with col2:
        if st.button("🔄 Train All Models", type="primary", key="train_delay_btn"):
            with st.spinner("Training delay prediction models (this may take a few minutes)..."):
                try:
                    from src.ml.delay_prediction import build_delay_models
                    result = build_delay_models()
                    st.session_state["delay_results"] = result
                    st.success("All models trained!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {e}")
    with col3:
        st.metric("Forecast Horizon", "28 days")

    # Load existing results for every model. The loader returns the nested
    # ``{model_type: {metric: value}}`` shape this section renders; reading one
    # flat metrics file gave the loop bare floats and raised TypeError.
    if "delay_results" not in st.session_state:
        from src.ml.delay_prediction import load_saved_model_results

        saved_results = load_saved_model_results()
        if saved_results:
            st.session_state["delay_results"] = saved_results

    # Display model results
    if "delay_results" in st.session_state:
        results = st.session_state["delay_results"]

        st.markdown('<div class="sub-header">Model Performance</div>', unsafe_allow_html=True)

        for model_name, result in results.items():
            if "error" in result:
                st.error(f"{model_name}: {result['error']}")
                continue

            with st.expander(f"{result['algorithm'].title()} (R² = {result.get('r2', 0):.4f})", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("MAE", f"{result.get('mae', 0):.3f} min")
                with col2:
                    st.metric("RMSE", f"{result.get('rmse', 0):.3f} min")
                with col3:
                    st.metric("MAPE %", f"{result.get('mape_pct', 0):.1f}%")
                with col4:
                    st.metric("R²", f"{result.get('r2', 0):.4f}")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Train Rows", f"{result.get('train_rows', 0):,}")
                with col2:
                    st.metric("Test Rows", f"{result.get('test_rows', 0):,}")

    st.markdown("---")

    # Delay prediction generation
    st.markdown('<div class="sub-header">Generate Delay Prediction</div>', unsafe_allow_html=True)

    all_routes = sorted(route_m["route_id"].unique().tolist())

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_routes = st.multiselect(
            "Select Routes (empty = all)",
            all_routes,
            default=[],
            key="delay_pred_routes",
        )
    with col2:
        horizon = st.slider("Prediction Horizon (days)", 1, 28, 7, key="delay_pred_horizon")
    with col3:
        model_type = st.selectbox("Model", ["random_forest", "gradient_boosting", "linear"], key="delay_pred_model")

    if st.button("🔮 Generate Prediction", type="primary", key="generate_delay_pred_btn"):
        with st.spinner("Generating delay prediction..."):
            try:
                from src.ml.delay_prediction import predict_delay
                pred = predict_delay(
                    route_ids=selected_routes if selected_routes else None,
                    date=None,
                    model_type=model_type,
                )
                st.session_state["delay_pred_df"] = pred
                st.success(
                    f"Generated predictions for {len(pred)} route/date/time-band rows"
                )
                st.rerun()
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # Display predictions
    if "delay_pred_df" in st.session_state:
        pred = st.session_state["delay_pred_df"]

        st.markdown('<div class="sub-header">Prediction Results</div>', unsafe_allow_html=True)

        if not pred.empty:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Predictions", len(pred))
            with col2:
                st.metric("Avg Predicted Delay", f"{pred['predicted_delay_min'].mean():.1f} min")
            with col3:
                st.metric("Max Predicted Delay", f"{pred['predicted_delay_min'].max():.1f} min")
            with col4:
                high_delay = (pred['predicted_delay_min'] >= 15).sum()
                st.metric("High Delay Predicted (>15 min)", int(high_delay))

            # Prediction table
            # Grain is route/date/time_band (the model's training grain), so the
            # table shows the observed average delay for that slot against the
            # prediction. The old trip-level names silently dropped every
            # identity column instead of failing loudly.
            display_cols = ["route_id", "date", "time_band", "route_avg_delay",
                            "predicted_delay_min"]
            available = [c for c in display_cols if c in pred.columns]

            df_display = pred[available].copy()
            if "route_avg_delay" in df_display.columns:
                df_display["route_avg_delay"] = df_display["route_avg_delay"].round(1)
            if "predicted_delay_min" in df_display.columns:
                df_display["predicted_delay_min"] = df_display["predicted_delay_min"].round(1)

            st.dataframe(df_display.sort_values("predicted_delay_min", ascending=False),
                         use_container_width=True, height=400)

            # Visualization
            st.markdown('<div class="sub-header">Prediction Visualization</div>', unsafe_allow_html=True)

            # Scatter: actual vs predicted (if actual available)
            if "route_avg_delay" in pred.columns and pred["route_avg_delay"].notna().any():
                fig = px.scatter(
                    pred.dropna(subset=["route_avg_delay"]),
                    x="route_avg_delay", y="predicted_delay_min",
                    color="route_id", hover_data=["date", "time_band"],
                    title="Actual vs Predicted Delay",
                    labels={"route_avg_delay": "Actual Delay (min)",
                            "predicted_delay_min": "Predicted Delay (min)"}
                )
                fig.add_trace(go.Scatter(
                    x=[0, 60], y=[0, 60],
                    mode="lines", line=dict(dash="dash", color="gray"),
                    name="Perfect Prediction", showlegend=True
                ))
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

            # Route-level prediction
            if len(pred["route_id"].unique()) <= 10:
                fig = px.bar(
                    pred.groupby("route_id")["predicted_delay_min"].mean().reset_index(),
                    x="route_id", y="predicted_delay_min",
                    title="Average Predicted Delay by Route",
                    labels={"predicted_delay_min": "Predicted Delay (min)", "route_id": "Route"}
                )
                fig.add_hline(y=15, line_dash="dash", line_color="red", annotation_text="High Delay (15 min)")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            # Download
            csv = pred.to_csv(index=False)
            st.download_button(
                "📥 Download Prediction (CSV)",
                csv,
                f"delay_prediction_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                "text/csv",
            )

    # Model evaluation on unseen cases
    st.markdown('<div class="sub-header">Unseen Case Evaluation</div>', unsafe_allow_html=True)

    if st.button("📊 Evaluate on Unseen Cases", key="eval_delay_unseen"):
        with st.spinner("Evaluating on test set..."):
            from src.ml.delay_prediction import evaluate_delay_model
            eval_result = evaluate_delay_model("random_forest")
            if "error" not in eval_result:
                st.json(eval_result)
            else:
                st.error(eval_result["error"])