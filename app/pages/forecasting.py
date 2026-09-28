"""Occupancy Forecasting Dashboard Page."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.forecasting.occupancy_forecast import (
    build_occupancy_models,
    train_occupancy_model,
    evaluate_occupancy_forecast,
    forecast_occupancy,
    load_saved_model_results,
)
from src.services.data_access import load_route_metrics


def render():
    st.markdown('<div class="main-header">📈 Occupancy Forecasting</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background: #e7f3ff; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <strong>📈 Occupancy Forecasting:</strong> Predicts future route-level maximum occupancy using historical
    occupancy patterns, demand features, and time-series lag features. Models: Random Forest, Gradient Boosting, Linear Regression.
    Now includes 95% prediction intervals based on residual quantiles.
    </div>
    """, unsafe_allow_html=True)

    # Model training section
    st.markdown('<div class="sub-header">Model Training</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.write("Train occupancy forecasting models on historical data.")
    with col2:
        if st.button("🔄 Train All Models", type="primary", key="train_forecast_btn"):
            with st.spinner("Training occupancy forecasting models (this may take a few minutes)..."):
                result = build_occupancy_models()
                st.session_state["forecast_results"] = result
                st.success("All models trained!")
                st.rerun()
    with col3:
        st.metric("Forecast Horizon", "28 days")

    # Load existing results from the per-model metrics written by the pipeline.
    if "forecast_results" not in st.session_state:
        saved_results = load_saved_model_results()
        if saved_results:
            st.session_state["forecast_results"] = saved_results

    # Display model results
    if "forecast_results" in st.session_state:
        results = st.session_state["forecast_results"]

        st.markdown('<div class="sub-header">Model Performance</div>', unsafe_allow_html=True)

        for model_name, result in results.items():
            if "error" in result:
                st.error(f"{model_name}: {result['error']}")
                continue

            with st.expander(f"{result['algorithm'].title()} (R² = {result.get('r2', 0):.4f})", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("MAE", f"{result.get('mae', 0):.3f}")
                with col2:
                    st.metric("RMSE", f"{result.get('rmse', 0):.3f}")
                with col3:
                    st.metric("MAPE %", f"{result.get('mape_pct', 0):.1f}%")
                with col4:
                    st.metric("R²", f"{result.get('r2', 0):.4f}")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Train Rows", f"{result.get('train_rows', 0):,}")
                with col2:
                    st.metric("Test Rows", f"{result.get('test_rows', 0):,}")

                # Show interval info
                if "residual_std" in result:
                    st.info(f"Prediction Interval (95%): ±{result['residual_q95']:.1f} / {result['residual_q5']:.1f} (residual std: {result['residual_std']:.2f})")

    st.markdown("---")

    # Forecast generation
    st.markdown('<div class="sub-header">Generate Forecast</div>', unsafe_allow_html=True)

    route_m = load_route_metrics()
    all_routes = sorted(route_m["route_id"].unique().tolist())

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_routes = st.multiselect(
            "Select Routes (empty = all)",
            all_routes,
            default=[],
            key="forecast_routes",
        )
    with col2:
        horizon = st.slider("Forecast Horizon (days)", 1, 28, 7, key="forecast_horizon")
    with col3:
        model_type = st.selectbox("Model", ["random_forest", "gradient_boosting", "linear"], key="forecast_model")

    if st.button("🔮 Generate Forecast", type="primary", key="generate_forecast_btn"):
        with st.spinner("Generating forecast..."):
            try:
                forecast = forecast_occupancy(
                    route_ids=selected_routes if selected_routes else None,
                    horizon_days=horizon,
                    model_type=model_type,
                )
                st.session_state["forecast_df"] = forecast
                st.success(f"Generated forecast for {len(forecast)} route-timeband combinations")
                st.rerun()
            except Exception as e:
                st.error(f"Forecast failed: {e}")

    # Display forecast
    if "forecast_df" in st.session_state:
        forecast = st.session_state["forecast_df"]

        st.markdown('<div class="sub-header">Forecast Results</div>', unsafe_allow_html=True)

        if not forecast.empty:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Forecasts", len(forecast))
            with col2:
                st.metric("Avg Predicted Occupancy", f"{forecast['predicted_occupancy_max_pct'].mean():.1f}%")
            with col3:
                st.metric("Max Predicted", f"{forecast['predicted_occupancy_max_pct'].max():.1f}%")
            with col4:
                overcrowded = (forecast['predicted_occupancy_max_pct'] >= 85).sum()
                st.metric("Projected Overcrowded", int(overcrowded))

            # Forecast table
            display_cols = ["route_id", "date", "time_band", "occupancy_max_pct",
                            "predicted_occupancy_max_pct", "lower_bound", "upper_bound", "horizon_days"]
            available = [c for c in display_cols if c in forecast.columns]

            df_display = forecast[available].copy()
            df_display["occupancy_max_pct"] = df_display["occupancy_max_pct"].round(1)
            if "predicted_occupancy_max_pct" in df_display.columns:
                df_display["predicted_occupancy_max_pct"] = df_display["predicted_occupancy_max_pct"].round(1)
            if "lower_bound" in df_display.columns:
                df_display["lower_bound"] = df_display["lower_bound"].round(1)
            if "upper_bound" in df_display.columns:
                df_display["upper_bound"] = df_display["upper_bound"].round(1)

            st.dataframe(df_display.sort_values("predicted_occupancy_max_pct", ascending=False),
                         use_container_width=True, height=400)

            # Visualization
            st.markdown('<div class="sub-header">Forecast Visualization</div>', unsafe_allow_html=True)

            # Scatter: current vs predicted with intervals
            fig = px.scatter(
                forecast, x="occupancy_max_pct", y="predicted_occupancy_max_pct",
                color="route_id", hover_data=["date", "time_band", "lower_bound", "upper_bound"],
                title="Current vs Predicted Occupancy (with 95% Prediction Intervals)",
                labels={"occupancy_max_pct": "Current Max Occupancy %",
                        "predicted_occupancy_max_pct": "Predicted Max Occupancy %"}
            )
            fig.add_trace(go.Scatter(
                x=[0, 150], y=[0, 150],
                mode="lines", line=dict(dash="dash", color="gray"),
                name="Perfect Prediction", showlegend=True
            ))
            fig.add_hline(y=85, line_dash="dash", line_color="red", annotation_text="High Crowding (85%)")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Route-level forecast with intervals
            if len(forecast["route_id"].unique()) <= 10:
                route_avg = forecast.groupby("route_id").agg({
                    "predicted_occupancy_max_pct": "mean",
                    "lower_bound": "mean",
                    "upper_bound": "mean"
                }).reset_index()
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=route_avg["route_id"], y=route_avg["predicted_occupancy_max_pct"],
                    name="Predicted", marker_color="#1f77b4"
                ))
                fig.add_trace(go.Scatter(
                    x=route_avg["route_id"], y=route_avg["upper_bound"],
                    mode="lines", line=dict(color="red", dash="dash"),
                    name="Upper Bound (97.5%)", showlegend=True
                ))
                fig.add_trace(go.Scatter(
                    x=route_avg["route_id"], y=route_avg["lower_bound"],
                    mode="lines", line=dict(color="red", dash="dash"),
                    name="Lower Bound (2.5%)", fill='tonexty', fillcolor='rgba(255,0,0,0.1)',
                    showlegend=True
                ))
                fig.add_hline(y=85, line_dash="dash", line_color="red")
                fig.update_layout(
                    title="Average Predicted Occupancy by Route with 95% Prediction Intervals",
                    yaxis_title="Predicted Occupancy %",
                    xaxis_title="Route",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

            # Download
            csv = forecast.to_csv(index=False)
            st.download_button(
                "📥 Download Forecast (CSV)",
                csv,
                f"occupancy_forecast_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                "text/csv",
            )

    # Model evaluation on unseen cases
    st.markdown('<div class="sub-header">Unseen Case Evaluation</div>', unsafe_allow_html=True)

    if st.button("📊 Evaluate on Unseen Cases", key="eval_unseen"):
        with st.spinner("Evaluating on 100 unseen cases..."):
            eval_result = evaluate_occupancy_forecast("random_forest")
            if "error" not in eval_result:
                st.json(eval_result)
            else:
                st.error(eval_result["error"])


if __name__ == "__main__":
    render()