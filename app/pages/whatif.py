"""What-If Scenario Simulator Dashboard Page."""

import json

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.simulator.whatif import WhatIfSimulator, Scenario, run_scenario, SCENARIO_TEMPLATES, get_overcrowded_routes
from src.services.data_access import load_route_metrics, calculate_kpis


def render():
    st.markdown('<div class="main-header">🔮 What-If Scenario Simulator</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background: #e7f3ff; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <strong>ℹ️ How it works:</strong> Adjust the scenario parameters below to simulate the impact of operational changes.
    The simulator uses actual route metrics as baseline and applies simplified elasticity models to estimate outcomes.
    <strong>Results are scenario calculations, not predictions.</strong> Clearly distinguish between observed historical values (baseline) and scenario outputs.
    </div>
    """, unsafe_allow_html=True)

    # Load baseline
    sim = WhatIfSimulator()
    sim.load_baseline()
    kpis = calculate_kpis()

    # Show baseline KPIs
    st.markdown('<div class="sub-header">Current Baseline (Historical)</div>', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Daily Passengers", f"{kpis['total_passengers_daily']:,.0f}")
    with col2:
        st.metric("Avg Delay", f"{kpis['avg_delay_min']:.1f} min")
    with col3:
        st.metric("On-Time %", f"{kpis['on_time_pct']:.1f}%")
    with col4:
        st.metric("Avg Occupancy", f"{kpis['avg_occupancy_pct']:.1f}%")
    with col5:
        st.metric("Overcrowding Events", kpis['overcrowding_events'])

    st.markdown("---")

    # Scenario selection
    tab1, tab2 = st.tabs(["📋 Predefined Scenarios", "🎛️ Custom Scenario"])

    with tab1:
        render_predefined_scenarios(sim)

    with tab2:
        render_custom_scenario(sim)


def render_predefined_scenarios(sim: WhatIfSimulator):
    """Render predefined scenario templates."""
    st.markdown('<div class="sub-header">Predefined Scenarios</div>', unsafe_allow_html=True)

    scenario_names = list(SCENARIO_TEMPLATES.keys())
    selected = st.multiselect(
        "Select Scenarios to Run",
        scenario_names,
        default=["demand_increase_10", "capacity_increase_10"],
        # ``.get`` keeps a stale session-state value from raising KeyError.
        format_func=lambda x: SCENARIO_TEMPLATES[x].name if x in SCENARIO_TEMPLATES else str(x),
        key="predefined_scenarios_select",
    )

    if st.button("▶️ Run Selected Scenarios", type="primary", key="run_predefined"):
        if not selected:
            st.warning("Select at least one scenario.")
            return

        results = []
        for name in [n for n in selected if n in SCENARIO_TEMPLATES]:
            with st.spinner(f"Running {SCENARIO_TEMPLATES[name].name}..."):
                # Special handling for route-specific scenario
                if name == "route_specific_overcrowded":
                    overcrowded = get_overcrowded_routes()
                    if not overcrowded:
                        st.warning("No overcrowded routes found.")
                        continue
                    result = run_scenario(name, affected_routes=overcrowded)
                else:
                    result = run_scenario(name)
                results.append(result)

        if results:
            st.session_state["whatif_results_predefined"] = results
            st.success(f"Completed {len(results)} scenarios")
            st.rerun()

    # Display results. Each tab owns its own result key and key prefix: both tabs
    # render in the same script run, so a single shared key put two identical
    # figures on the page and raised StreamlitDuplicateElementId.
    if st.session_state.get("whatif_results_predefined"):
        render_scenario_results(
            st.session_state["whatif_results_predefined"], key_prefix="predefined"
        )


def render_custom_scenario(sim: WhatIfSimulator):
    """Render custom scenario builder."""
    st.markdown('<div class="sub-header">Build Custom Scenario</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Demand & Capacity**")
        demand_mult = st.slider("Demand Multiplier", 0.5, 2.0, 1.0, 0.05,
                               help="1.0 = current, 1.2 = 20% increase")
        capacity_mult = st.slider("Capacity Multiplier", 0.5, 2.0, 1.0, 0.05,
                                 help="1.0 = current, 1.1 = 10% more capacity")
        vehicle_mult = st.slider("Vehicle Multiplier", 0.5, 2.0, 1.0, 0.05,
                                help="1.0 = current, 1.15 = 15% more vehicles")
        frequency_mult = st.slider("Frequency Multiplier", 0.5, 2.0, 1.0, 0.05,
                                  help="1.0 = current, 1.1 = 10% more frequency")

    with col2:
        st.markdown("**Delay & Peak**")
        delay_mult = st.slider("Delay Multiplier", 0.5, 2.0, 1.0, 0.05,
                              help="1.0 = current, 1.2 = 20% more delay")
        peak_mult = st.slider("Peak Demand Multiplier", 0.5, 2.0, 1.0, 0.05,
                             help="Extra demand during peak periods")
        event_mult = st.slider("Event Day Demand Multiplier", 0.5, 3.0, 1.0, 0.1,
                              help="Demand on special event days")

        st.markdown("**Scope**")
        route_m = load_route_metrics()
        all_routes = sorted(route_m["route_id"].unique().tolist())
        affected_routes = st.multiselect(
            "Affected Routes (empty = all)",
            all_routes,
            default=[],
            help="Leave empty to apply to all routes"
        )

        time_bands = st.multiselect(
            "Affected Time Bands",
            ["night", "morning peak", "midday", "evening peak"],
            default=[],
            help="Leave empty for all time bands"
        )

    scenario_name = st.text_input("Scenario Name", "Custom Scenario")
    scenario_desc = st.text_area("Description", "Custom what-if scenario")

    if st.button("▶️ Run Custom Scenario", type="primary", key="run_custom"):
        scenario = Scenario(
            scenario_id=f"scn-{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}",
            name=scenario_name,
            description=scenario_desc,
            demand_multiplier=demand_mult,
            capacity_multiplier=capacity_mult,
            vehicle_multiplier=vehicle_mult,
            frequency_multiplier=frequency_mult,
            delay_multiplier=delay_mult,
            peak_demand_multiplier=peak_mult,
            event_day_demand_multiplier=event_mult,
            affected_routes=affected_routes if affected_routes else None,
            affected_time_bands=time_bands if time_bands else None,
        )

        with st.spinner("Running simulation..."):
            result = sim.run(scenario)
            st.session_state["whatif_results_custom"] = [result]
            st.success("Scenario completed!")
            st.rerun()

    # Display results (own key + prefix, see render_predefined_scenarios).
    if st.session_state.get("whatif_results_custom"):
        render_scenario_results(
            st.session_state["whatif_results_custom"], key_prefix="custom"
        )


def render_scenario_results(results: list, key_prefix: str = "default"):
    """Render scenario comparison results.

    ``key_prefix`` gives every element an explicit, unique key. The predefined
    and custom tabs can both hold results during the same script run, and two
    charts built from identical inputs would otherwise share an auto-generated
    element ID and raise ``StreamlitDuplicateElementId``.
    """
    st.markdown('<div class="sub-header">Scenario Results</div>', unsafe_allow_html=True)

    if not results:
        return

    # Baseline comparison table
    st.markdown('<div class="sub-header">Key Metric Comparison</div>', unsafe_allow_html=True)

    # Build comparison DataFrame
    rows = []
    for r in results:
        row = {"Scenario": r.scenario_name}
        # Key metrics
        key_metrics = [
            "total_passengers_daily", "avg_delay_min", "avg_occupancy_avg_pct",
            "on_time_share", "overcrowding_events_est", "underutilized_routes",
            "avg_actual_travel_min"
        ]
        for m in key_metrics:
            base = r.baseline.get(m, 0)
            est = r.estimated.get(m, 0)
            delta = r.delta.get(m, 0)
            pct = r.delta_pct.get(m, 0)
            row[f"{m}_baseline"] = base
            row[f"{m}_estimated"] = est
            row[f"{m}_delta"] = delta
            row[f"{m}_pct"] = pct
        rows.append(row)

    comp_df = pd.DataFrame(rows)

    # Display with formatting
    display_cols = ["Scenario"]
    for m in key_metrics:
        display_cols.extend([f"{m}_baseline", f"{m}_estimated", f"{m}_delta", f"{m}_pct"])

    # Format for display
    fmt_df = comp_df[display_cols].copy()
    rename_map = {
        "total_passengers_daily_baseline": "Passengers (Base)",
        "total_passengers_daily_estimated": "Passengers (Est)",
        "total_passengers_daily_delta": "Δ Passengers",
        "total_passengers_daily_pct": "Δ% Passengers",
        "avg_delay_min_baseline": "Delay (Base)",
        "avg_delay_min_estimated": "Delay (Est)",
        "avg_delay_min_delta": "Δ Delay",
        "avg_delay_min_pct": "Δ% Delay",
        "avg_occupancy_avg_pct_baseline": "Occupancy (Base)",
        "avg_occupancy_avg_pct_estimated": "Occupancy (Est)",
        "avg_occupancy_avg_pct_delta": "Δ Occupancy",
        "avg_occupancy_avg_pct_pct": "Δ% Occupancy",
        "on_time_share_baseline": "On-Time (Base)",
        "on_time_share_estimated": "On-Time (Est)",
        "on_time_share_delta": "Δ On-Time",
        "on_time_share_pct": "Δ% On-Time",
        "overcrowding_events_est_baseline": "Overcrowding (Base)",
        "overcrowding_events_est_estimated": "Overcrowding (Est)",
        "overcrowding_events_est_delta": "Δ Overcrowding",
        "overcrowding_events_est_pct": "Δ% Overcrowding",
        "underutilized_routes_baseline": "Underutil (Base)",
        "underutilized_routes_estimated": "Underutil (Est)",
        "underutilized_routes_delta": "Δ Underutil",
        "underutilized_routes_pct": "Δ% Underutil",
        "avg_actual_travel_min_baseline": "Travel (Base)",
        "avg_actual_travel_min_estimated": "Travel (Est)",
        "avg_actual_travel_min_delta": "Δ Travel",
        "avg_actual_travel_min_pct": "Δ% Travel",
    }
    fmt_df = fmt_df.rename(columns=rename_map)

    # Format percentages
    pct_cols = [c for c in fmt_df.columns if c.startswith("Δ%")]
    for col in pct_cols:
        if col in fmt_df.columns:
            fmt_df[col] = fmt_df[col].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A")

    # Delta columns
    delta_cols = [c for c in fmt_df.columns if c.startswith("Δ ") and not c.startswith("Δ%")]
    for col in delta_cols:
        if col in fmt_df.columns:
            fmt_df[col] = fmt_df[col].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "N/A")

    st.dataframe(fmt_df, use_container_width=True, key=f"whatif_compare_{key_prefix}")

    # Visual comparison
    st.markdown('<div class="sub-header">Visual Comparison</div>', unsafe_allow_html=True)

    # Bar chart for key deltas
    delta_metrics = {
        "Passengers": "total_passengers_daily_pct",
        "Delay": "avg_delay_min_pct",
        "Occupancy": "avg_occupancy_avg_pct_pct",
        "On-Time": "on_time_share_pct",
        "Overcrowding": "overcrowding_events_est_pct",
        "Underutilized": "underutilized_routes_pct",
    }

    # Prepare data for grouped bar chart
    plot_data = []
    for r in results:
        for label, key in delta_metrics.items():
            val = r.delta_pct.get(key.replace("_pct", ""), 0)
            plot_data.append({
                "Scenario": r.scenario_name,
                "Metric": label,
                "Delta %": val,
            })

    plot_df = pd.DataFrame(plot_data)

    fig = px.bar(
        plot_df, x="Metric", y="Delta %", color="Scenario",
        barmode="group",
        title="Percentage Change from Baseline by Metric",
        labels={"Delta %": "Change (%)"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True, key=f"whatif_delta_bar_{key_prefix}")

    # Detailed results per scenario
    st.markdown('<div class="sub-header">Detailed Results</div>', unsafe_allow_html=True)

    for r in results:
        with st.expander(f"📊 {r.scenario_name} - Details"):
            st.write(f"**Description:** {r.scenario_name}")
            st.write(f"**Assumptions:**")
            for k, v in r.assumptions.items():
                st.write(f"  - {k}: {v}")

            st.write("**Baseline vs Estimated:**")
            detail_rows = []
            for key in ["total_passengers_daily", "avg_delay_min", "avg_occupancy_avg_pct",
                        "on_time_share", "avg_actual_travel_min", "overcrowding_events_est",
                        "underutilized_routes", "avg_bunching_share", "avg_headway_min"]:
                base = r.baseline.get(key, 0)
                est = r.estimated.get(key, 0)
                delta = r.delta.get(key, 0)
                pct = r.delta_pct.get(key, 0)
                detail_rows.append({
                    "Metric": key.replace("_", " ").title(),
                    "Baseline": f"{base:,.2f}" if isinstance(base, float) else f"{base:,}",
                    "Estimated": f"{est:,.2f}" if isinstance(est, float) else f"{est:,}",
                    "Delta": f"{delta:+.2f}",
                    "Delta %": f"{pct:+.1f}%",
                })
            st.dataframe(
                pd.DataFrame(detail_rows),
                use_container_width=True,
                key=f"whatif_detail_{key_prefix}_{r.scenario_id}",
            )

    # Download results. The download button is rendered unconditionally once a
    # scenario run has been stored: nesting it inside the "Export" button branch
    # means it disappears on the rerun the download click itself triggers, so the
    # file never downloads.
    if st.button("📥 Export All Results (JSON)", key=f"whatif_export_btn_{key_prefix}"):
        st.session_state[f"whatif_export_{key_prefix}"] = [r.to_dict() for r in results]

    export_payload = st.session_state.get(f"whatif_export_{key_prefix}")
    if export_payload:
        st.download_button(
            "📥 Download",
            json.dumps(export_payload, indent=2, default=str),
            f"whatif_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json",
            key=f"whatif_download_{key_prefix}",
        )