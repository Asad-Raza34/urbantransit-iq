"""Route Clustering Dashboard Page."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.clustering.route_clustering import (
    build_clustering,
    load_clustering_results,
    load_route_clusters,
    load_cluster_profiles,
    load_cluster_centers,
    get_route_cluster,
    get_cluster_routes,
)
from src.services.data_access import load_route_metrics


def render():
    st.markdown('<div class="main-header">🔬 Route Clustering</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background: #e7f3ff; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <strong>🔬 Route Clustering:</strong> Groups routes into operational clusters using K-Means on 17+ features
    including demand, delay, occupancy, crowding, reliability, and headway metrics.
    Cluster count selected automatically via silhouette analysis.
    </div>
    """, unsafe_allow_html=True)

    # Build/refresh button
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown('<div class="sub-header">Route Clusters</div>', unsafe_allow_html=True)
    with col2:
        if st.button("🔄 Rebuild Clusters", type="primary"):
            with st.spinner("Running K-Means clustering with silhouette optimization..."):
                result = build_clustering()
                st.success(f"Generated {result['n_clusters']} clusters (silhouette: {result['silhouette_score']:.3f})")
                st.rerun()
    with col3:
        st.metric("Total Routes", 100)

    # Load clustering results
    try:
        results = load_clustering_results()
        n_clusters = results.get("n_clusters", 0)
        cluster_labels = results.get("cluster_labels", {})
        profiles_df = pd.DataFrame(results.get("profiles", []))
        assignments = pd.DataFrame(results.get("route_assignments", []))
    except Exception as e:
        st.warning(f"No clustering results found: {e}. Click 'Rebuild Clusters' to generate.")
        return

    if n_clusters == 0:
        st.info("No clusters generated yet. Click 'Rebuild Clusters' to run clustering.")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Clusters", n_clusters)
    with col2:
        st.metric("Silhouette Score", f"{results.get('silhouette_score', 0):.3f}")
    with col3:
        st.metric("Inertia", f"{results.get('inertia', 0):.1f}")
    with col3:
        st.metric("Routes Clustered", len(assignments))

    st.markdown("---")

    # Cluster summary
    st.markdown('<div class="sub-header">Cluster Summary</div>', unsafe_allow_html=True)

    profiles = pd.DataFrame(load_clustering_results().get("profiles", []))
    if not profiles.empty:
        for _, profile in profiles.iterrows():
            cid = int(profile["cluster_id"])
            label = profile.get("label", f"Cluster {cid}")
            n_routes = int(profile["n_routes"])

            with st.expander(f"{label} ({n_routes} routes)", expanded=False):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Avg Passengers", f"{profile.get('passengers_mean', 0):,.0f}")
                    st.metric("Avg Delay", f"{profile.get('avg_delay_min_mean', 0):.1f} min")
                    st.metric("On-Time %", f"{profile.get('on_time_share_mean', 0)*100:.1f}%")

                with col2:
                    st.metric("Avg Occupancy", f"{profile.get('occupancy_avg_pct_mean', 0):.1f}%")
                    st.metric("Max Occupancy", f"{profile.get('occupancy_max_pct_mean', 0):.1f}%")
                    st.metric("Crowding %", f"{profile.get('crowding_share_mean', 0)*100:.1f}%")

                with col3:
                    st.metric("Bunching %", f"{profile.get('bunching_share_mean', 0)*100:.1f}%")
                    st.metric("Underutilized %", f"{profile.get('underutilized_share_mean', 0)*100:.1f}%")
                    st.metric("Route Score", f"{profile.get('route_score_mean', 0):.1f}")

                # Show routes in cluster
                cluster_routes = [r for r in load_route_clusters().to_dict("records") if r["cluster"] == int(profile["cluster_id"])]
                if cluster_routes:
                    st.write("**Routes:**")
                    route_ids = [r["route_id"] for r in cluster_routes]
                    st.write(", ".join(sorted(route_ids)))

    st.markdown("---")

    # Cluster comparison chart
    st.markdown('<div class="sub-header">Cluster Feature Comparison</div>', unsafe_allow_html=True)

    profiles = pd.DataFrame(load_clustering_results().get("profiles", []))
    if not profiles.empty:
        # Radar chart comparison
        key_metrics = [
            "passengers_mean", "avg_delay_min_mean", "on_time_share_mean",
            "occupancy_avg_pct_mean", "bunching_share_mean", "crowding_share_mean",
            "route_score_mean", "avg_boardings_per_trip_mean", "avg_headway_min_mean"
        ]

        available = [m for m in key_metrics if m in profiles.columns]

        fig = go.Figure()
        for _, row in profiles.iterrows():
            values = [row.get(m, 0) for m in available]
            # Normalize
            normed = []
            for i, m in enumerate(available):
                col_vals = profiles[m].dropna()
                if len(col_vals) > 1:
                    mn, mx = col_vals.min(), col_vals.max()
                    if mx > mn:
                        normed.append((row[m] - mn) / (mx - mn))
                    else:
                        normed.append(0.5)
                else:
                    normed.append(0.5)

            normed.append(normed[0])
            labels = [m.replace("_mean", "").replace("_", " ").title() for m in available]
            labels.append(labels[0])

            fig.add_trace(go.Scatterpolar(
                r=normed,
                theta=labels,
                fill='toself',
                name=f"Cluster {int(row['cluster_id'])}: {row.get('label', '')[:30]}",
                opacity=0.7,
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Normalized Cluster Profiles (0=worst, 1=best)",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Route table with cluster assignments
    st.markdown('<div class="sub-header">Route Cluster Assignments</div>', unsafe_allow_html=True)

    assignments = load_route_clusters()
    if not assignments.empty:
        route_m = load_route_metrics()
        merged = assignments.merge(
            route_m[["route_id", "route_name", "category", "route_score", "route_class",
                     "avg_delay_min", "on_time_share", "occupancy_avg_pct",
                     "crowding_share", "bunching_share"]],
            on="route_id", how="left"
        )

        merged["cluster_label"] = merged["cluster"].map(
            load_clustering_results().get("cluster_labels", {})
        )

        display_cols = ["route_id", "route_name", "category", "cluster", "cluster_label",
                        "route_score", "avg_delay_min", "on_time_share",
                        "occupancy_avg_pct", "crowding_share", "bunching_share"]
        available = [c for c in display_cols if c in merged.columns]

        df_display = merged[available].copy()
        if "on_time_share" in df_display.columns:
            df_display["on_time_share"] = (df_display["on_time_share"] * 100).round(1)
        if "crowding_share" in df_display.columns:
            df_display["crowding_share"] = (df_display["crowding_share"] * 100).round(1)

        st.dataframe(df_display.sort_values(["cluster", "route_score"], ascending=[True, False]),
                     use_container_width=True, height=500)

        # Download
        csv = merged.to_csv(index=False)
        st.download_button(
            "📥 Download Cluster Assignments (CSV)",
            csv,
            f"route_clusters_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            "text/csv",
        )


if __name__ == "__main__":
    render()