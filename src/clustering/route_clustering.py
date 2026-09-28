"""Route clustering for UrbanTransit IQ.

Groups routes into meaningful clusters based on operational characteristics
using K-Means clustering with silhouette-based cluster selection.
"""

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config import settings
from src.analytics.core import read_processed
from src.log import get_logger
from src.paths import DATA_ANALYTICS, MODELS_PYTHON
from src.storage import write_dataset

logger = get_logger(__name__)

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)


# Features used for clustering (must exist in route_metrics)
CLUSTERING_FEATURES = [
    "passengers",
    "avg_delay_min",
    "on_time_share",
    "occupancy_avg_pct",
    "occupancy_max_pct",
    "crowding_share",
    "critical_crowding_share",
    "underutilized_share",
    "avg_headway_min",
    "headway_cv",
    "bunching_share",
    "gapping_share",
    "severe_critical_share",
    "adherence_share",
    "avg_boardings_per_trip",
    "avg_actual_travel_min",
    "route_score",
]


CLUSTER_LABELS = {
    0: "High Demand / High Performance",
    1: "High Delay / Low Reliability",
    2: "High Crowding / Capacity Pressure",
    3: "Low Demand / Underutilized",
    2: "Balanced / Moderate",
    5: "High Bunching / Headway Issues",
    6: "Long Distance / Express",
    7: "Short Distance / Feeder",
}


def _load_route_metrics() -> pd.DataFrame:
    """Load route metrics from analytics."""
    from src.analytics.core import read_dataset
    from src.paths import DATA_ANALYTICS
    return read_dataset(DATA_ANALYTICS / "route_metrics.parquet")


def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Prepare features for clustering.

    Returns:
        - X: scaled feature matrix
        - feature_names: list of feature names used
        - route_ids: DataFrame with route_id for joining back
    """
    # Select available features
    available_features = [f for f in CLUSTERING_FEATURES if f in df.columns]
    logger.info(f"Using {len(available_features)} features for clustering: {available_features}")

    # Keep route identifiers
    route_ids = df[["route_id"]].copy()

    # Extract features
    X = df[available_features].copy()

    # Handle missing values
    X = X.fillna(X.median())

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=available_features, index=df.index)

    return X_scaled, available_features, route_ids


def _find_optimal_clusters(
    X: pd.DataFrame,
    min_clusters: int = 2,
    max_clusters: int = 10,
    random_state: int = 42,
) -> tuple[int, list[float]]:
    """Find optimal number of clusters using silhouette score."""
    silhouette_scores = []

    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
        logger.info(f"  n_clusters={n_clusters}, silhouette={score:.4f}")

    optimal_n = min_clusters + np.argmax(silhouette_scores)
    logger.info(f"Optimal clusters: {optimal_n} (silhouette={max(silhouette_scores):.4f})")

    return optimal_n, silhouette_scores


def _create_cluster_labels(
    df: pd.DataFrame,
    labels: np.ndarray,
    X: pd.DataFrame,
    feature_names: list[str],
) -> dict[int, str]:
    """Generate descriptive labels for clusters based on their characteristics."""
    cluster_labels = {}

    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        cluster_routes = df[mask]

        if len(cluster_routes) == 0:
            cluster_labels[cluster_id] = f"Cluster {cluster_id}"
            continue

        # Compute mean feature values for this cluster
        means = {}
        for feat in feature_names:
            if feat in cluster_routes.columns:
                means[feat] = cluster_routes[feat].mean()

        # Generate descriptive label based on dominant characteristics
        label = _generate_cluster_label(means, cluster_id)
        cluster_labels[cluster_id] = label

    return cluster_labels


def _generate_cluster_label(means: dict, cluster_id: int) -> str:
    """Generate a human-readable label for a cluster based on feature means."""
    # Define thresholds for characterization
    passengers = means.get("passengers", 0)
    avg_delay = means.get("avg_delay_min", 0)
    on_time = means.get("on_time_share", 0)
    occupancy_avg = means.get("occupancy_avg_pct", 0)
    occupancy_max = means.get("occupancy_max_pct", 0)
    crowding = means.get("crowding_share", 0)
    critical_crowding = means.get("critical_crowding_share", 0)
    underutil = means.get("underutilized_share", 0)
    bunching = means.get("bunching_share", 0)
    route_score = means.get("route_score", 0)
    avg_boardings = means.get("avg_boardings_per_trip", 0)
    headway = means.get("avg_headway_min", 0)

    # High-level characterization
    parts = []

    # Demand level
    if passengers > 50000:
        parts.append("Very High Demand")
    elif passengers > 20000:
        parts.append("High Demand")
    elif passengers > 5000:
        parts.append("Moderate Demand")
    else:
        parts.append("Low Demand")

    # Crowding
    if critical_crowding > 0.2 or crowding > 0.4:
        parts.append("Critical Crowding")
    elif crowding > 0.2:
        parts.append("High Crowding")
    elif occupancy_max > 85:
        parts.append("High Occupancy")

    # Delay/Reliability
    if avg_delay > 15 or on_time < 0.55:
        parts.append("Severe Delays")
    elif avg_delay > 10 or on_time < 0.7:
        parts.append("High Delays")
    elif on_time > 0.9:
        parts.append("High Reliability")

    # Bunching
    if bunching > 0.2:
        parts.append("Frequent Bunching")

    # Underutilization
    if underutil > 0.5:
        parts.append("Underutilized")

    # Bunching/Headway
    if headway > 20:
        parts.append("Long Headways")

    if not parts:
        parts.append("Balanced")

    # Shorten to max 3 descriptors
    label = " / ".join(parts[:3])
    return f"Cluster {cluster_id}: {label}"


def _compute_cluster_profiles(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Compute summary statistics for each cluster."""
    profiles = []

    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        cluster_df = df[mask]

        profile = {
            "cluster_id": cluster_id,
            "n_routes": int(mask.sum()),
        }

        # Compute mean/std for key features
        for feat in [
            "passengers", "avg_delay_min", "on_time_share",
            "occupancy_avg_pct", "occupancy_max_pct", "crowding_share",
            "underutilized_share", "bunching_share", "route_score",
            "avg_boardings_per_trip", "avg_headway_min", "avg_actual_travel_min",
        ]:
            if feat in cluster_df.columns:
                vals = cluster_df[feat].dropna()
                profile[f"{feat}_mean"] = float(vals.mean()) if len(vals) > 0 else 0.0
                profile[f"{feat}_std"] = float(vals.std()) if len(vals) > 1 else 0.0

        profiles.append(profile)

    return pd.DataFrame(profiles)


def run_clustering(
    min_clusters: int = 2,
    max_clusters: int = 10,
    n_clusters: int = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run route clustering and return results."""
    logger.info("Loading route metrics for clustering...")
    df = _load_route_metrics()

    if df.empty:
        raise ValueError("No route metrics available for clustering")

    logger.info(f"Loaded {len(df)} routes for clustering")

    # Prepare features
    X_scaled, feature_names, route_ids = _prepare_features(df)
    X_df = pd.DataFrame(X_scaled, columns=feature_names, index=df.index)

    # Determine number of clusters
    if n_clusters is None:
        n_clusters, silhouette_scores = _find_optimal_clusters(
            X_scaled, min_clusters, max_clusters, random_state
        )
    else:
        silhouette_scores = []

    # Run K-Means
    logger.info(f"Running K-Means with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    labels = kmeans.fit_predict(X_scaled)

    # Add cluster assignments to route data
    df_with_clusters = df.copy()
    df_with_clusters["cluster"] = labels

    # Generate cluster labels
    cluster_labels = _create_cluster_labels(df, labels, df, feature_names)

    # Compute cluster profiles
    profiles = _compute_cluster_profiles(df, labels, feature_names)

    # Add label to profiles
    profiles["label"] = profiles["cluster_id"].map(cluster_labels)

    # Save cluster centers (in original scale)
    scaler = StandardScaler()
    scaler.fit(df[feature_names].fillna(df[feature_names].median()))
    centers_original = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=feature_names,
    )
    centers_original["cluster_id"] = range(n_clusters)

    # Prepare results
    results = {
        "n_clusters": int(n_clusters),
        # Scalar score of the selected model, so a consumer that needs one number
        # does not have to guess which entry of ``silhouette_scores`` matches
        # ``n_clusters``. This used to be computed only in build_clustering's
        # return value, so the persisted JSON lacked it and the dashboard's
        # "Silhouette Score" KPI always rendered 0.000 while the toast showed the
        # real score.
        "silhouette_score": float(max(silhouette_scores)) if len(silhouette_scores) else 0.0,
        "silhouette_scores": [float(s) for s in silhouette_scores],
        "labels": [int(l) for l in labels],
        "feature_names": feature_names,
        "cluster_labels": {int(k): v for k, v in cluster_labels.items()},
        "profiles": profiles.to_dict("records"),
        "centers": centers_original.to_dict("records"),
        "inertia": float(kmeans.inertia_),
        "route_assignments": df_with_clusters[["route_id", "cluster"]].to_dict("records"),
    }

    logger.info(f"Clustering complete: {n_clusters} clusters, inertia={kmeans.inertia_:.2f}")
    for cid, label in cluster_labels.items():
        count = (labels == cid).sum()
        logger.info(f"  {label}: {count} routes")

    return results


def save_clustering_results(results: dict[str, Any], output_dir: Path = None) -> dict[str, Path]:
    """Save clustering results to disk."""
    if output_dir is None:
        output_dir = DATA_ANALYTICS / "clustering"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save main results
    results_path = output_dir / "clustering_results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    # Save route assignments as parquet
    assignments_df = pd.DataFrame(results["route_assignments"])
    assignments_path = write_dataset(assignments_df, output_dir / "route_clusters.parquet")

    # Save profiles as parquet
    profiles_df = pd.DataFrame(results["profiles"])
    profiles_path = write_dataset(profiles_df, output_dir / "cluster_profiles.parquet")

    # Save centers as parquet
    centers_df = pd.DataFrame(results["centers"])
    centers_path = write_dataset(centers_df, output_dir / "cluster_centers.parquet")

    logger.info(f"Clustering results saved to {output_dir}")

    return {
        "results": results_path,
        "assignments": assignments_path,
        "profiles": profiles_path,
        "centers": centers_path,
    }


def load_clustering_results(results_dir: Path = None) -> dict[str, Any]:
    """Load clustering results from disk."""
    if results_dir is None:
        results_dir = DATA_ANALYTICS / "clustering"

    results_path = results_dir / "clustering_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Clustering results not found at {results_path}")

    return json.loads(results_path.read_text(encoding="utf-8"))


def load_route_clusters() -> pd.DataFrame:
    """Load route cluster assignments."""
    from src.paths import DATA_ANALYTICS
    from src.storage import read_dataset

    path = DATA_ANALYTICS / "clustering" / "route_clusters.parquet"
    if path.exists():
        return read_dataset(path)
    return pd.DataFrame()


def load_cluster_profiles() -> pd.DataFrame:
    """Load cluster profiles."""
    from src.paths import DATA_ANALYTICS
    from src.storage import read_dataset

    path = DATA_ANALYTICS / "clustering" / "cluster_profiles.parquet"
    if path.exists():
        return read_dataset(path)
    return pd.DataFrame()


def get_route_cluster(route_id: str) -> int | None:
    """Get cluster assignment for a specific route."""
    df = load_route_clusters()
    if df.empty:
        return None
    row = df[df["route_id"] == route_id]
    if row.empty:
        return None
    return int(row["cluster"].iloc[0])


def get_cluster_routes(cluster_id: int) -> list[str]:
    """Get all route IDs in a cluster."""
    df = load_route_clusters()
    if df.empty:
        return []
    return df[df["cluster"] == cluster_id]["route_id"].tolist()


def build_clustering() -> dict:
    """Main entry point: run clustering and persist results."""
    import time

    t0 = time.perf_counter()
    logger.info("Starting route clustering...")

    results = run_clustering()
    paths = save_clustering_results(results)

    duration_ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.info(f"Clustering completed in {duration_ms:.1f}ms")

    return {
        "status": "completed",
        "n_clusters": results["n_clusters"],
        "inertia": results["inertia"],
        "silhouette_score": results["silhouette_score"],
        "duration_ms": duration_ms,
        "paths": {k: str(v) for k, v in paths.items()},
    }


def load_cluster_centers() -> pd.DataFrame:
    """Load cluster centers."""
    from src.paths import DATA_ANALYTICS
    from src.storage import read_dataset

    path = DATA_ANALYTICS / "clustering" / "cluster_centers.parquet"
    if path.exists():
        return read_dataset(path)
    return pd.DataFrame()


if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    result = build_clustering()
    print(json.dumps(result, indent=2, default=str))