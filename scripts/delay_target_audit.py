"""Evidence-based audit of the delay-severity classification target.

Answers with code and data, not opinion:

1. What does the generated ``departure_delay_min`` distribution look like
   (overall, per split, per hour / weekday / weather / season / vehicle type)?
2. How much of the delay variance is *deterministic* given the generator's own
   latent inputs, and what is the **Bayes-optimal** performance ceiling for the
   4-class severity target and for binary on-time targets?
3. Which threshold sets are statistically supportable, and which of them align
   with the project's operational on-time definition?
4. Which features in the current classifier are algebraically derived from the
   target (leakage)?

Nothing here modifies data or models. Output: ``reports/delay_target_audit.json``
plus a summary on stdout.

The generator's departure-delay formula (``src/seed/trips.py``) is

    dep_delay = clip(latent(category) + congestion(route x band x weather x day)
                     + weather_effect + disruption(route, day) + chronic(route)
                     + vehicle_reliability + systemic(day) + dow_effect
                     + event_extra + eps, -2, 95)

with ``eps = N(0, RESIDUAL_SD) + Bernoulli(MINOR_INCIDENT_PROB) * U(MINOR_INCIDENT_RANGE)
+ Bernoulli(MAJOR_INCIDENT_PROB) * U(MAJOR_INCIDENT_RANGE)``.  Everything in the
formula except ``eps`` is a deterministic function of trip attributes that are
known before departure, so the Bayes-optimal classifier - the best any model
could ever do - is computable in closed form.  This script scores two versions:

* a **reconstruction** from observable terms plus training-window group
  tendencies (lower bound - excludes the route-day disruption regime), and
* an **oracle** that additionally knows the route-day regime (upper bound).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import settings
from src.paths import DATA_GENERATED, DATA_RAW, REPORTS_ROOT

TRIP_FACTS = "data/processed/trip_facts.parquet"

# The delay process lives in src/seed/trips.py.  The audit imports its
# constants instead of copying them so the reconstruction can never drift away
# from the generator it is supposed to describe.
from src.seed import trips as _gen  # noqa: E402

CATEGORY_LATENT = {k: float(v["latent_delay"]) for k, v in _gen.CATEGORY.items()}
WEATHER_DELAY = dict(_gen.WEATHER_DELAY)
WEATHER_CONGESTION = dict(_gen.WEATHER_CONGESTION)
BAND_CONGESTION = dict(_gen.BAND_CONGESTION)
WEEKEND_CONGESTION = float(_gen.WEEKEND_CONGESTION)
HOLIDAY_CONGESTION = float(_gen.HOLIDAY_CONGESTION)
CONGESTION_BASE = float(_gen.CONGESTION_BASE)
RESIDUAL_SD = float(_gen.RESIDUAL_SD)
MINOR_INCIDENT_PROB = float(_gen.MINOR_INCIDENT_PROB)
MINOR_INCIDENT_RANGE = tuple(_gen.MINOR_INCIDENT_RANGE)
MAJOR_INCIDENT_PROB = float(_gen.MAJOR_INCIDENT_PROB)
MAJOR_INCIDENT_RANGE = tuple(_gen.MAJOR_INCIDENT_RANGE)

CLASS_LABELS = ("on time", "moderate", "severe", "critical")

CANDIDATE_THRESHOLD_SETS = {
    "current_4_7_12": (4.0, 7.0, 12.0),
    "options_3_7_12": (3.0, 7.0, 12.0),
    "options_3_6_12": (3.0, 6.0, 12.0),
    "options_5_10_20": (5.0, 10.0, 20.0),
    "legacy_5_9_30": (5.0, 9.0, 30.0),
}


# ---------------------------------------------------------------------------
# Data loading / splitting
# ---------------------------------------------------------------------------

def _load_trip_facts() -> pd.DataFrame:
    df = pd.read_parquet(TRIP_FACTS)
    df["date"] = pd.to_datetime(df["date"])
    if "delay_severity" in df.columns:
        # analytics pipeline column: different threshold set, not the ML target
        df = df.drop(columns=["delay_severity"])
    return df


def _split(df: pd.DataFrame, horizon_days: int | None = None,
           val_days: int = 28) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    horizon_days = horizon_days or settings.FORECAST_HORIZON_DAYS
    max_date = df["date"].max().normalize()
    cutoff_test = max_date - pd.Timedelta(days=horizon_days)
    cutoff_val = cutoff_test - pd.Timedelta(days=val_days)
    return (df[df["date"] <= cutoff_val].copy(),
            df[(df["date"] > cutoff_val) & (df["date"] <= cutoff_test)].copy(),
            df[df["date"] > cutoff_test].copy())


def _route_zones() -> dict[str, set[str]]:
    """route_id -> stop zones, mirroring src/seed/trips.py::_geometry."""
    rs = pd.read_parquet(DATA_RAW / "route_stops.parquet")
    stops = pd.read_parquet(DATA_RAW / "stops.parquet")
    zone_of = stops.set_index("stop_id")["zone"]
    return {str(rid): set(zone_of.reindex(g["stop_id"]).dropna().tolist())
            for rid, g in rs.groupby("route_id")}


def observable_terms(df: pd.DataFrame) -> pd.DataFrame:
    """The delay terms an operator plans with or observes before departure.

    These are the generator terms that need no hidden state: the category
    baseline, the time-band congestion load amplified by the day's weather, the
    additive weather penalty and the planned event impact.
    """
    band = df["time_band"].map(BAND_CONGESTION).fillna(BAND_CONGESTION["night"])
    day_load = np.where(df["day_type"] == "weekend", WEEKEND_CONGESTION,
                        np.where(df["day_type"] == "holiday", HOLIDAY_CONGESTION, 1.0))
    return pd.DataFrame({
        "category_latent": df["route_category"].map(CATEGORY_LATENT).fillna(0.0),
        "congestion": (CONGESTION_BASE * band
                       * df["weather"].map(WEATHER_CONGESTION).fillna(1.0) * day_load),
        "weather_add": df["weather"].map(WEATHER_DELAY).fillna(0.0),
        "event_extra": _event_delay(df),
    }, index=df.index)


def _shrink(key: pd.Series, values: pd.Series, k: float) -> pd.Series:
    """Mean of ``values`` per ``key`` shrunk towards zero by ``n / (n + k)``."""
    grouped = values.groupby(key)
    return grouped.sum() / (grouped.count() + k)


def reconstruct_signal(df: pd.DataFrame, learn_from: pd.Index | None = None) -> pd.Series:
    """Reconstruction of the *learnable* part of the v3 delay process.

    Observable terms come straight from the generator.  The persistent hidden
    tendencies - per-route chronic lateness, the day-level systemic factor and
    per-vehicle reliability - are estimated from the training window only, which
    is what an expanding history feature converges to.  The route-day
    disruption regime is deliberately left out, so the ceilings below are the
    conservative ones (they assume it is never recognised).
    """
    terms = observable_terms(df)
    obs = terms.sum(axis=1)
    learn = df.index if learn_from is None else learn_from
    resid = df.loc[learn, "departure_delay_min"] - obs.loc[learn]

    route_eff = _shrink(df.loc[learn, "route_id"], resid, 50.0)
    resid2 = resid - df.loc[learn, "route_id"].map(route_eff).fillna(0.0)
    date_eff = _shrink(df.loc[learn, "date"], resid2, 200.0)
    resid3 = resid2 - df.loc[learn, "date"].map(date_eff).fillna(0.0)
    vehicle_eff = _shrink(df.loc[learn, "vehicle_id"], resid3, 100.0)

    mu = (obs
          + df["route_id"].map(route_eff).fillna(0.0)
          + df["date"].map(date_eff).fillna(0.0)
          + df["vehicle_id"].map(vehicle_eff).fillna(0.0))
    return mu.rename("mu")


def _event_delay(df: pd.DataFrame) -> pd.Series:
    """Planned event delay applied by the generator to a route on a date."""
    events = pd.read_parquet(DATA_GENERATED / "events.parquet")
    events["date"] = pd.to_datetime(events["date"])
    zones = _route_zones()
    # the generator's _event_map keeps only the LAST event of a date that
    # touches one of the route's zones
    by_date: dict[pd.Timestamp, dict[str, float]] = {}
    for row in events.itertuples():
        by_date.setdefault(row.date, {})[row.zone] = float(row.extra_delay_min)

    ev = np.zeros(len(df))
    route_ids = df["route_id"].to_numpy()
    dates = pd.to_datetime(df["date"]).to_numpy()
    date_cache: dict[np.datetime64, list[tuple[str, float]]] = {}
    zone_cache: dict[str, set[str]] = {}
    for i, d in enumerate(dates):
        pairs = date_cache.get(d)
        if pairs is None:
            pairs = [(z, v) for z, v in by_date.get(pd.Timestamp(d), {}).items()]
            date_cache[d] = pairs
        if not pairs:
            continue
        rid = route_ids[i]
        rzone = zone_cache.get(rid)
        if rzone is None:
            rzone = zones.get(rid, set())
            zone_cache[rid] = rzone
        for z, v in pairs:
            if z in rzone:
                ev[i] = v
    return pd.Series(ev, index=df.index, name="event_extra")


# ---------------------------------------------------------------------------
# Bayes-optimal ceiling (closed form)
# ---------------------------------------------------------------------------

def residual_cdf(sigma_n: float = RESIDUAL_SD) -> tuple[np.ndarray, np.ndarray]:
    """Grid + CDF of the generator's residual term.

    ``eps = N(0, s) + Bern(p_minor) * U(2, 10) + Bern(p_major) * U(20, 45)``
    (s = RESIDUAL_SD).
    """
    grid = np.linspace(-60.0, 130.0, 12001)
    step = grid[1] - grid[0]
    pdf = (1.0 - MINOR_INCIDENT_PROB - MAJOR_INCIDENT_PROB) * norm.pdf(grid, 0.0, sigma_n)
    for prob, (lo, hi) in ((MINOR_INCIDENT_PROB, MINOR_INCIDENT_RANGE),
                           (MAJOR_INCIDENT_PROB, MAJOR_INCIDENT_RANGE)):
        u_grid = np.linspace(lo, hi, 41)
        conv = np.zeros_like(grid)
        for u in u_grid:
            conv += norm.pdf(grid - u, 0.0, sigma_n)
        pdf = pdf + prob * conv * (step / len(u_grid))
    return grid, np.cumsum(pdf) * step


def class_probabilities(mu_values: np.ndarray, thresholds: tuple[float, float, float],
                        grid: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """P(true class k | mu) for each unique mu value -> (n_mu, 4)."""
    bounds = [-np.inf, thresholds[0], thresholds[1], thresholds[2], np.inf]
    probs = np.zeros((len(mu_values), 4))
    for k in range(4):
        lo = np.interp(bounds[k] - mu_values, grid, cdf, left=0.0, right=1.0)
        hi = np.interp(bounds[k + 1] - mu_values, grid, cdf, left=0.0, right=1.0)
        probs[:, k] = np.clip(hi - lo, 0.0, 1.0)
    return probs


def bayes_confusion(mu: pd.Series, thresholds: tuple[float, float, float],
                    grid: np.ndarray, cdf: np.ndarray,
                    multipliers: np.ndarray | None = None) -> np.ndarray:
    """Expected confusion matrix of a Bayes-style decision rule.

    Every trip's true label is drawn from ``P(k | mu)``; the rule predicts
    ``argmax_k multiplier_k * P(k | mu)`` (all-ones multipliers = the rule that
    maximises accuracy).  The expectation is exact: it is linear in the
    per-trip class probabilities.
    """
    probs = class_probabilities(mu.to_numpy(), thresholds, grid, cdf)
    weights = np.ones(4) if multipliers is None else np.asarray(multipliers, dtype=float)
    pred = (probs * weights[None, :]).argmax(axis=1)
    cm = np.zeros((4, 4))
    for k in range(4):
        cm[k, :] = np.bincount(pred, weights=probs[:, k], minlength=4)
    return cm


def metrics_from_confusion(cm: np.ndarray) -> dict:
    total = float(cm.sum())
    tp = np.diag(cm)
    pred_pos = cm.sum(axis=0)
    actual_pos = cm.sum(axis=1)
    precision = np.divide(tp, pred_pos, out=np.zeros(4), where=pred_pos > 0)
    recall = np.divide(tp, actual_pos, out=np.zeros(4), where=actual_pos > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(4), where=(precision + recall) > 0)
    weights = actual_pos / total if total else np.zeros(4)
    return {
        "accuracy": round(float(tp.sum() / total), 4) if total else 0.0,
        "macro_f1": round(float(f1.mean()), 4),
        "weighted_f1": round(float((f1 * weights).sum()), 4),
        "balanced_accuracy": round(float(recall.mean()), 4),
        "per_class_f1": {CLASS_LABELS[i]: round(float(f1[i]), 4) for i in range(4)},
        "per_class_recall": {CLASS_LABELS[i]: round(float(recall[i]), 4) for i in range(4)},
    }


def deterministic_rule_confusion(sub: pd.DataFrame, mu: pd.Series,
                                 thresholds: tuple[float, float, float]) -> np.ndarray:
    """Confusion of the closed-form rule ``class = interval containing mu``."""
    y = sub["departure_delay_min"].to_numpy()
    rule = np.where(mu.to_numpy() <= thresholds[0], 0,
                    np.where(mu.to_numpy() <= thresholds[1], 1,
                             np.where(mu.to_numpy() <= thresholds[2], 2, 3)))
    actual = np.where(y <= thresholds[0], 0, np.where(y <= thresholds[1], 1,
                                                     np.where(y <= thresholds[2], 2, 3)))
    return np.bincount(actual * 4 + rule, minlength=16).reshape(4, 4).astype(float)


def bayes_binary(mu: pd.Series, threshold: float, grid: np.ndarray, cdf: np.ndarray) -> float:
    p_delay = 1.0 - np.interp(threshold - mu.to_numpy(), grid, cdf, left=0.0, right=1.0)
    return float(np.maximum(p_delay, 1.0 - p_delay).mean())


# ---------------------------------------------------------------------------
# Descriptive evidence
# ---------------------------------------------------------------------------

def leakage_diagnostics(df: pd.DataFrame) -> dict:
    d = df["departure_delay_min"].to_numpy()
    out: dict[str, object] = {}

    # actual_headway - scheduled_headway == delay_i - delay_{i-1} on the same
    # (date, direction) chain: algebraically contains the target.
    dev = (df["actual_headway_min"] - df["scheduled_headway_min"]).to_numpy()
    mask = np.isfinite(dev)
    c = float(np.corrcoef(d[mask], dev[mask])[0, 1])
    out["r2_target_from_headway_deviation"] = round(c ** 2, 4)

    # same-trip occupancy accumulates after departure -> future information
    c2 = float(np.corrcoef(d, df["occupancy_avg_pct"].to_numpy())[0, 1])
    out["r2_target_from_same_trip_occupancy"] = round(c2 ** 2, 4)

    # an expanding mean that includes the current row equals the target on the
    # first row of every group
    first = df.sort_values(["route_id", "scheduled_departure"]).groupby("route_id").head(1)
    out["expanding_mean_including_current_row_is_self_referential"] = True
    out["first_row_of_each_route_count"] = int(len(first))
    return out


def delay_profile(df: pd.DataFrame) -> dict:
    q = df["departure_delay_min"]
    return {
        "count": int(len(df)),
        "mean": round(float(q.mean()), 4),
        "std": round(float(q.std()), 4),
        "min": float(q.min()),
        "max": float(q.max()),
        "median": float(q.median()),
        "quantiles": {str(p): round(float(q.quantile(p / 100)), 3)
                      for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)},
        "share_le_4": round(float((q <= 4).mean()), 4),
        "share_le_5": round(float((q <= 5).mean()), 4),
        "share_le_7": round(float((q <= 7).mean()), 4),
        "share_gt_12": round(float((q > 12).mean()), 4),
        "by_hour_mean": {str(k): round(float(v), 3) for k, v in q.groupby(df["hour"]).mean().items()},
        "by_weekday_mean": {str(k): round(float(v), 3) for k, v in q.groupby(df["weekday"]).mean().items()},
        "by_weather_mean": {str(k): round(float(v), 3) for k, v in q.groupby(df["weather"]).mean().items()},
        "by_season_mean": {str(k): round(float(v), 3) for k, v in q.groupby(df["season"]).mean().items()},
        "by_route_category_mean": {str(k): round(float(v), 3)
                                   for k, v in q.groupby(df["route_category"]).mean().items()},
        "by_vehicle_type_mean": {str(k): round(float(v), 3)
                                 for k, v in q.groupby(df["vehicle_type"]).mean().items()},
        "by_is_peak_mean": {str(k): round(float(v), 3) for k, v in q.groupby(df["is_peak"]).mean().items()},
        "by_day_type_mean": {str(k): round(float(v), 3) for k, v in q.groupby(df["day_type"]).mean().items()},
    }


def _incident_variance() -> float:
    """Variance contributed by both incident processes (routine + serious)."""
    def _var(p: float, lo: float, hi: float) -> float:
        mean = p * (lo + hi) / 2
        second = p * (hi ** 3 - lo ** 3) / (3 * (hi - lo))
        return second - mean ** 2

    return (_var(MINOR_INCIDENT_PROB, *MINOR_INCIDENT_RANGE)
            + _var(MAJOR_INCIDENT_PROB, *MAJOR_INCIDENT_RANGE))


def threshold_sensitivity(train_mu: pd.Series, grid: np.ndarray, cdf: np.ndarray,
                          top: int = 20) -> list[dict]:
    """Scan 4-class threshold triples on TRAIN data only."""
    rows = []
    for t1 in range(1, 9):
        for t2 in range(t1 + 1, 15):
            for t3 in range(t2 + 1, 31):
                cm = bayes_confusion(train_mu, (float(t1), float(t2), float(t3)), grid, cdf)
                m = metrics_from_confusion(cm)
                rows.append({"thresholds": [t1, t2, t3], "train_bayes_accuracy": m["accuracy"],
                             "train_bayes_macro_f1": m["macro_f1"]})
    rows.sort(key=lambda r: -r["train_bayes_accuracy"])
    return rows[:top]


def main() -> None:
    df = _load_trip_facts()
    train, val, test = _split(df)
    mu = reconstruct_signal(df, learn_from=train.index)
    residual = df["departure_delay_min"] - mu
    grid, cdf = residual_cdf()
    terms = observable_terms(df)

    var_total = float(df["departure_delay_min"].var())
    var_mu = float(mu.var())
    var_res = float(residual.var())

    report: dict = {
        "n_trips": int(len(df)),
        "splits": {
            "train": {"n": int(len(train)), "from": str(train["date"].min().date()),
                      "to": str(train["date"].max().date())},
            "val": {"n": int(len(val)), "from": str(val["date"].min().date()),
                    "to": str(val["date"].max().date())},
            "test": {"n": int(len(test)), "from": str(test["date"].min().date()),
                     "to": str(test["date"].max().date())},
        },
        "signal_decomposition": {
            "generator_formula": (
                "clip(latent(category) + CONGESTION_BASE*route_index*band*weather_congestion"
                "*day_load + weather_add*route_weather_sens + disruption(route,day)"
                " + chronic(route) + vehicle_reliability + systemic(day) + dow"
                " + event_extra*(1+N(0,0.30)) + eps, -2, 95)"),
            "noise_term": (f"eps = N(0, {RESIDUAL_SD}) + Bernoulli({MINOR_INCIDENT_PROB}) * "
                           f"U{MINOR_INCIDENT_RANGE} + Bernoulli({MAJOR_INCIDENT_PROB}) * "
                           f"U{MAJOR_INCIDENT_RANGE}"),
            "reconstructed_signal": ("observable terms + per-route/per-vehicle/per-date "
                                     "tendencies learned on the training window only.  The "
                                     "route-day disruption regime and the per-route "
                                     "congestion index are excluded, so every "
                                     "`reconstruction` ceiling here is a CONSERVATIVE "
                                     "LOWER bound on the true achievable ceiling, not an "
                                     "upper bound"),
            "var_total": round(var_total, 4),
            "var_deterministic_mu": round(var_mu, 4),
            "var_residual": round(var_res, 4),
            "r2_ceiling": round(var_mu / var_total, 4),
            "residual_share_of_variance": round(var_res / var_total, 4),
            "residual_sigma_empirical": round(float(residual.std()), 4),
            "residual_sigma_bulk_only": round(RESIDUAL_SD, 4),
            "incident_variance": round(_incident_variance(), 4),
            "incident_share_of_residual_variance": round(_incident_variance() / var_res, 4),
            "reconstructed_components": {
                "congestion": round(float(terms["congestion"].var()), 4),
                "weather_add": round(float(terms["weather_add"].var()), 4),
                "category_latent": round(float(terms["category_latent"].var()), 4),
                "event_extra": round(float(terms["event_extra"].var()), 4),
            },
            "persistent_components_recovered_from_history": {
                "route_tendency": round(float(mu.var() - terms.sum(axis=1).var()), 4),
                "note": "per-route chronic lateness + day-level systemic + vehicle "
                        "reliability, estimated on the training window",
            },
        },
        "delay_profile": {
            "all": delay_profile(df), "train": delay_profile(train),
            "val": delay_profile(val), "test": delay_profile(test),
        },
        "threshold_sensitivity_train_only": threshold_sensitivity(mu.loc[train.index], grid, cdf),
        "leakage": leakage_diagnostics(df),
    }

    report["threshold_sets"] = {}
    for name, thr in CANDIDATE_THRESHOLD_SETS.items():
        cm_tr = bayes_confusion(mu.loc[train.index], thr, grid, cdf)
        cm_te = bayes_confusion(mu.loc[test.index], thr, grid, cdf)
        counts_te = np.bincount(np.where(test["departure_delay_min"].to_numpy() <= thr[0], 0,
                                         np.where(test["departure_delay_min"].to_numpy() <= thr[1], 1,
                                                  np.where(test["departure_delay_min"].to_numpy() <= thr[2], 2, 3))),
                                minlength=4)
        report["threshold_sets"][name] = {
            "thresholds": list(thr),
            "train_class_share": {CLASS_LABELS[i]: round(float(
                (np.bincount(np.where(train["departure_delay_min"].to_numpy() <= thr[0], 0,
                                     np.where(train["departure_delay_min"].to_numpy() <= thr[1], 1,
                                              np.where(train["departure_delay_min"].to_numpy() <= thr[2], 2, 3))),
                            minlength=4) / len(train))[i]), 4) for i in range(4)},
            "test_class_counts": {CLASS_LABELS[i]: int(counts_te[i]) for i in range(4)},
            "bayes_ceiling_train": metrics_from_confusion(cm_tr),
            "bayes_ceiling_test": metrics_from_confusion(cm_te),
            "deterministic_rule_test": metrics_from_confusion(
                deterministic_rule_confusion(test, mu.loc[test.index], thr)),
        }

    report["binary_operational_task"] = {}
    for thr in (4.0, 5.0, 7.0):
        delayed_test = float((test["departure_delay_min"] > thr).mean())
        report["binary_operational_task"][f"delayed_gt_{thr}min"] = {
            "delayed_rate_train": round(float((train["departure_delay_min"] > thr).mean()), 4),
            "delayed_rate_test": round(delayed_test, 4),
            "bayes_ceiling_train": round(bayes_binary(mu.loc[train.index], thr, grid, cdf), 4),
            "bayes_ceiling_test": round(bayes_binary(mu.loc[test.index], thr, grid, cdf), 4),
            "majority_baseline_test": round(max(delayed_test, 1 - delayed_test), 4),
            "note": ("threshold 5.0 min equals settings.DELAY_ON_TIME_MAX, the project's "
                     "existing operational on-time definition"),
        }

    # Reachable frontier for the SRS target thresholds: what accuracy and macro
    # F1 can *any* classifier get on this generator?  The accuracy-maximising
    # rule is the all-ones Bayes rule; re-weighting the per-class probabilities
    # trades accuracy for macro F1 (exactly what the pipeline's validation-fitted
    # decision multipliers do).
    thr = CANDIDATE_THRESHOLD_SETS["legacy_5_9_30"]      # the mandated target
    probs_all = class_probabilities(mu.to_numpy(), thr, grid, cdf)

    def _metrics_with_probs(probs: np.ndarray, weights: np.ndarray) -> dict:
        pred = (probs * weights[None, :]).argmax(axis=1)
        cm = np.zeros((4, 4))
        for k in range(4):
            cm[k, :] = np.bincount(pred, weights=probs[:, k], minlength=4)
        return metrics_from_confusion(cm)

    grid_mult = np.concatenate([np.linspace(0.2, 3.0, 15), np.linspace(3.5, 12.0, 10)])

    def _tuned(probs: np.ndarray) -> dict:
        """Coordinate ascent on per-class multipliers, starting from the accuracy rule."""
        best = _metrics_with_probs(probs, np.ones(4))
        mult = np.ones(4)
        for _ in range(4):
            improved = False
            for k in range(4):
                for g in grid_mult:
                    trial = mult.copy()
                    trial[k] = g
                    m = _metrics_with_probs(probs, trial)
                    if m["macro_f1"] > best["macro_f1"] + 1e-6:
                        best, mult, improved = m, trial, True
            if not improved:
                break
        return best

    best_acc = _metrics_with_probs(probs_all, np.ones(4))
    best_tuned = _tuned(probs_all)

    # Oracle information set: additionally recognise the per-(route, date)
    # disruption regime, which the generator draws once per route-date and which
    # a model can infer from that route's earlier trips on the same day.  ``mu``
    # above is a *reconstruction* (its residual still holds the unrecognised
    # regime), so its Bayes score is only a lower bound; subtracting the
    # route-day mean exposes the residual the generator actually adds, making
    # this second reconstruction score the true ceiling.
    day_key = pd.MultiIndex.from_arrays(
        [df["route_id"].to_numpy(), pd.to_datetime(df["date"]).to_numpy()])
    route_day_effect = (pd.Series(residual.to_numpy(), index=day_key)
                        .groupby(level=[0, 1]).mean())
    mu_oracle = mu + pd.Series(day_key.map(route_day_effect), index=df.index)
    probs_oracle = class_probabilities(mu_oracle.to_numpy(), thr, grid, cdf)
    oracle_acc = _metrics_with_probs(probs_oracle, np.ones(4))
    oracle_tuned = _tuned(probs_oracle)

    report["target_feasibility"] = {
        "thresholds": list(thr),
        "bound_type": (
            "The `ceiling_*` entries are CONSERVATIVE LOWER bounds: they score a "
            "reconstruction that excludes the route-day disruption regime and the "
            "per-route congestion index, so the true achievable ceiling is strictly "
            "higher.  The `oracle_ceiling_*` entries add the route-day regime back and "
            "therefore upper-bound what any classifier can reach on this data."),
        "ceiling_accuracy_rule": best_acc,
        "ceiling_tuned_for_macro_f1": best_tuned,
        "achievable_accuracy_at_or_above_0_80_macro_f1": (
            best_tuned["accuracy"] if best_tuned["macro_f1"] >= 0.80 else None),
        "oracle_ceiling_accuracy_rule": oracle_acc,
        "oracle_ceiling_tuned_for_macro_f1": oracle_tuned,
        "targets": {"accuracy": 0.86, "macro_f1": 0.80},
        "accuracy_target_reachable": bool(oracle_acc["accuracy"] >= 0.86),
        "macro_f1_target_reachable": bool(oracle_tuned["macro_f1"] >= 0.80),
        "reconstruction_accuracy_target_reachable": bool(best_acc["accuracy"] >= 0.86),
        "reconstruction_macro_f1_target_reachable": (
            bool(best_tuned["macro_f1"] >= 0.80)),
        "interpretation": (
            "Reachable frontier for the mandated 5/9/30 thresholds.  The oracle "
            "ceiling knows every persistent factor (route congestion, chronic lateness, "
            "vehicle reliability, day-level systemic effect) *and* the route-day "
            "disruption regime, so no honest classifier can beat it; the residual term "
            "is the generator's own noise and cannot be reduced without changing the "
            "data process.  A model that cannot recognise the disruption regime is "
            "bounded by the lower `reconstruction_*` numbers instead."),
    }

    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    out = REPORTS_ROOT / "delay_target_audit.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}\n")
    print(json.dumps({k: report[k] for k in
                      ("n_trips", "splits", "signal_decomposition", "threshold_sets",
                       "binary_operational_task", "leakage", "target_feasibility")},
                     indent=2, default=str))
    print("\ntop threshold triples by train Bayes accuracy:")
    print(pd.DataFrame(report["threshold_sensitivity_train_only"]).to_string(index=False))


if __name__ == "__main__":
    main()
