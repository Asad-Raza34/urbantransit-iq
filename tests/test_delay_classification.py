"""Tests for the delay severity / delay status classification module.

Covers (SRS phases 3/18): target boundaries, feature catalogue discipline,
leakage prevention, chronological splitting, metric/support consistency,
artifact consistency and report generation.  The heavy feature frame is built
once per module and shared.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import settings
from src.integrate.integrator import _delay_severity as integrator_severity
from src.ml.delay_classification import (
    CLASS_ORDER,
    FEATURE_NAMES,
    FEATURE_SPECS,
    FORBIDDEN_COLUMNS,
    PRIMARY_TARGET,
    TARGETS,
    TARGET_SEVERITY,
    TARGET_STATUS,
    attach_target,
    build_features,
    candidate_specs,
    check_metric_consistency,
    classification_metrics,
    class_weight_mapping,
    split_train_val_test,
)
from src.ml.delay_classification import OrdinalCumulativeClassifier
from src.ml.delay_classification_report import REPORT_PATH
from src.paths import MODELS_PYTHON, REPORTS_ROOT

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return build_features()


@pytest.fixture(scope="module")
def labelled(features: pd.DataFrame) -> pd.DataFrame:
    return attach_target(features, TARGET_SEVERITY)


# ---------------------------------------------------------------------------
# Target definitions (Phase 3: audit / boundaries)
# ---------------------------------------------------------------------------


class TestTargetDefinitions:
    def test_ml_severity_matches_integrator(self):
        delay = pd.Series([-2.0, 0.0, 4.9, 5.0, 5.1, 8.9, 9.0, 9.1, 15.0, 29.9, 30.0, 30.1, 62.0])
        mine = TARGET_SEVERITY.label_delay(delay)
        theirs = integrator_severity(delay)
        assert (mine.to_numpy() == theirs.to_numpy()).all(), (
            "the ML severity target must be identical to the analytics/integrator definition")

    def test_status_binary_mapping(self):
        delay = pd.Series([0.0, 5.0, 5.1, 30.0])
        labels = TARGET_STATUS.label_delay(delay)
        assert labels.tolist() == ["on time", "on time", "delayed", "delayed"]

    def test_thresholds_strictly_increasing(self):
        for spec in TARGETS.values():
            assert list(spec.thresholds) == sorted(spec.thresholds)
            assert all(t > 0 for t in spec.thresholds)

    def test_primary_target_is_platform_severity(self):
        assert PRIMARY_TARGET == "severity"
        assert TARGETS[PRIMARY_TARGET].labels == ("on time", "moderate", "severe", "critical")
        assert TARGETS[PRIMARY_TARGET].thresholds == (
            settings.DELAY_ON_TIME_MAX, settings.DELAY_MODERATE_MAX,
            settings.DELAY_CRITICAL_MAX)

    def test_class_order_covers_severity_labels(self):
        assert list(CLASS_ORDER) == list(TARGET_SEVERITY.labels)


# ---------------------------------------------------------------------------
# Feature catalogue discipline (Phase 5/15)
# ---------------------------------------------------------------------------


class TestFeatureCatalogue:
    def test_every_feature_has_a_spec(self):
        assert {s.name for s in FEATURE_SPECS} == set(FEATURE_NAMES)

    def test_no_forbidden_column_in_features(self):
        overlap = set(FORBIDDEN_COLUMNS) & set(FEATURE_NAMES)
        assert not overlap, f"leaky features present: {overlap}"

    def test_availability_classes_only_abc(self):
        bad = [s.name for s in FEATURE_SPECS if s.availability not in {"A", "B", "C"}]
        assert not bad, f"features with future/target availability: {bad}"

    def test_historical_features_are_documented_as_historical(self):
        for spec in FEATURE_SPECS:
            if "prior" in spec.name or "lag" in spec.name or "rolling" in spec.name:
                assert spec.availability == "B", spec.name

    def test_candidate_specs_generated(self):
        specs = candidate_specs(TARGET_SEVERITY, full_sweep=True)
        families = {s.family for s in specs}
        assert {"logistic_regression", "random_forest", "extra_trees",
                "hist_gradient_boosting", "gradient_boosting", "ordinal_cumulative"} <= families
        weights = {s.weight for s in specs}
        assert weights <= {"none", "balanced", "sqrt"}
        assert {"none", "balanced", "sqrt"} <= weights

    def test_class_weight_mapping_ordering(self):
        y = pd.Series(["a"] * 100 + ["b"] * 20 + ["c"] * 5)
        none = class_weight_mapping(y, "none", ["a", "b", "c"])
        balanced = class_weight_mapping(y, "balanced", ["a", "b", "c"])
        sqrt = class_weight_mapping(y, "sqrt", ["a", "b", "c"])
        assert none is None
        assert balanced["c"] > sqrt["c"] > sqrt["a"] > balanced["a"] > 0

    def test_ordinal_classifier_orders_classes(self):
        rng = np.random.default_rng(7)
        X = np.column_stack([rng.normal(size=400)])
        score = X[:, 0]
        codes = np.digitize(score, [-1.0, 1.0])
        y = np.array(["low", "mid", "high"], dtype=object)[codes]
        model = OrdinalCumulativeClassifier(
            base_family="logistic_regression",
            base_params={"C": 1.0, "max_iter": 200},
            labels=["low", "mid", "high"])
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (400, 3)
        assert np.allclose(proba.sum(axis=1), 1.0)
        pred = model.predict(X)
        # monotone signal: high-score rows are never predicted 'low' in bulk
        assert (pred[score > 1.5] == "low").mean() < 0.2


# ---------------------------------------------------------------------------
# Feature engineering: leakage guards (Phase 15)
# ---------------------------------------------------------------------------


class TestFeatureEngineering:
    def test_all_features_present(self, features: pd.DataFrame):
        missing = [f for f in FEATURE_NAMES if f not in features.columns]
        assert not missing, f"missing features: {missing}"

    def test_history_excludes_current_trip(self, features: pd.DataFrame):
        """Expanding mean must equal the mean of strictly earlier trips."""
        frame = features[features["route_id"] == features["route_id"].iloc[0]]
        frame = frame.sort_values("scheduled_departure").head(30)
        for i in range(1, len(frame)):
            past = frame["departure_delay_min"].iloc[:i]
            expected = float(past.mean())
            got = float(frame["route_prior_delay_mean"].iloc[i])
            assert abs(got - expected) < 1e-6, f"row {i}: {got} != {expected}"

    def test_first_trip_has_no_history(self, features: pd.DataFrame):
        first = (features.sort_values(["route_id", "scheduled_departure"])
                 .groupby("route_id").head(1))
        assert (first["route_date_prior_trip_count"] == 0).all()
        assert first["route_prior_delay_mean"].isna().all()

    def test_same_day_history_uses_only_earlier_trips(self, features: pd.DataFrame):
        group = (features[(features["route_id"] == "R-001")]
                 .sort_values("scheduled_departure").head(10))
        delays = group["departure_delay_min"].to_numpy()
        for i in range(1, len(group)):
            expected = float(delays[:i].mean())
            assert abs(float(group["route_date_prior_delay_mean"].iloc[i]) - expected) < 1e-6

    def test_feature_nan_share_is_bounded(self, features: pd.DataFrame):
        shares = features[FEATURE_NAMES].isna().mean()
        assert shares.max() < 0.30, shares.sort_values(ascending=False).head(5).to_dict()

    def test_event_features_route_matched(self, features: pd.DataFrame):
        assert set(features["has_event"].unique()) <= {0, 1}
        assert (features.loc[features["has_event"] == 0, "event_extra_delay"] == 0).all()
        # the generator applies events to a minority of route-days only
        assert features["has_event"].mean() < 0.25


# ---------------------------------------------------------------------------
# Chronological split (Phase 7/16 discipline)
# ---------------------------------------------------------------------------


class TestChronologicalSplit:
    def test_no_date_overlap(self, labelled: pd.DataFrame):
        train, val, test = split_train_val_test(labelled)
        assert train["date"].max() < val["date"].min()
        assert val["date"].max() < test["date"].min()

    def test_test_window_is_last_28_days(self, labelled: pd.DataFrame):
        train, val, test = split_train_val_test(labelled)
        expected_start = (labelled["date"].max().normalize()
                          - pd.Timedelta(days=settings.FORECAST_HORIZON_DAYS)
                          + pd.Timedelta(days=1))
        assert test["date"].min().normalize() == expected_start

    def test_validation_window_is_28_days(self, labelled: pd.DataFrame):
        _, val, _ = split_train_val_test(labelled)
        span = (val["date"].max().normalize() - val["date"].min().normalize()).days
        assert 27 <= span <= 28

    def test_no_shuffling(self, labelled: pd.DataFrame):
        """Splits preserve the deterministic (route, date, departure) ordering."""
        train, val, test = split_train_val_test(labelled)
        for frame in (train, val, test):
            per_route = frame.groupby("route_id")["scheduled_departure"].apply(
                lambda s: s.is_monotonic_increasing)
            assert per_route.all(), "rows within a route must stay in departure order"


# ---------------------------------------------------------------------------
# Metric computation + consistency (Phase 2: no contradictory reporting)
# ---------------------------------------------------------------------------


class TestMetricConsistency:
    def test_perfect_predictions(self):
        y = np.array(["a", "b", "c", "a", "b", "c"])
        m = classification_metrics(y, y.copy())
        assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0
        assert check_metric_consistency(m) == []

    def test_support_matches_confusion_row_sums(self):
        rng = np.random.default_rng(3)
        y_true = rng.choice(["a", "b", "c"], size=500)
        y_pred = rng.choice(["a", "b", "c"], size=500)
        m = classification_metrics(y_true, y_pred)
        cm = np.array(m["confusion_matrix"])
        for i, label in enumerate(m["labels"]):
            assert m["per_class"][label]["support"] == int(cm[i].sum())
        assert sum(m["class_counts"].values()) == m["n"]
        assert abs(sum(m["class_shares"].values()) - 1.0) < 1e-6
        assert check_metric_consistency(m) == []

    def test_binary_auc_present(self):
        rng = np.random.default_rng(11)
        y_true = np.array(["neg"] * 300 + ["pos"] * 100)
        y_prob = np.zeros((400, 2))
        y_prob[:300, 0] = rng.uniform(0.6, 1.0, 300)
        y_prob[:300, 1] = 1 - y_prob[:300, 0]
        y_prob[300:, 1] = rng.uniform(0.6, 1.0, 100)
        y_prob[300:, 0] = 1 - y_prob[300:, 1]
        y_pred = np.where(y_prob[:, 1] > 0.5, "pos", "neg")
        m = classification_metrics(y_true, y_pred, y_prob=y_prob)
        assert 0.0 <= m.get("roc_auc", 0) <= 1.0
        assert 0.0 <= m.get("pr_auc", 0) <= 1.0

    def test_consistency_checker_detects_planted_errors(self):
        m = classification_metrics(np.array(["a", "b", "b", "a"]),
                                   np.array(["a", "b", "a", "b"]))
        broken = dict(m)
        broken["accuracy"] = 0.99
        assert check_metric_consistency(broken)  # non-empty list of problems
        broken2 = dict(m)
        broken2["confusion_matrix"] = [[1, 1], [1, 0]]
        assert check_metric_consistency(broken2)


# ---------------------------------------------------------------------------
# Persisted artifacts (single source of truth for the report/dashboard)
# ---------------------------------------------------------------------------


def _frozen_files() -> dict[str, Path]:
    """Metrics artifacts of the *current* targets (legacy names are ignored)."""
    return {name: MODELS_PYTHON / f"delay_classification_{name}_metrics.json"
            for name in TARGETS
            if (MODELS_PYTHON / f"delay_classification_{name}_metrics.json").exists()}


class TestPersistedArtifacts:
    def test_metrics_files_are_internally_consistent(self):
        files = _frozen_files()
        assert files, "no delay classification metrics found - run the training stages first"
        for name, path in files.items():
            data = json.loads(path.read_text(encoding="utf-8"))
            for split in ("validation", "test"):
                assert split in data, f"{name}: missing {split}"
                problems = check_metric_consistency(data[split])
                assert problems == [], f"{name}/{split}: {problems}"

    def test_predictions_file_reproduces_test_metrics(self):
        files = _frozen_files()
        for name, path in files.items():
            data = json.loads(path.read_text(encoding="utf-8"))
            pred_path = Path(data["predictions_path"])
            if not pred_path.exists():
                continue
            preds = pd.read_parquet(pred_path)
            recomputed = classification_metrics(
                preds["actual_target"].to_numpy(dtype=object),
                preds["predicted_target"].to_numpy(dtype=object),
                labels=data["labels"])
            assert recomputed["accuracy"] == data["test"]["accuracy"], name
            assert recomputed["macro_f1"] == data["test"]["macro_f1"], name
            assert recomputed["confusion_matrix"] == data["test"]["confusion_matrix"], name

    def test_test_rows_match_split_record(self):
        for name, path in _frozen_files().items():
            data = json.loads(path.read_text(encoding="utf-8"))
            split = data["split"]
            assert split["test"]["rows"] == data["test"]["n"], name
            assert split["final_fit_rows"] == split["train"]["rows"] + split["validation"]["rows"]
            # percentages derive from counts: guards against the old
            # support/percentage mismatch bug class
            counts = data["class_distribution"]["test"]
            assert sum(counts.values()) == data["test"]["n"], name
            for label, share in data["test"]["class_shares"].items():
                assert abs(share - counts[label] / data["test"]["n"]) < 1e-4, (name, label)

    def test_frozen_records_keep_test_discipline(self):
        for name, path in _frozen_files().items():
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["test_evaluated"] is True
            assert data["test_evaluation_count"] >= 1
            exp = json.loads((REPORTS_ROOT /
                              f"delay_classification_experiments_{name}.json").read_text(
                                  encoding="utf-8"))
            assert exp["test_period_touched"] is False
            selected = exp["selected"]["candidate"]
            frozen = {k: v for k, v in data["candidate"].items() if k in selected}
            assert frozen == selected, (
                f"{name}: frozen configuration differs from the selection winner")
            assert data["artifact_digest"]

    def test_feature_catalogue_persisted(self):
        for _name, path in _frozen_files().items():
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["features"] == FEATURE_NAMES
            names = [s["name"] for s in data["feature_catalogue"]]
            assert names == FEATURE_NAMES
            assert data["forbidden_columns"] == list(FORBIDDEN_COLUMNS)


# ---------------------------------------------------------------------------
# Report generation (Phase 20: every number from code)
# ---------------------------------------------------------------------------


class TestGeneratedReport:
    def test_report_exists_and_is_generated(self):
        assert REPORT_PATH.exists(), "run `python -m src.ml.delay_classification --stage report`"
        text = REPORT_PATH.read_text(encoding="utf-8")
        assert "generated by" in text.lower()
        for section in range(1, 24):
            assert f"## {section}. " in text, f"missing section {section}"

    def test_report_numbers_come_from_artifacts(self):
        text = REPORT_PATH.read_text(encoding="utf-8")
        for name, path in _frozen_files().items():
            data = json.loads(path.read_text(encoding="utf-8"))
            accuracy = f"{data['test']['accuracy']:.4f}"
            assert accuracy in text, f"{name}: test accuracy {accuracy} absent from report"
            cm_first = str(data["test"]["confusion_matrix"][0][0])
            assert cm_first in text, f"{name}: confusion matrix values absent from report"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
