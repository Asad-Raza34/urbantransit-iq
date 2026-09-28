"""Regression tests for bugs found during end-to-end verification.

Each test here exists because a specific defect shipped and was fixed. The
docstring names the symptom that was actually observed, so a future failure
points straight back at the behaviour that broke.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestNumpyClipBounds:
    """``model.predict(...)`` returns a NumPy array, not a pandas Series."""

    def test_fixed_form_clips_to_bounds(self):
        """The fixed call shape bounds predictions correctly."""
        preds = np.array([-5.0, 10.0, 200.0])
        assert np.clip(preds, 0, 150).tolist() == [0.0, 10.0, 150.0]
        # One-sided bounds are still valid for np.clip.
        assert np.clip(np.array([-3.0, 4.0]), 0, None).tolist() == [0.0, 4.0]
        assert np.clip(np.array([160.0, 4.0]), None, 150).tolist() == [150.0, 4.0]

    def test_pandas_signature_on_array_raises(self):
        """Documents why the old idiom failed.

        ``Series.clip(lower=, upper=)`` is pandas-only. On a NumPy array the
        keywords are ignored and NumPy complains about missing bounds, which is
        the exact message users saw as "Forecast failed: One of max or min must
        be given".
        """
        arr = np.array([-5.0, 10.0, 200.0])
        with pytest.raises(ValueError, match="max or min"):
            arr.clip(lower=0, upper=150)

    def test_no_predict_result_uses_the_pandas_clip_signature(self):
        """Static guard: no module may clip a prediction with pandas keywords.

        This is the bug that made the Occupancy Forecasting "Generate Forecast"
        button and the Delay & Reliability "Generate Prediction" button fail for
        every user while the training pipelines looked healthy.
        """
        offenders = []
        for path in (PROJECT_ROOT / "src").rglob("*.py"):
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if ".predict(" in line and ".clip(lower=" in line:
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{lineno}")
        assert offenders == [], f"pandas clip signature used on predictions: {offenders}"


class TestSingleSaveParquet:
    """Two ``save_parquet`` helpers diverged and broke model evaluation."""

    def test_delay_module_reuses_the_shared_helper(self):
        """Delay predictions must land where the evaluator reads them.

        ``delay_prediction`` used to define its own ``save_parquet`` writing to
        ``data/analytics`` while ``evaluate_delay_model`` read from
        ``models/python``, so evaluation always reported "Predictions not found".
        """
        from src.ml import delay_prediction, demand

        assert delay_prediction.save_parquet is demand.save_parquet

    def test_shared_helper_writes_under_models_python(self, tmp_path, monkeypatch):
        import src.ml.demand as demand
        import src.paths as paths

        monkeypatch.setattr(paths, "MODELS_PYTHON", tmp_path)
        frame = pd.DataFrame({"route_id": ["R-001"], "value": [1.0]})

        written = demand.save_parquet(frame, "unit_test_artifact")

        assert written.parent == tmp_path
        assert written.exists()
        pd.testing.assert_frame_equal(pd.read_parquet(written), frame)


class TestDelayModelResultShape:
    """The page iterated a flat metrics file as if it were nested."""

    def test_loader_returns_nested_model_dict(self):
        """Guards the crash: "'float' object is not iterable".

        The dashboard loop expects ``{model_type: {metric: value}}``; reading a
        single flat metrics JSON produced bare floats for ``result``, so
        ``if "error" in result`` raised ``TypeError``.
        """
        from src.ml.delay_prediction import MODEL_TYPES, load_saved_model_results

        results = load_saved_model_results()
        if not results:
            pytest.skip("no delay metrics on disk; run the pipeline 'delay' stage first")

        assert set(results).issubset(set(MODEL_TYPES))
        for model_name, result in results.items():
            assert isinstance(result, dict), f"{model_name} is {type(result).__name__}, not a dict"
            assert "mae" in result and "r2" in result

    def test_predictions_use_route_date_timeband_grain(self):
        """Training and prediction must share one grain.

        The module was refactored from trip-level to route/date/time_band rows,
        but three output sites still selected ``trip_id`` and ``hour`` and raised
        ``KeyError: "['trip_id', 'hour'] not in index"``.
        """
        from src.ml.delay_prediction import DELAY_TARGET, predict_delay

        try:
            forecast = predict_delay(model_type="random_forest")
        except FileNotFoundError:
            pytest.skip("delay model not trained; run the pipeline 'delay' stage first")

        assert list(forecast.columns) == ["route_id", "date", "time_band", DELAY_TARGET,
                                          "predicted_delay_min"]
        assert "trip_id" not in forecast.columns
        assert "hour" not in forecast.columns
        assert forecast["predicted_delay_min"].notna().all()


class TestNetworkMapRendering:
    """"Generate Map" wrote an ~800 KB file into the working directory each click."""

    def test_map_renders_in_memory_and_writes_nothing(self, tmp_path, monkeypatch):
        if not (PROJECT_ROOT / "data" / "analytics" / "route_metrics.parquet").exists():
            pytest.skip("analytics missing; run the pipeline first")

        from src.map.network_map import create_network_map, render_map_html

        # Any file the renderer writes lands here and fails the assertion below.
        monkeypatch.chdir(tmp_path)

        html = render_map_html(create_network_map(metric_mode="default"))

        assert len(html) > 1000, "map HTML looks empty"
        assert "leaflet" in html.lower(), "map HTML is not a Leaflet document"
        assert list(tmp_path.iterdir()) == [], "map rendering wrote a file to disk"


class TestDualPipelineParity:
    """SRS 31: the Python and Spark pipelines are compared on the same cases."""

    def test_forecast_parity_scores_every_case_with_both_engines(self):
        import json

        summary_path = PROJECT_ROOT / "models" / "python" / "forecast_parity_summary.json"
        cases_path = PROJECT_ROOT / "models" / "python" / "forecast_parity.parquet"
        if not (summary_path.exists() and cases_path.exists()):
            pytest.skip("parity artifacts missing; run the pipeline 'ml' stage first")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        cases = pd.read_parquet(cases_path)

        assert len(cases) == summary["cases"] == 100
        assert cases["forecast_mllib"].notna().all(), "MLlib did not score every case"
        assert cases["forecast_sklearn"].notna().all(), "sklearn did not score every case"
        for engine in ("mllib", "sklearn"):
            assert {"mae", "rmse", "r2"} <= set(summary[engine])

    def test_pipeline_parity_covers_every_route(self):
        import json

        summary_path = PROJECT_ROOT / "models" / "python" / "pipeline_parity_summary.json"
        cases_path = PROJECT_ROOT / "models" / "python" / "pipeline_parity.parquet"
        if not (summary_path.exists() and cases_path.exists()):
            pytest.skip("parity artifacts missing; run the pipeline 'spark' stage first")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        cases = pd.read_parquet(cases_path)

        assert len(cases) == summary["routes"] == 100
        assert "on_time_delta" in cases.columns and "delay_delta" in cases.columns


class TestClusteringResultKeys:
    """The dashboard KPI read a scalar the writer never persisted."""

    def test_persisted_results_carry_the_scalar_silhouette_score(self):
        """Route Clustering showed "Silhouette Score 0.000" on every load.

        ``run_clustering`` saved only the ``silhouette_scores`` list while the
        page read ``results.get("silhouette_score", 0)``, so the KPI always fell
        back to 0.000 even though the toast reported the real score.
        """
        import json

        path = PROJECT_ROOT / "data" / "analytics" / "clustering" / "clustering_results.json"
        if not path.exists():
            pytest.skip("clustering results missing; run the pipeline first")

        data = json.loads(path.read_text(encoding="utf-8"))
        assert "silhouette_score" in data, "scalar silhouette_score was not persisted"
        assert abs(data["silhouette_score"] - max(data["silhouette_scores"])) < 1e-9


class TestCacheInvalidation:
    """Writes were invisible until the server restarted."""

    def test_invalidate_drops_only_named_keys(self, monkeypatch):
        from src.services import data_access

        monkeypatch.setitem(data_access._cache, "alerts", "stale")
        monkeypatch.setitem(data_access._cache, "recommendations", "stale")

        data_access.invalidate("alerts")

        assert "alerts" not in data_access._cache
        assert "recommendations" in data_access._cache

    def test_alert_status_change_invalidates_cached_alerts(self, monkeypatch):
        """Acknowledging reported success while the table still showed NEW."""
        from src.alerts import center
        from src.services import data_access

        frame = pd.DataFrame({"alert_id": ["alt-1", "alt-2"], "status": ["new", "new"]})
        monkeypatch.setattr(center, "read_dataset", lambda *a, **k: frame.copy())
        monkeypatch.setattr(center, "write_dataset", lambda df, path: path)
        monkeypatch.setattr(center, "record", lambda **k: None)
        monkeypatch.setitem(data_access._cache, "alerts", "stale")

        updated = center.update_alert_status(["alt-1"], "acknowledged")

        assert updated == 1
        assert "alerts" not in data_access._cache

    def test_recommendation_rebuild_invalidates_cached_recommendations(self, monkeypatch, tmp_path):
        """Regenerating recommendations left the previous results on screen.

        ``DATA_ANALYTICS`` is redirected to a temp directory. Without that, the
        stub engine's payload overwrote the real
        ``recommendations_explainable.json`` and the Recommendations page then
        failed with ``KeyError: 'recommendation'``.
        """
        from src.recommend import engine as engine_module
        from src.services import data_access

        monkeypatch.setattr(engine_module, "DATA_ANALYTICS", tmp_path)

        class _StubEngine:
            def build_all(self):
                return ["rec-1"]

            def to_dataframe(self):
                return pd.DataFrame({"recommendation_id": ["rec-1"], "priority": ["HIGH"]})

            def to_explainable_format(self):
                return [{"recommendation_id": "rec-1"}]

        monkeypatch.setattr(engine_module, "RecommendationEngine", _StubEngine)
        monkeypatch.setattr(engine_module, "write_dataset", lambda df, path: path)
        monkeypatch.setattr(engine_module, "record", lambda **k: None)
        monkeypatch.setattr(engine_module, "_priority_counts", lambda recs: {"HIGH": 1})
        monkeypatch.setattr(engine_module, "_category_counts", lambda recs: {})
        monkeypatch.setitem(data_access._cache, "recommendations", "stale")

        engine_module.build_recommendations()

        assert "recommendations" not in data_access._cache


class TestWhatIfResultsRender:
    """Both tabs rendered results from one shared key."""

    def test_running_scenarios_renders_results_without_duplicate_ids(self):
        """Guards ``StreamlitDuplicateElementId`` on "Run Selected Scenarios".

        The predefined and custom tabs both called ``render_scenario_results``
        from the same session-state key, so two identical figures were built in
        one script run with no explicit keys and Streamlit refused to render.
        """
        app = PROJECT_ROOT / "app" / "main.py"
        if not (PROJECT_ROOT / "data" / "analytics" / "route_metrics.parquet").exists():
            pytest.skip("analytics missing; run the pipeline first")

        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(app), default_timeout=180)
        at.run()

        nav = next((i for i, sb in enumerate(at.selectbox) if sb.label == "Navigate"), -1)
        assert nav >= 0, "sidebar Navigate selectbox not found"
        at.selectbox[nav].select("🔮 What-If Simulator").run(timeout=180)

        button = next(
            (b for b in at.get("button") if "Run Selected Scenarios" in (b.label or "")),
            None,
        )
        assert button is not None, "what-if run button not found"

        button.click().run(timeout=300)

        assert not at.exception, f"what-if run raised: {at.exception[0].value}"
        assert len(at.get("plotly_chart")) >= 1, "no results chart rendered"
