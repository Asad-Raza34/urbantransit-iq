"""Tests for hidden data validator."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json

from src.validate.hidden_data_validator import (
    validate_dataset_schema,
    validate_hidden_dataset,
    validate_pipeline_input,
    generate_schema_documentation,
    ValidationResult,
    REQUIRED_SCHEMAS,
)


class TestSchemaDocumentation:
    """Test schema documentation generation."""

    def test_generate_docs(self):
        docs = generate_schema_documentation()
        assert "version" in docs
        assert "datasets" in docs
        assert "routes" in docs["datasets"]
        assert "stops" in docs["datasets"]
        assert "trips" in docs["datasets"]
        
        # Check routes schema
        routes_schema = docs["datasets"]["routes"]
        assert "required_columns" in routes_schema
        assert "route_id" in routes_schema["required_columns"]
        assert "categorical_values" in routes_schema


class TestValidateDatasetSchema:
    """Test dataset schema validation."""

    def test_valid_routes(self):
        df = pd.DataFrame({
            "route_id": ["R-001", "R-002"],
            "route_name": ["Route 1", "Route 2"],
            "category": ["core", "feeder"],
            "length_km": [10.5, 5.2],
        })
        result = validate_dataset_schema(df, "routes")
        assert result.valid is True
        assert len(result.errors) == 0

    def test_valid_stops(self):
        df = pd.DataFrame({
            "stop_id": ["S-001", "S-002"],
            "stop_name": ["Stop 1", "Stop 2"],
            "zone": ["residential", "commercial"],
            "lat": [33.68, 33.69],
            "lon": [73.05, 73.06],
        })
        result = validate_dataset_schema(df, "stops")
        assert result.valid is True

    def test_valid_trips(self):
        df = pd.DataFrame({
            "trip_id": ["T-001", "T-002"],
            "route_id": ["R-001", "R-001"],
            "vehicle_id": ["V-001", "V-002"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "day_type": ["weekday", "weekday"],
            "direction": [0, 1],
            "is_peak": [True, False],
            "hour": [8, 14],
            "scheduled_departure": pd.to_datetime(["2024-01-01 08:00", "2024-01-01 14:00"]),
            "actual_departure": pd.to_datetime(["2024-01-01 08:05", "2024-01-01 14:00"]),
            "departure_delay_min": [5.0, 0.0],
            "weather": ["clear", "clear"],
            "distance_km": [10.0, 10.0],
            "scheduled_travel_min": [30, 30],
            "actual_travel_min": [35, 30],
            "boardings": [20, 15],
            "alightings": [18, 12],
            "capacity": [50, 50],
            "occupancy_avg_pct": [40.0, 30.0],
            "occupancy_max_pct": [60.0, 50.0],
            "on_time": [False, True],
            "scheduled_headway_min": [10.0, 10.0],
            "actual_headway_min": [12.0, 10.0],
        })
        result = validate_dataset_schema(df, "trips")
        assert result.valid is True

    def test_missing_required_column(self):
        df = pd.DataFrame({
            "route_id": ["R-001"],
            "route_name": ["Route 1"],
            # Missing category and length_km
        })
        result = validate_dataset_schema(df, "routes")
        assert result.valid is False
        assert any("Missing required columns" in e for e in result.errors)

    def test_invalid_categorical(self):
        df = pd.DataFrame({
            "route_id": ["R-001"],
            "route_name": ["Route 1"],
            "category": ["invalid_category"],  # Not in allowed values
            "length_km": [10.5],
        })
        result = validate_dataset_schema(df, "routes")
        assert result.valid is False
        assert any("invalid categorical values" in e for e in result.errors)

    def test_numeric_out_of_bounds(self):
        df = pd.DataFrame({
            "route_id": ["R-001"],
            "route_name": ["Route 1"],
            "category": ["core"],
            "length_km": [100.0],  # Exceeds max 80
        })
        result = validate_dataset_schema(df, "routes")
        assert result.valid is False
        assert any("above 80" in e or "out of bounds" in e.lower() for e in result.errors)

    def test_duplicate_primary_key(self):
        df = pd.DataFrame({
            "route_id": ["R-001", "R-001"],  # Duplicate
            "route_name": ["Route 1", "Route 1 Duplicate"],
            "category": ["core", "core"],
            "length_km": [10.5, 10.5],
        })
        result = validate_dataset_schema(df, "routes")
        assert result.valid is False
        assert any("Primary key violation" in e for e in result.errors)

    def test_lat_lon_bounds(self):
        df = pd.DataFrame({
            "stop_id": ["S-001"],
            "stop_name": ["Stop 1"],
            "zone": ["residential"],
            "lat": [100.0],  # Invalid latitude
            "lon": [73.05],
        })
        result = validate_dataset_schema(df, "stops")
        assert result.valid is False
        assert any("lat" in e.lower() for e in result.errors)

    def test_unknown_dataset(self):
        df = pd.DataFrame({"col1": [1, 2]})
        result = validate_dataset_schema(df, "unknown_dataset")
        assert result.valid is False
        assert any("Unknown dataset" in e for e in result.errors)

    def test_warnings_for_extra_columns(self):
        df = pd.DataFrame({
            "route_id": ["R-001"],
            "route_name": ["Route 1"],
            "category": ["core"],
            "length_km": [10.5],
            "extra_col": ["extra"],  # Extra column
        })
        result = validate_dataset_schema(df, "routes", strict=False)
        assert result.valid is True
        assert any("Extra columns" in w for w in result.warnings)

    def test_strict_mode_converts_warnings(self):
        df = pd.DataFrame({
            "route_id": ["R-001"],
            "route_name": ["Route 1"],
            "category": ["core"],
            "length_km": [10.5],
            "extra_col": ["extra"],
        })
        result = validate_dataset_schema(df, "routes", strict=True)
        assert result.valid is False
        assert any("Extra columns" in e for e in result.errors)


class TestValidateHiddenDataset:
    """Test hidden dataset file validation."""

    def test_validate_parquet_file(self, tmp_path):
        # Create a valid parquet file
        df = pd.DataFrame({
            "route_id": ["R-001"],
            "route_name": ["Route 1"],
            "category": ["core"],
            "length_km": [10.5],
        })
        file_path = tmp_path / "routes.parquet"
        df.to_parquet(file_path, index=False)
        
        result = validate_hidden_dataset(file_path, "routes")
        assert result.valid is True

    def test_validate_csv_file(self, tmp_path):
        df = pd.DataFrame({
            "stop_id": ["S-001"],
            "stop_name": ["Stop 1"],
            "zone": ["residential"],
            "lat": [33.68],
            "lon": [73.05],
        })
        file_path = tmp_path / "stops.csv"
        df.to_csv(file_path, index=False)
        
        result = validate_hidden_dataset(file_path, "stops")
        assert result.valid is True

    def test_validate_json_file(self, tmp_path):
        df = pd.DataFrame({
            "vehicle_id": ["V-001"],
            "vehicle_type": ["standard"],
            "capacity": [50],
        })
        file_path = tmp_path / "vehicles.json"
        df.to_json(file_path, orient="records")
        
        result = validate_hidden_dataset(file_path, "vehicles")
        assert result.valid is True

    def test_invalid_file_format(self, tmp_path):
        file_path = tmp_path / "routes.txt"
        file_path.write_text("some text")
        
        result = validate_hidden_dataset(file_path, "routes")
        assert result.valid is False
        assert any("Unsupported file format" in e for e in result.errors)

    def test_missing_file(self, tmp_path):
        file_path = tmp_path / "nonexistent.parquet"
        result = validate_hidden_dataset(file_path, "routes")
        assert result.valid is False
        assert any("Failed to load" in e for e in result.errors)


class TestValidatePipelineInput:
    """Test pipeline input validation."""

    def test_validate_complete_pipeline(self, tmp_path):
        # Create all required datasets
        datasets = {
            "routes": pd.DataFrame({
                "route_id": ["R-001"], "route_name": ["Route 1"],
                "category": ["core"], "length_km": [10.5]
            }),
            "stops": pd.DataFrame({
                "stop_id": ["S-001"], "stop_name": ["Stop 1"],
                "zone": ["residential"], "lat": [33.68], "lon": [73.05]
            }),
            "route_stops": pd.DataFrame({
                "route_id": ["R-001"], "stop_id": ["S-001"],
                "seq": [1], "segment_km": [1.0]
            }),
            "vehicles": pd.DataFrame({
                "vehicle_id": ["V-001"], "vehicle_type": ["standard"], "capacity": [50]
            }),
            "calendar": pd.DataFrame({
                "date": pd.to_datetime(["2024-01-01"]), "day_type": ["weekday"],
                "weather": ["clear"], "season": ["winter"]
            }),
            "events": pd.DataFrame({
                "event_id": ["E-001"], "date": pd.to_datetime(["2024-01-01"]),
                "kind": ["sports"], "demand_multiplier": [1.5]
            }),
            "trips": pd.DataFrame({
                "trip_id": ["T-001"], "route_id": ["R-001"], "vehicle_id": ["V-001"],
                "date": pd.to_datetime(["2024-01-01"]), "day_type": ["weekday"],
                "direction": [0], "is_peak": [True], "hour": [8],
                "scheduled_departure": pd.to_datetime(["2024-01-01 08:00"]),
                "actual_departure": pd.to_datetime(["2024-01-01 08:05"]),
                "departure_delay_min": [5.0], "weather": ["clear"],
                "distance_km": [10.0], "scheduled_travel_min": [30],
                "actual_travel_min": [35], "boardings": [20], "alightings": [18],
                "capacity": [50], "occupancy_avg_pct": [40.0], "occupancy_max_pct": [60.0],
                "on_time": [False], "scheduled_headway_min": [10.0], "actual_headway_min": [12.0]
            }),
            "stop_trips": pd.DataFrame({
                "trip_id": ["T-001"], "stop_id": ["S-001"], "seq": [1],
                "scheduled_arrival": pd.to_datetime(["2024-01-01 08:30"]),
                "actual_arrival": pd.to_datetime(["2024-01-01 08:35"]),
                "arr_delay_min": [5.0], "boarded": [10], "alighted": [5], "onboard_after": [25]
            }),
            "tickets": pd.DataFrame({
                "ticket_id": ["TK-001"], "passenger_id": ["P-001"], "route_id": ["R-001"],
                "boarding_datetime": pd.to_datetime(["2024-01-01 08:00"]),
                "alighting_datetime": pd.to_datetime(["2024-01-01 08:30"]),
                "boarding_stop_id": ["S-001"], "alighting_stop_id": ["S-002"],
                "distance_km": [5.0], "travel_min": [30], "fare": [2.5], "pass_type": ["single"]
            }),
            "passengers": pd.DataFrame({
                "passenger_id": ["P-001"], "home_zone": ["residential"],
                "pass_type": ["monthly"], "birth_year": [1990], "annual_trips": [200]
            }),
        }
        
        for name, df in datasets.items():
            df.to_parquet(tmp_path / f"{name}.parquet", index=False)
        
        results = validate_pipeline_input(tmp_path)
        
        # All should be valid
        for name, result in results.items():
            assert result.valid is True, f"{name} failed: {result.errors}"

    def test_missing_dataset(self, tmp_path):
        # Only create some datasets
        df = pd.DataFrame({
            "route_id": ["R-001"], "route_name": ["Route 1"],
            "category": ["core"], "length_km": [10.5]
        })
        df.to_parquet(tmp_path / "routes.parquet", index=False)
        
        results = validate_pipeline_input(tmp_path)
        
        # routes should be valid
        assert results["routes"].valid is True
        # Others should be invalid (missing)
        for name in ["stops", "route_stops", "vehicles"]:
            assert results[name].valid is False
            assert any("not found" in e for e in results[name].errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])