"""Hidden data validation for UrbanTransit IQ.

Provides schema validation for external datasets to ensure compatibility
with the UrbanTransit IQ pipeline before processing.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from config import settings
from src.log import get_logger
from src.storage import read_dataset

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of hidden data validation."""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    schema_info: dict = field(default_factory=dict)


# Required schemas for each dataset
REQUIRED_SCHEMAS = {
    "routes": {
        "required_columns": ["route_id", "route_name", "category", "length_km"],
        "column_types": {
            "route_id": "string",
            "route_name": "string", 
            "category": "category",
            "length_km": "numeric",
        },
        "categorical_values": {
            "category": {"core", "feeder", "express"},
        },
        "numeric_bounds": {
            "length_km": (0.5, 80.0),
        },
        "primary_key": ["route_id"],
    },
    "stops": {
        "required_columns": ["stop_id", "stop_name", "zone", "lat", "lon"],
        "column_types": {
            "stop_id": "string",
            "stop_name": "string",
            "zone": "string",
            "lat": "numeric",
            "lon": "numeric",
        },
        "numeric_bounds": {
            "lat": (-90, 90),
            "lon": (-180, 180),
        },
        "primary_key": ["stop_id"],
    },
    "route_stops": {
        "required_columns": ["route_id", "stop_id", "seq", "segment_km"],
        "column_types": {
            "route_id": "string",
            "stop_id": "string",
            "seq": "integer",
            "segment_km": "numeric",
        },
        "numeric_bounds": {
            "seq": (1, 60),
            "segment_km": (0.0, 20.0),
        },
        "primary_key": ["route_id", "seq"],
    },
    "vehicles": {
        "required_columns": ["vehicle_id", "vehicle_type", "capacity"],
        "column_types": {
            "vehicle_id": "string",
            "vehicle_type": "category",
            "capacity": "integer",
        },
        "categorical_values": {
            "vehicle_type": {"standard", "articulated", "mini"},
        },
        "numeric_bounds": {
            "capacity": (1, 400),
        },
        "primary_key": ["vehicle_id"],
    },
    "calendar": {
        "required_columns": ["date", "day_type", "weather", "season"],
        "column_types": {
            "date": "datetime",
            "day_type": "category",
            "weather": "category",
            "season": "category",
        },
        "categorical_values": {
            "day_type": {"weekday", "weekend", "holiday"},
            "weather": {"clear", "rain", "snow", "storm"},
            "season": {"winter", "spring", "summer", "autumn"},
        },
        "primary_key": ["date"],
    },
    "events": {
        "required_columns": ["event_id", "date", "kind", "demand_multiplier"],
        "column_types": {
            "event_id": "string",
            "date": "datetime",
            "kind": "category",
            "demand_multiplier": "numeric",
        },
        "categorical_values": {
            "kind": {"sports", "concert", "festival", "accident", "construction"},
        },
        "numeric_bounds": {
            "demand_multiplier": (0.5, 3.0),
        },
        "primary_key": ["event_id"],
    },
    "trips": {
        "required_columns": [
            "trip_id", "route_id", "vehicle_id", "date", "day_type",
            "direction", "is_peak", "hour", "scheduled_departure",
            "actual_departure", "departure_delay_min", "weather",
            "distance_km", "scheduled_travel_min", "actual_travel_min",
            "boardings", "alightings", "capacity", "occupancy_avg_pct",
            "occupancy_max_pct", "on_time", "scheduled_headway_min",
            "actual_headway_min",
        ],
        "column_types": {
            "trip_id": "string",
            "route_id": "string",
            "vehicle_id": "string",
            "date": "datetime",
            "day_type": "category",
            "direction": "integer",
            "is_peak": "boolean",
            "hour": "integer",
            "scheduled_departure": "datetime",
            "actual_departure": "datetime",
            "departure_delay_min": "numeric",
            "weather": "category",
            "distance_km": "numeric",
            "scheduled_travel_min": "numeric",
            "actual_travel_min": "numeric",
            "boardings": "integer",
            "alightings": "integer",
            "capacity": "integer",
            "occupancy_avg_pct": "numeric",
            "occupancy_max_pct": "numeric",
            "on_time": "boolean",
            "scheduled_headway_min": "numeric",
            "actual_headway_min": "numeric",
        },
        "categorical_values": {
            "day_type": {"weekday", "weekend", "holiday"},
            "weather": {"clear", "rain", "snow", "storm"},
        },
        "numeric_bounds": {
            "hour": (0, 23),
            "direction": (0, 1),
            "departure_delay_min": (settings.DELAY_ON_TIME_MAX - 20, 120),
            "occupancy_max_pct": (0, 200),
            "occupancy_avg_pct": (0, 200),
            "boardings": (0, 400),
            "alightings": (0, 400),
            "capacity": (1, 400),
            "distance_km": (0.1, 100),
            "scheduled_travel_min": (1, 300),
            "actual_travel_min": (1, 300),
        },
        "primary_key": ["trip_id"],
    },
    "stop_trips": {
        "required_columns": [
            "trip_id", "stop_id", "seq", "scheduled_arrival",
            "actual_arrival", "arr_delay_min", "boarded", "alighted", "onboard_after"
        ],
        "column_types": {
            "trip_id": "string",
            "stop_id": "string",
            "seq": "integer",
            "scheduled_arrival": "datetime",
            "actual_arrival": "datetime",
            "arr_delay_min": "numeric",
            "boarded": "integer",
            "alighted": "integer",
            "onboard_after": "integer",
        },
        "numeric_bounds": {
            "seq": (1, 60),
            "arr_delay_min": (-10, 160),
            "boarded": (0, 300),
            "alighted": (0, 300),
            "onboard_after": (0, 500),
        },
        "primary_key": ["trip_id", "seq"],
    },
    "tickets": {
        "required_columns": [
            "ticket_id", "passenger_id", "route_id", "boarding_datetime",
            "alighting_datetime", "boarding_stop_id", "alighting_stop_id",
            "distance_km", "travel_min", "fare", "pass_type"
        ],
        "column_types": {
            "ticket_id": "string",
            "passenger_id": "string",
            "route_id": "string",
            "boarding_datetime": "datetime",
            "alighting_datetime": "datetime",
            "boarding_stop_id": "string",
            "alighting_stop_id": "string",
            "distance_km": "numeric",
            "travel_min": "numeric",
            "fare": "numeric",
            "pass_type": "category",
        },
        "categorical_values": {
            "pass_type": {"single", "day", "weekly", "monthly"},
        },
        "numeric_bounds": {
            "distance_km": (0.1, 100),
            "travel_min": (0.1, 300),
            "fare": (0.1, 1000),
        },
    },
    "passengers": {
        "required_columns": [
            "passenger_id", "home_zone", "pass_type", "birth_year", "annual_trips"
        ],
        "column_types": {
            "passenger_id": "string",
            "home_zone": "category",
            "pass_type": "category",
            "birth_year": "integer",
            "annual_trips": "integer",
        },
        "categorical_values": {
            "pass_type": {"single", "day", "weekly", "monthly"},
            "home_zone": {"residential", "mixed", "commercial", "university", "industrial"},
        },
        "numeric_bounds": {
            "birth_year": (1940, 2012),
            "annual_trips": (1, 400),
        },
        "primary_key": ["passenger_id"],
    },
}


def validate_dataset_schema(
    df: pd.DataFrame,
    dataset_name: str,
    strict: bool = True
) -> ValidationResult:
    """Validate a single dataset against its required schema.
    
    Args:
        df: DataFrame to validate
        dataset_name: Name of the dataset (must match REQUIRED_SCHEMAS keys)
        strict: If True, treat warnings as errors
        
    Returns:
        ValidationResult with validation status and details
    """
    result = ValidationResult(valid=True)
    
    if dataset_name not in REQUIRED_SCHEMAS:
        result.valid = False
        result.errors.append(f"Unknown dataset: {dataset_name}. No schema defined.")
        return result
    
    schema = REQUIRED_SCHEMAS[dataset_name]
    
    # Check required columns
    missing_cols = set(schema["required_columns"]) - set(df.columns)
    if missing_cols:
        result.valid = False
        result.errors.append(f"Missing required columns: {sorted(missing_cols)}")
    
    # Check extra columns (warning only)
    extra_cols = set(df.columns) - set(schema["required_columns"])
    if extra_cols:
        result.warnings.append(f"Extra columns found: {sorted(extra_cols)}")
    
    # Check primary key uniqueness
    if "primary_key" in schema:
        pk_cols = schema["primary_key"]
        if all(c in df.columns for c in pk_cols):
            dup_count = df.duplicated(subset=pk_cols).sum()
            if dup_count > 0:
                result.valid = False
                result.errors.append(f"Primary key violation: {dup_count} duplicate rows on {pk_cols}")
    
    # Check column types
    for col, expected_type in schema.get("column_types", {}).items():
        if col not in df.columns:
            continue
        if not _check_column_type(df[col], expected_type):
            result.valid = False
            result.errors.append(f"Column '{col}': expected type {expected_type}, got {df[col].dtype}")
    
    # Check categorical values
    for col, allowed_values in schema.get("categorical_values", {}).items():
        if col not in df.columns:
            continue
        invalid = df[col].dropna().unique()
        invalid = [v for v in invalid if v not in allowed_values]
        if len(invalid) > 0:
            result.valid = False
            result.errors.append(f"Column '{col}': invalid categorical values {invalid[:5]}. Allowed: {allowed_values}")
    
    # Check numeric bounds
    for col, (lo, hi) in schema.get("numeric_bounds", {}).items():
        if col not in df.columns:
            continue
        col_numeric = pd.to_numeric(df[col], errors="coerce")
        below = (col_numeric < lo).sum()
        above = (col_numeric > hi).sum()
        if below > 0 or above > 0:
            result.valid = False
            result.errors.append(f"Column '{col}': {below} values below {lo}, {above} values above {hi}")
    
    # Check datetime columns
    for col in df.columns:
        if schema.get("column_types", {}).get(col) == "datetime":
            if col in df.columns:
                try:
                    pd.to_datetime(df[col], errors="raise")
                except Exception as e:
                    result.valid = False
                    result.errors.append(f"Column '{col}': invalid datetime format - {e}")
    
    # Add schema info
    result.schema_info = {
        "dataset": dataset_name,
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
    }
    
    if strict and result.warnings:
        result.valid = False
        result.errors.extend(result.warnings)
        result.warnings = []
    
    return result


def _check_column_type(series: pd.Series, expected_type: str) -> bool:
    """Check if a pandas Series matches the expected type."""
    dtype = series.dtype
    
    type_mapping = {
        "string": ["object", "string", "str"],
        "integer": ["int64", "int32", "int16", "int8", "Int64", "Int32"],
        "numeric": ["float64", "float32", "int64", "int32", "Int64", "Int32"],
        "datetime": ["datetime64[ns]", "datetime64", "datetime64[us]"],
        "boolean": ["bool"],
        "category": ["category", "object", "str"],
    }
    
    expected_dtypes = type_mapping.get(expected_type, [expected_type])
    dtype_str = str(dtype)
    return any(dtype_str == dt or dtype_str.startswith(dt.rstrip("[]")) for dt in expected_dtypes)


def validate_hidden_dataset(
    data_path: Path,
    dataset_name: str = None
) -> ValidationResult:
    """Validate a hidden dataset file (Parquet/CSV/JSON) against schema.
    
    Args:
        data_path: Path to the dataset file
        dataset_name: Optional dataset name (inferred from filename if not provided)
        
    Returns:
        ValidationResult with validation status
    """
    # Infer dataset name from filename if not provided
    if dataset_name is None:
        dataset_name = data_path.stem
    
    # Load the dataset
    try:
        if data_path.suffix == ".parquet":
            df = read_dataset(data_path)
        elif data_path.suffix == ".csv":
            df = pd.read_csv(data_path)
        elif data_path.suffix == ".json":
            df = pd.read_json(data_path)
        else:
            return ValidationResult(
                valid=False,
                errors=[f"Unsupported file format: {data_path.suffix}. Use Parquet, CSV, or JSON."]
            )
    except Exception as e:
        return ValidationResult(
            valid=False,
            errors=[f"Failed to load dataset: {e}"]
        )
    
    return validate_dataset_schema(df, dataset_name)


def validate_pipeline_input(
    data_dir: Path,
    layer: str = "raw"
) -> dict[str, ValidationResult]:
    """Validate all datasets in a pipeline input directory.
    
    Args:
        data_dir: Directory containing dataset files
        layer: Pipeline layer name (raw, cleaned, etc.)
        
    Returns:
        Dict mapping dataset names to validation results
    """
    results = {}
    
    for dataset_name in REQUIRED_SCHEMAS:
        file_path = data_dir / f"{dataset_name}.parquet"
        if not file_path.exists():
            # Try CSV
            file_path = data_dir / f"{dataset_name}.csv"
        if not file_path.exists():
            # Try JSON
            file_path = data_dir / f"{dataset_name}.json"
        
        if file_path.exists():
            results[dataset_name] = validate_hidden_dataset(file_path, dataset_name)
        else:
            results[dataset_name] = ValidationResult(
                valid=False,
                errors=[f"Dataset file not found: {dataset_name}.parquet/csv/json"]
            )
    
    return results


def generate_schema_documentation() -> dict:
    """Generate schema documentation for external data providers."""
    docs = {
        "version": "1.0",
        "description": "UrbanTransit IQ Hidden Data Schema Specification",
        "datasets": {}
    }
    
    for name, schema in REQUIRED_SCHEMAS.items():
        docs["datasets"][name] = {
            "description": f"{name.capitalize()} reference data",
            "required_columns": schema["required_columns"],
            "column_types": schema["column_types"],
            "categorical_values": {k: list(v) for k, v in schema.get("categorical_values", {}).items()},
            "numeric_bounds": schema.get("numeric_bounds", {}),
            "primary_key": schema.get("primary_key", []),
        }
    
    return docs


if __name__ == "__main__":
    # Print schema documentation
    docs = generate_schema_documentation()
    print(json.dumps(docs, indent=2))