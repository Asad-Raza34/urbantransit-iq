"""Data-layer validation package.

Logic lives in :mod:`src.validate` so both the pipeline (``pipeline_stages``)
and the app ("data quality" page) can use the same checks.

Exports:
- validator: Rule-based validation with reference layer support
- hidden_data_validator: Schema validation for external/hidden datasets
"""

from src.validate.hidden_data_validator import (
    validate_dataset_schema,
    validate_hidden_dataset,
    validate_pipeline_input,
    generate_schema_documentation,
    ValidationResult,
    REQUIRED_SCHEMAS,
)