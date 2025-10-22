"""
Data validator for user-uploaded datasets.

This module provides functionality to validate, sanitize, and check
the integrity of user-uploaded activity datasets.
"""

import pandas as pd
import numpy as np
from typing import List, Set
from models.data_models import ValidationResult, Category


class DataValidator:
    """Validates user-uploaded activity datasets."""

    REQUIRED_COLUMNS = ["Activity", "Avg_CO2_Emission(kg/day)", "Category"]
    VALID_CATEGORIES = {cat.value for cat in Category}

    def __init__(self):
        """Initialize the DataValidator."""
        pass

    def validate_schema(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate that the DataFrame has the required schema.

        Args:
            df: DataFrame to validate

        Returns:
            ValidationResult with validation status and any errors
        """
        errors = []
        warnings = []

        # Check if DataFrame is empty
        if df.empty:
            errors.append("Dataset is empty")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # Check for required columns
        missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")

        # Check for extra columns
        extra_columns = [col for col in df.columns if col not in self.REQUIRED_COLUMNS]
        if extra_columns:
            warnings.append(f"Extra columns will be ignored: {', '.join(extra_columns)}")

        # Check column data types
        if "Avg_CO2_Emission(kg/day)" in df.columns:
            if not pd.api.types.is_numeric_dtype(df["Avg_CO2_Emission(kg/day)"]):
                errors.append("Column 'Avg_CO2_Emission(kg/day)' must contain numeric values")

        if "Activity" in df.columns:
            if not pd.api.types.is_string_dtype(df["Activity"]) and not pd.api.types.is_object_dtype(df["Activity"]):
                warnings.append("Column 'Activity' should contain text values")

        if "Category" in df.columns:
            if not pd.api.types.is_string_dtype(df["Category"]) and not pd.api.types.is_object_dtype(df["Category"]):
                warnings.append("Column 'Category' should contain text values")

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)

    def validate_values(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate the values in the DataFrame.

        Args:
            df: DataFrame to validate

        Returns:
            ValidationResult with validation status and any errors
        """
        errors = []
        warnings = []

        # Check for null values
        if df.isnull().any().any():
            null_columns = df.columns[df.isnull().any()].tolist()
            errors.append(f"Null values found in columns: {', '.join(null_columns)}")

        # Validate emission values
        if "Avg_CO2_Emission(kg/day)" in df.columns:
            emission_col = df["Avg_CO2_Emission(kg/day)"]

            # Check for negative values
            negative_mask = emission_col < 0
            if negative_mask.any():
                negative_count = negative_mask.sum()
                errors.append(f"Found {negative_count} negative emission values (must be >= 0)")

            # Check for unreasonably high values (warning only)
            high_mask = emission_col > 1000
            if high_mask.any():
                high_count = high_mask.sum()
                warnings.append(f"Found {high_count} unusually high emission values (> 1000 kg/day)")

            # Check for NaN or infinite values
            if emission_col.isna().any():
                errors.append("Emission column contains NaN values")
            if np.isinf(emission_col).any():
                errors.append("Emission column contains infinite values")

        # Validate categories
        if "Category" in df.columns:
            invalid_categories = []
            for idx, category in enumerate(df["Category"]):
                if pd.notna(category) and str(category) not in self.VALID_CATEGORIES:
                    invalid_categories.append((idx, str(category)))

            if invalid_categories:
                invalid_list = [f"Row {idx}: '{cat}'" for idx, cat in invalid_categories[:5]]
                error_msg = f"Invalid categories found. Valid categories are: {', '.join(self.VALID_CATEGORIES)}. "
                error_msg += f"Examples: {', '.join(invalid_list)}"
                if len(invalid_categories) > 5:
                    error_msg += f" (and {len(invalid_categories) - 5} more)"
                errors.append(error_msg)

        # Validate activity names
        if "Activity" in df.columns:
            # Check for empty activity names
            empty_mask = df["Activity"].astype(str).str.strip() == ""
            if empty_mask.any():
                empty_count = empty_mask.sum()
                errors.append(f"Found {empty_count} empty activity names")

            # Check for duplicate activities
            duplicates = df["Activity"].duplicated()
            if duplicates.any():
                duplicate_count = duplicates.sum()
                warnings.append(f"Found {duplicate_count} duplicate activity names")

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)

    def sanitize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalize the input data.

        Args:
            df: DataFrame to sanitize

        Returns:
            Sanitized DataFrame
        """
        # Create a copy to avoid modifying the original
        df_clean = df.copy()

        # Keep only required columns
        available_columns = [col for col in self.REQUIRED_COLUMNS if col in df_clean.columns]
        df_clean = df_clean[available_columns]

        # Remove rows with null values in required columns
        df_clean = df_clean.dropna(subset=available_columns)

        # Clean activity names
        if "Activity" in df_clean.columns:
            # Strip whitespace
            df_clean["Activity"] = df_clean["Activity"].astype(str).str.strip()
            # Remove empty activity names
            df_clean = df_clean[df_clean["Activity"] != ""]

        # Clean emission values
        if "Avg_CO2_Emission(kg/day)" in df_clean.columns:
            # Convert to numeric, coercing errors to NaN
            df_clean["Avg_CO2_Emission(kg/day)"] = pd.to_numeric(
                df_clean["Avg_CO2_Emission(kg/day)"], errors="coerce"
            )
            # Remove rows with NaN or negative emissions
            df_clean = df_clean[df_clean["Avg_CO2_Emission(kg/day)"].notna()]
            df_clean = df_clean[df_clean["Avg_CO2_Emission(kg/day)"] >= 0]
            # Remove infinite values
            df_clean = df_clean[~np.isinf(df_clean["Avg_CO2_Emission(kg/day)"])]

        # Clean category values
        if "Category" in df_clean.columns:
            # Strip whitespace and standardize
            df_clean["Category"] = df_clean["Category"].astype(str).str.strip()
            # Keep only valid categories
            df_clean = df_clean[df_clean["Category"].isin(self.VALID_CATEGORIES)]

        # Remove duplicate activities (keep first occurrence)
        if "Activity" in df_clean.columns:
            df_clean = df_clean.drop_duplicates(subset=["Activity"], keep="first")

        # Reset index
        df_clean = df_clean.reset_index(drop=True)

        return df_clean

    def validate_and_sanitize(self, df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationResult]:
        """
        Validate and sanitize the DataFrame in one step.

        Args:
            df: DataFrame to validate and sanitize

        Returns:
            Tuple of (sanitized DataFrame, ValidationResult)
        """
        # First validate schema
        schema_result = self.validate_schema(df)
        if not schema_result.is_valid:
            return df, schema_result

        # Then validate values
        value_result = self.validate_values(df)

        # Sanitize the data
        df_clean = self.sanitize_data(df)

        # Combine errors and warnings
        all_errors = schema_result.errors + value_result.errors
        all_warnings = schema_result.warnings + value_result.warnings

        # Add info about sanitization
        if len(df_clean) < len(df):
            removed_count = len(df) - len(df_clean)
            all_warnings.append(f"Removed {removed_count} invalid rows during sanitization")

        final_result = ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings
        )

        return df_clean, final_result

    def get_validation_summary(self, df: pd.DataFrame) -> dict:
        """
        Get a summary of validation results.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with validation summary
        """
        schema_result = self.validate_schema(df)
        value_result = self.validate_values(df)

        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "schema_valid": schema_result.is_valid,
            "values_valid": value_result.is_valid,
            "total_errors": len(schema_result.errors) + len(value_result.errors),
            "total_warnings": len(schema_result.warnings) + len(value_result.warnings),
            "errors": schema_result.errors + value_result.errors,
            "warnings": schema_result.warnings + value_result.warnings
        }
