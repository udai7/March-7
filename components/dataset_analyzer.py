"""
Dataset analyzer for processing uploaded activity datasets.

This module validates, analyzes, and extracts insights from user-uploaded
CO2 emission datasets.
"""

import pandas as pd
from typing import List, Dict, Any
from models.data_models import (
    Activity, Category, DatasetAnalysis, ValidationResult
)
from components.data_validator import DataValidator
from components.emission_calculator import EmissionCalculator


class DatasetAnalyzer:
    """Analyzes uploaded activity datasets for CO2 emissions."""
    
    def __init__(self):
        """Initialize the DatasetAnalyzer with validator and calculator."""
        self.validator = DataValidator()
        self.calculator = EmissionCalculator()
    
    def validate_dataset(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate the uploaded dataset using DataValidator.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult with validation status and messages
        """
        # Use the validator's comprehensive validation
        schema_result = self.validator.validate_schema(df)
        if not schema_result.is_valid:
            return schema_result
        
        value_result = self.validator.validate_values(df)
        return value_result
    
    def calculate_total_emissions(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate total daily and annual emissions for uploaded data.
        
        Args:
            df: DataFrame with validated activity data
            
        Returns:
            Dictionary with total_daily and total_annual emissions
        """
        if df.empty:
            return {
                'total_daily': 0.0,
                'total_annual': 0.0
            }
        
        # Convert DataFrame to Activity objects
        activities = self._dataframe_to_activities(df)
        
        # Calculate emissions
        total_daily = self.calculator.calculate_daily_emission(activities)
        total_annual = self.calculator.calculate_annual_emission(total_daily)
        
        return {
            'total_daily': total_daily,
            'total_annual': total_annual
        }
    
    def identify_top_emitters(
        self, 
        df: pd.DataFrame, 
        n: int = 3
    ) -> List[Activity]:
        """
        Find the highest emission activities from the dataset.
        
        Args:
            df: DataFrame with validated activity data
            n: Number of top emitters to return (default: 3)
            
        Returns:
            List of top N Activity objects sorted by emission
        """
        if df.empty:
            return []
        
        # Convert DataFrame to Activity objects
        activities = self._dataframe_to_activities(df)
        
        # Use calculator to get top emitters
        top_emitters = self.calculator.get_top_emitters(activities, n=n)
        
        return top_emitters
    
    def analyze_dataset(
        self, 
        df: pd.DataFrame,
        top_n: int = 3
    ) -> DatasetAnalysis:
        """
        Perform comprehensive analysis of the dataset.
        
        Args:
            df: DataFrame with activity data
            top_n: Number of top emitters to identify
            
        Returns:
            DatasetAnalysis object with complete analysis
            
        Raises:
            ValueError: If dataset validation fails
        """
        # Validate the dataset
        validation_result = self.validate_dataset(df)
        if not validation_result.is_valid:
            error_msg = "; ".join(validation_result.errors)
            raise ValueError(f"Dataset validation failed: {error_msg}")
        
        # Sanitize the data
        df_clean = self.validator.sanitize_data(df)
        
        if df_clean.empty:
            raise ValueError("No valid data remaining after sanitization")
        
        # Convert to activities
        activities = self._dataframe_to_activities(df_clean)
        
        # Calculate total emissions
        total_daily = self.calculator.calculate_daily_emission(activities)
        total_annual = self.calculator.calculate_annual_emission(total_daily)
        
        # Identify top emitters
        top_emitters = self.calculator.get_top_emitters(activities, n=top_n)
        
        # Calculate category breakdown
        category_breakdown = self.calculator.calculate_category_emissions(activities)
        
        # Create analysis result (recommendations will be added by recommendation generator)
        analysis = DatasetAnalysis(
            total_daily_emission=total_daily,
            total_annual_emission=total_annual,
            top_emitters=top_emitters,
            category_breakdown=category_breakdown,
            recommendations=[]  # Will be populated by recommendation generator
        )
        
        return analysis
    
    def get_category_breakdown(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate emissions grouped by category.
        
        Args:
            df: DataFrame with activity data
            
        Returns:
            Dictionary mapping category names to total emissions
        """
        if df.empty:
            return {}
        
        activities = self._dataframe_to_activities(df)
        return self.calculator.calculate_category_emissions(activities)
    
    def get_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the dataset.
        
        Args:
            df: DataFrame with activity data
            
        Returns:
            Dictionary with various statistics
        """
        if df.empty:
            return {
                'total_activities': 0,
                'total_daily_emission': 0.0,
                'total_annual_emission': 0.0,
                'average_emission_per_activity': 0.0,
                'category_breakdown': {},
                'top_emitters': []
            }
        
        activities = self._dataframe_to_activities(df)
        
        total_daily = self.calculator.calculate_daily_emission(activities)
        total_annual = self.calculator.calculate_annual_emission(total_daily)
        
        avg_emission = total_daily / len(activities) if activities else 0.0
        
        return {
            'total_activities': len(activities),
            'total_daily_emission': total_daily,
            'total_annual_emission': total_annual,
            'average_emission_per_activity': round(avg_emission, 2),
            'category_breakdown': self.calculator.calculate_category_emissions(activities),
            'top_emitters': self.calculator.get_top_emitters(activities, n=3)
        }
    
    def compare_with_average(
        self, 
        df: pd.DataFrame,
        average_daily_emission: float = 10.0
    ) -> Dict[str, Any]:
        """
        Compare dataset emissions with an average baseline.
        
        Args:
            df: DataFrame with activity data
            average_daily_emission: Average daily emission to compare against (kg/day)
            
        Returns:
            Dictionary with comparison metrics
        """
        if df.empty:
            return {
                'user_daily_emission': 0.0,
                'average_daily_emission': average_daily_emission,
                'difference': -average_daily_emission,
                'percentage_difference': -100.0,
                'status': 'below_average'
            }
        
        activities = self._dataframe_to_activities(df)
        user_daily = self.calculator.calculate_daily_emission(activities)
        
        difference = user_daily - average_daily_emission
        percentage_diff = (difference / average_daily_emission * 100) if average_daily_emission > 0 else 0.0
        
        if user_daily < average_daily_emission:
            status = 'below_average'
        elif user_daily > average_daily_emission:
            status = 'above_average'
        else:
            status = 'at_average'
        
        return {
            'user_daily_emission': user_daily,
            'average_daily_emission': average_daily_emission,
            'difference': round(difference, 2),
            'percentage_difference': round(percentage_diff, 2),
            'status': status
        }
    
    def _dataframe_to_activities(self, df: pd.DataFrame) -> List[Activity]:
        """
        Convert DataFrame to list of Activity objects.
        
        Args:
            df: DataFrame with activity data
            
        Returns:
            List of Activity objects
        """
        activities = []
        
        for _, row in df.iterrows():
            try:
                activity = Activity(
                    name=str(row['Activity']),
                    emission_kg_per_day=float(row['Avg_CO2_Emission(kg/day)']),
                    category=Category(row['Category'])
                )
                activities.append(activity)
            except (ValueError, KeyError) as e:
                # Skip invalid rows
                continue
        
        return activities
    
    def export_analysis_summary(self, analysis: DatasetAnalysis) -> str:
        """
        Generate a text summary of the analysis.
        
        Args:
            analysis: DatasetAnalysis object
            
        Returns:
            Formatted text summary
        """
        summary_lines = []
        summary_lines.append("=== CO2 Emission Analysis Summary ===\n")
        
        summary_lines.append(f"Total Daily Emission: {analysis.total_daily_emission:.2f} kg CO2/day")
        summary_lines.append(f"Total Annual Emission: {analysis.total_annual_emission:.2f} kg CO2/year")
        summary_lines.append(f"Annual Emission (tons): {analysis.total_annual_emission / 1000:.2f} tons\n")
        
        summary_lines.append("Top Emitters:")
        for i, activity in enumerate(analysis.top_emitters, 1):
            summary_lines.append(
                f"  {i}. {activity.name}: {activity.emission_kg_per_day:.2f} kg CO2/day "
                f"({activity.category.value})"
            )
        
        summary_lines.append("\nCategory Breakdown:")
        for category, emission in sorted(
            analysis.category_breakdown.items(), 
            key=lambda x: x[1], 
            reverse=True
        ):
            percentage = (emission / analysis.total_daily_emission * 100) if analysis.total_daily_emission > 0 else 0
            summary_lines.append(f"  {category}: {emission:.2f} kg CO2/day ({percentage:.1f}%)")
        
        return "\n".join(summary_lines)
