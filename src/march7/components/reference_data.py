"""
Reference data manager for comprehensive environmental impact activities.

This module provides functionality to load and query the reference dataset
of activities and their associated environmental metrics (CO2, water, energy, waste, etc.).
"""

import pandas as pd
from typing import Optional, List, Dict
from pathlib import Path
from difflib import get_close_matches
from models.data_models import Activity, Category, EnvironmentalMetrics


class ReferenceDataManager:
    """Manages the reference dataset of activities and their environmental impacts."""

    def __init__(self, filepath: str = "data/reference_activities.csv"):
        """
        Initialize the ReferenceDataManager.

        Args:
            filepath: Path to the reference activities CSV file
        """
        self.filepath = filepath
        self.data: Optional[pd.DataFrame] = None
        self._activity_lookup: dict = {}

    def load_reference_data(self) -> pd.DataFrame:
        """
        Load the reference dataset from CSV file.

        Returns:
            DataFrame containing the reference activities

        Raises:
            FileNotFoundError: If the CSV file doesn't exist
            ValueError: If the CSV file has invalid structure
        """
        if not Path(self.filepath).exists():
            raise FileNotFoundError(f"Reference data file not found: {self.filepath}")

        try:
            self.data = pd.read_csv(self.filepath)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {str(e)}")

        # Validate required columns
        required_columns = ["Activity", "Avg_CO2_Emission(kg/day)", "Category"]
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Create lookup dictionary for faster access with all environmental metrics
        self._activity_lookup = {}
        for _, row in self.data.iterrows():
            activity_key = row["Activity"].lower()
            self._activity_lookup[activity_key] = {
                "emission": row["Avg_CO2_Emission(kg/day)"],
                "category": row["Category"],
                "water": row.get("Water_Usage(L/day)", 0.0),
                "energy": row.get("Energy_Usage(kWh/day)", 0.0),
                "waste": row.get("Waste_Generation(kg/day)", 0.0),
                "land_use": row.get("Land_Use(m2)", 0.0),
                "pollution_index": row.get("Pollution_Index", 0)
            }

        return self.data

    def get_activity_emission(self, activity: str, fuzzy_match: bool = True) -> Optional[float]:
        """
        Get the CO2 emission for a specific activity.

        Args:
            activity: Name of the activity to look up
            fuzzy_match: If True, use fuzzy matching to find similar activities

        Returns:
            CO2 emission in kg/day, or None if not found
        """
        if self.data is None:
            self.load_reference_data()

        activity_lower = activity.lower()

        # Try exact match first
        if activity_lower in self._activity_lookup:
            return self._activity_lookup[activity_lower]["emission"]

        # Try fuzzy matching if enabled
        if fuzzy_match:
            activity_names = list(self._activity_lookup.keys())
            matches = get_close_matches(activity_lower, activity_names, n=1, cutoff=0.6)
            if matches:
                return self._activity_lookup[matches[0]]["emission"]

        return None
    
    def get_activity_environmental_metrics(self, activity: str, fuzzy_match: bool = True) -> Optional[EnvironmentalMetrics]:
        """
        Get comprehensive environmental metrics for a specific activity.

        Args:
            activity: Name of the activity to look up
            fuzzy_match: If True, use fuzzy matching to find similar activities

        Returns:
            EnvironmentalMetrics object with all metrics, or None if not found
        """
        if self.data is None:
            self.load_reference_data()

        activity_lower = activity.lower()
        activity_data = None

        # Try exact match first
        if activity_lower in self._activity_lookup:
            activity_data = self._activity_lookup[activity_lower]
        # Try fuzzy matching if enabled
        elif fuzzy_match:
            activity_names = list(self._activity_lookup.keys())
            matches = get_close_matches(activity_lower, activity_names, n=1, cutoff=0.6)
            if matches:
                activity_data = self._activity_lookup[matches[0]]

        if activity_data:
            return EnvironmentalMetrics(
                co2_kg_per_day=activity_data["emission"],
                water_liters_per_day=activity_data["water"],
                energy_kwh_per_day=activity_data["energy"],
                waste_kg_per_day=activity_data["waste"],
                land_use_m2=activity_data["land_use"],
                pollution_index=activity_data["pollution_index"]
            )

        return None

    def get_activities_by_category(self, category: Category) -> List[Activity]:
        """
        Get all activities for a specific category.

        Args:
            category: Category to filter by

        Returns:
            List of Activity objects in the specified category
        """
        if self.data is None:
            self.load_reference_data()

        category_data = self.data[self.data["Category"] == category.value]

        activities = []
        for _, row in category_data.iterrows():
            try:
                activity = Activity(
                    name=row["Activity"],
                    emission_kg_per_day=row["Avg_CO2_Emission(kg/day)"],
                    category=Category(row["Category"])
                )
                activities.append(activity)
            except Exception:
                # Skip invalid rows
                continue

        return activities

    def search_similar_activities(self, query: str, n: int = 5, cutoff: float = 0.6) -> List[Activity]:
        """
        Search for activities similar to the query string.

        Args:
            query: Search query
            n: Maximum number of results to return
            cutoff: Similarity threshold (0-1)

        Returns:
            List of similar Activity objects
        """
        if self.data is None:
            self.load_reference_data()

        query_lower = query.lower()
        activity_names = list(self._activity_lookup.keys())
        
        # Strategy 1: Try keyword-based matching first (more reliable)
        keyword_matches = []
        query_words = set(query_lower.split())
        
        for activity_name in activity_names:
            activity_words = set(activity_name.split())
            # Count how many words match
            common_words = query_words & activity_words
            if common_words:
                # Calculate a simple score based on word overlap
                score = len(common_words) / max(len(query_words), len(activity_words))
                keyword_matches.append((activity_name, score))
        
        # Sort by score and take top matches
        keyword_matches.sort(key=lambda x: x[1], reverse=True)
        matches = [match[0] for match in keyword_matches[:n] if match[1] > 0.2]
        
        # Strategy 2: If no keyword matches, fall back to fuzzy matching
        if not matches:
            matches = get_close_matches(query_lower, activity_names, n=n, cutoff=cutoff)

        activities = []
        for match in matches:
            activity_data = self._activity_lookup[match]
            try:
                # Find the original activity name (with proper casing)
                original_name = self.data[
                    self.data["Activity"].str.lower() == match
                ]["Activity"].iloc[0]

                activity = Activity(
                    name=original_name,
                    emission_kg_per_day=activity_data["emission"],
                    category=Category(activity_data["category"])
                )
                activities.append(activity)
            except Exception:
                # Skip invalid entries
                continue

        return activities

    def get_all_activities(self) -> List[Activity]:
        """
        Get all activities from the reference dataset.

        Returns:
            List of all Activity objects
        """
        if self.data is None:
            self.load_reference_data()

        activities = []
        for _, row in self.data.iterrows():
            try:
                activity = Activity(
                    name=row["Activity"],
                    emission_kg_per_day=row["Avg_CO2_Emission(kg/day)"],
                    category=Category(row["Category"])
                )
                activities.append(activity)
            except Exception:
                # Skip invalid rows
                continue

        return activities

    def get_category_statistics(self) -> dict:
        """
        Get statistics about emissions by category.

        Returns:
            Dictionary with category statistics
        """
        if self.data is None:
            self.load_reference_data()

        stats = {}
        for category in Category:
            category_data = self.data[self.data["Category"] == category.value]
            if not category_data.empty:
                stats[category.value] = {
                    "count": len(category_data),
                    "mean_emission": category_data["Avg_CO2_Emission(kg/day)"].mean(),
                    "median_emission": category_data["Avg_CO2_Emission(kg/day)"].median(),
                    "min_emission": category_data["Avg_CO2_Emission(kg/day)"].min(),
                    "max_emission": category_data["Avg_CO2_Emission(kg/day)"].max()
                }

        return stats
