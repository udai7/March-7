"""
Emission calculator for CO2 reduction calculations.

This module provides functionality to calculate daily and annual emissions,
as well as emission reductions from alternative activities.
"""

from typing import List
from models.data_models import Activity, ReductionMetrics


class EmissionCalculator:
    """Calculates CO2 emissions and reductions."""

    DAYS_PER_YEAR = 365

    def __init__(self):
        """Initialize the EmissionCalculator."""
        pass

    def calculate_daily_emission(self, activities: List[Activity]) -> float:
        """
        Calculate total daily CO2 emission from a list of activities.

        Args:
            activities: List of Activity objects

        Returns:
            Total daily CO2 emission in kg/day
        """
        if not activities:
            return 0.0

        total_emission = sum(activity.emission_kg_per_day for activity in activities)
        return round(total_emission, 2)

    def calculate_annual_emission(self, daily_emission: float) -> float:
        """
        Calculate annual CO2 emission from daily emission.

        Args:
            daily_emission: Daily CO2 emission in kg/day

        Returns:
            Annual CO2 emission in kg/year
        """
        annual_emission = daily_emission * self.DAYS_PER_YEAR
        return round(annual_emission, 2)

    def calculate_reduction(
        self,
        current_emission: float,
        alternative_emission: float
    ) -> ReductionMetrics:
        """
        Calculate emission reduction metrics when switching from current to alternative.

        Args:
            current_emission: Current CO2 emission in kg/day
            alternative_emission: Alternative CO2 emission in kg/day

        Returns:
            ReductionMetrics object with detailed reduction information
        """
        # Calculate absolute reduction
        absolute_reduction = current_emission - alternative_emission

        # Calculate percentage reduction
        if current_emission > 0:
            percentage_reduction = (absolute_reduction / current_emission) * 100
        else:
            percentage_reduction = 0.0

        # Calculate annual savings
        annual_savings = absolute_reduction * self.DAYS_PER_YEAR

        return ReductionMetrics(
            current_emission=round(current_emission, 2),
            alternative_emission=round(alternative_emission, 2),
            absolute_reduction=round(absolute_reduction, 2),
            percentage_reduction=round(percentage_reduction, 2),
            annual_savings=round(annual_savings, 2)
        )

    def calculate_multiple_reductions(
        self,
        current_emission: float,
        alternatives: List[Activity]
    ) -> List[ReductionMetrics]:
        """
        Calculate reduction metrics for multiple alternative activities.

        Args:
            current_emission: Current CO2 emission in kg/day
            alternatives: List of alternative Activity objects

        Returns:
            List of ReductionMetrics, sorted by reduction potential (highest first)
        """
        reductions = []

        for alternative in alternatives:
            metrics = self.calculate_reduction(
                current_emission,
                alternative.emission_kg_per_day
            )
            reductions.append(metrics)

        # Sort by absolute reduction (descending)
        reductions.sort(key=lambda x: x.absolute_reduction, reverse=True)

        return reductions

    def calculate_category_emissions(
        self,
        activities: List[Activity]
    ) -> dict[str, float]:
        """
        Calculate total emissions grouped by category.

        Args:
            activities: List of Activity objects

        Returns:
            Dictionary mapping category names to total emissions
        """
        category_emissions = {}

        for activity in activities:
            category_name = activity.category.value
            if category_name not in category_emissions:
                category_emissions[category_name] = 0.0
            category_emissions[category_name] += activity.emission_kg_per_day

        # Round all values
        for category in category_emissions:
            category_emissions[category] = round(category_emissions[category], 2)

        return category_emissions

    def get_top_emitters(
        self,
        activities: List[Activity],
        n: int = 3
    ) -> List[Activity]:
        """
        Get the top N highest emission activities.

        Args:
            activities: List of Activity objects
            n: Number of top emitters to return

        Returns:
            List of top N Activity objects sorted by emission (highest first)
        """
        if not activities:
            return []

        # Sort by emission (descending)
        sorted_activities = sorted(
            activities,
            key=lambda x: x.emission_kg_per_day,
            reverse=True
        )

        return sorted_activities[:n]

    def calculate_potential_savings(
        self,
        current_activities: List[Activity],
        alternative_activities: List[Activity]
    ) -> ReductionMetrics:
        """
        Calculate potential savings when switching from current to alternative activities.

        Args:
            current_activities: List of current Activity objects
            alternative_activities: List of alternative Activity objects

        Returns:
            ReductionMetrics showing total potential savings
        """
        current_total = self.calculate_daily_emission(current_activities)
        alternative_total = self.calculate_daily_emission(alternative_activities)

        return self.calculate_reduction(current_total, alternative_total)

    def convert_to_tons(self, kg_emission: float) -> float:
        """
        Convert kg CO2 to metric tons.

        Args:
            kg_emission: Emission in kg

        Returns:
            Emission in metric tons
        """
        return round(kg_emission / 1000, 3)

    def convert_to_trees_equivalent(self, annual_kg: float) -> int:
        """
        Convert annual CO2 emission to equivalent number of trees needed to offset.
        
        Assumes one tree absorbs approximately 21 kg of CO2 per year.

        Args:
            annual_kg: Annual CO2 emission in kg

        Returns:
            Number of trees needed to offset the emission
        """
        KG_PER_TREE_PER_YEAR = 21
        return int(annual_kg / KG_PER_TREE_PER_YEAR)

    def get_emission_summary(self, activities: List[Activity]) -> dict:
        """
        Get a comprehensive summary of emissions.

        Args:
            activities: List of Activity objects

        Returns:
            Dictionary with emission summary statistics
        """
        if not activities:
            return {
                "total_daily_kg": 0.0,
                "total_annual_kg": 0.0,
                "total_annual_tons": 0.0,
                "trees_to_offset": 0,
                "category_breakdown": {},
                "top_emitters": []
            }

        daily_emission = self.calculate_daily_emission(activities)
        annual_emission = self.calculate_annual_emission(daily_emission)

        return {
            "total_daily_kg": daily_emission,
            "total_annual_kg": annual_emission,
            "total_annual_tons": self.convert_to_tons(annual_emission),
            "trees_to_offset": self.convert_to_trees_equivalent(annual_emission),
            "category_breakdown": self.calculate_category_emissions(activities),
            "top_emitters": self.get_top_emitters(activities, n=3)
        }
