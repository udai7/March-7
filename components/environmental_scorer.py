"""
Environmental scoring and grading system.

This module calculates comprehensive environmental impact scores
and assigns sustainability grades based on multiple metrics.
"""

from typing import Dict, Tuple
import config


class EnvironmentalScorer:
    """Calculate environmental impact scores and grades."""
    
    def __init__(self):
        """Initialize the environmental scorer with grading thresholds."""
        self.thresholds = config.GRADE_THRESHOLDS
    
    def calculate_environmental_score(
        self,
        co2: float,
        water: float,
        energy: float,
        waste: float,
        pollution_index: float = 0
    ) -> float:
        """
        Calculate a composite environmental impact score (0-100).
        Lower scores are better (less environmental impact).
        
        Args:
            co2: CO2 emissions in kg/day
            water: Water usage in liters/day
            energy: Energy usage in kWh/day
            waste: Waste generation in kg/day
            pollution_index: Pollution index (0-100)
            
        Returns:
            Environmental impact score (0-100)
        """
        # Weight factors for each metric
        weights = {
            "co2": 0.35,
            "water": 0.20,
            "energy": 0.25,
            "waste": 0.15,
            "pollution": 0.05
        }
        
        # Normalize each metric to 0-100 scale based on thresholds
        co2_score = min((co2 / self.thresholds["F"]["co2"]) * 100, 100) if co2 > 0 else 0
        water_score = min((water / self.thresholds["F"]["water"]) * 100, 100) if water > 0 else 0
        energy_score = min((energy / self.thresholds["F"]["energy"]) * 100, 100) if energy > 0 else 0
        waste_score = min((waste / self.thresholds["F"]["waste"]) * 100, 100) if waste > 0 else 0
        pollution_score = pollution_index  # Already 0-100
        
        # Calculate weighted score
        total_score = (
            weights["co2"] * co2_score +
            weights["water"] * water_score +
            weights["energy"] * energy_score +
            weights["waste"] * waste_score +
            weights["pollution"] * pollution_score
        )
        
        return round(total_score, 1)
    
    def assign_sustainability_grade(
        self,
        co2: float,
        water: float,
        energy: float,
        waste: float
    ) -> str:
        """
        Assign a sustainability grade (A+ to F) based on environmental metrics.
        
        Args:
            co2: CO2 emissions in kg/day
            water: Water usage in liters/day
            energy: Energy usage in kWh/day
            waste: Waste generation in kg/day
            
        Returns:
            Sustainability grade (A+, A, B, C, D, or F)
        """
        # Check each grade from best to worst
        for grade in ["A+", "A", "B", "C", "D", "F"]:
            thresholds = self.thresholds[grade]
            
            # All metrics must be below threshold for this grade
            if (co2 <= thresholds["co2"] and
                water <= thresholds["water"] and
                energy <= thresholds["energy"] and
                waste <= thresholds["waste"]):
                return grade
        
        return "F"
    
    def get_grade_description(self, grade: str) -> str:
        """
        Get a description of what a sustainability grade means.
        
        Args:
            grade: Sustainability grade (A+ to F)
            
        Returns:
            Description of the grade
        """
        descriptions = {
            "A+": "Exceptional - Well below average impact across all metrics. Outstanding environmental stewardship!",
            "A": "Excellent - Significantly below average impact. Great job maintaining sustainable practices!",
            "B": "Good - Below average impact. You're on the right track with room for improvement.",
            "C": "Average - Typical environmental impact. Many opportunities for meaningful reductions.",
            "D": "Fair - Above average impact. Consider prioritizing high-impact reduction strategies.",
            "F": "Poor - Significantly above average impact. Immediate action recommended across multiple areas."
        }
        
        return descriptions.get(grade, "Unknown grade")
    
    def get_improvement_path(
        self,
        current_grade: str,
        target_grade: str,
        current_metrics: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate what improvements are needed to reach target grade.
        
        Args:
            current_grade: Current sustainability grade
            target_grade: Desired sustainability grade
            current_metrics: Dict with current co2, water, energy, waste values
            
        Returns:
            Dict showing required reductions for each metric
        """
        target_thresholds = self.thresholds[target_grade]
        improvements = {}
        
        for metric in ["co2", "water", "energy", "waste"]:
            current_value = current_metrics.get(metric, 0)
            target_value = target_thresholds[metric]
            
            if current_value > target_value:
                reduction_needed = current_value - target_value
                reduction_percentage = (reduction_needed / current_value) * 100
                
                improvements[metric] = {
                    "current": current_value,
                    "target": target_value,
                    "reduction_needed": reduction_needed,
                    "reduction_percentage": reduction_percentage
                }
            else:
                improvements[metric] = {
                    "current": current_value,
                    "target": target_value,
                    "reduction_needed": 0,
                    "reduction_percentage": 0
                }
        
        return improvements
    
    def compare_activities(
        self,
        activity1_metrics: Dict[str, float],
        activity2_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compare environmental impact between two activities.
        
        Args:
            activity1_metrics: Metrics for first activity
            activity2_metrics: Metrics for second activity
            
        Returns:
            Dict showing percentage differences for each metric
        """
        comparison = {}
        
        for metric in ["co2", "water", "energy", "waste"]:
            val1 = activity1_metrics.get(metric, 0)
            val2 = activity2_metrics.get(metric, 0)
            
            if val1 > 0:
                percentage_diff = ((val2 - val1) / val1) * 100
                comparison[metric] = round(percentage_diff, 1)
            else:
                comparison[metric] = 0 if val2 == 0 else 100
        
        return comparison
    
    def get_metric_summary(self, value: float, metric_type: str) -> str:
        """
        Get a human-readable summary for a metric value.
        
        Args:
            value: Metric value
            metric_type: Type of metric (co2, water, energy, waste)
            
        Returns:
            Human-readable summary with context
        """
        equivalents = {
            "co2": [
                (1, "driving 5 km in a gasoline car"),
                (5, "charging 250 smartphones"),
                (10, "a short flight (100 km)"),
                (100, "a long international flight")
            ],
            "water": [
                (10, "flushing a toilet twice"),
                (50, "a 10-minute shower"),
                (100, "running a dishwasher"),
                (1000, "1 kg of beef production")
            ],
            "energy": [
                (1, "watching TV for 3 hours"),
                (3, "running a refrigerator for a day"),
                (10, "using central AC for 8 hours"),
                (50, "powering a home for 2 days")
            ],
            "waste": [
                (0.1, "one coffee cup"),
                (0.5, "a day's worth of food scraps"),
                (1, "a small trash bag"),
                (5, "a week's worth of household waste")
            ]
        }
        
        if metric_type in equivalents:
            for threshold, equivalent in reversed(equivalents[metric_type]):
                if value >= threshold:
                    return f"~{equivalent}"
        
        return "minimal impact"
