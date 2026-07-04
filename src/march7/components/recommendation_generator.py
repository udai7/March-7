"""
Recommendation generator for creating CO2 reduction suggestions.

This module generates, ranks, and formats recommendations for reducing
CO2 emissions based on activities and retrieved knowledge.
"""

from typing import List, Dict, Any, Optional
from models.data_models import Activity, Recommendation, Category
from components.reference_data import ReferenceDataManager
from components.emission_calculator import EmissionCalculator


class RecommendationGenerator:
    """Generates and ranks CO2 reduction recommendations."""
    
    def __init__(
        self,
        reference_data_manager: Optional[ReferenceDataManager] = None
    ):
        """
        Initialize the RecommendationGenerator.
        
        Args:
            reference_data_manager: Optional ReferenceDataManager instance
        """
        self.reference_manager = reference_data_manager or ReferenceDataManager()
        self.calculator = EmissionCalculator()
        
        # Ensure reference data is loaded
        if self.reference_manager.data is None:
            self.reference_manager.load_reference_data()
    
    def generate_alternatives(
        self,
        activity: Activity,
        max_alternatives: int = 5
    ) -> List[Activity]:
        """
        Find lower-emission alternatives for a given activity.
        
        Args:
            activity: Current activity to find alternatives for
            max_alternatives: Maximum number of alternatives to return
            
        Returns:
            List of alternative Activity objects with lower emissions
        """
        alternatives = []
        
        # Get all activities in the same category
        category_activities = self.reference_manager.get_activities_by_category(
            activity.category
        )
        
        # Filter for activities with lower emissions
        for alt_activity in category_activities:
            if alt_activity.emission_kg_per_day < activity.emission_kg_per_day:
                # Avoid suggesting the same activity
                if alt_activity.name.lower() != activity.name.lower():
                    alternatives.append(alt_activity)
        
        # Sort by emission (lowest first)
        alternatives.sort(key=lambda x: x.emission_kg_per_day)
        
        return alternatives[:max_alternatives]
    
    def rank_recommendations(
        self,
        current_activity: Activity,
        alternatives: List[Activity]
    ) -> List[Recommendation]:
        """
        Rank alternatives by emission reduction potential.
        
        Args:
            current_activity: Current activity
            alternatives: List of alternative activities
            
        Returns:
            List of Recommendation objects ranked by reduction potential
        """
        recommendations = []
        
        for alternative in alternatives:
            # Calculate reduction metrics
            metrics = self.calculator.calculate_reduction(
                current_activity.emission_kg_per_day,
                alternative.emission_kg_per_day
            )
            
            # Determine implementation difficulty based on emission difference
            difficulty = self._determine_difficulty(
                current_activity,
                alternative,
                metrics.percentage_reduction
            )
            
            # Determine timeframe
            timeframe = self._determine_timeframe(
                current_activity,
                alternative
            )
            
            # Generate additional benefits
            benefits = self._generate_benefits(
                current_activity.category,
                alternative
            )
            
            # Create recommendation
            recommendation = Recommendation(
                action=f"Switch from {current_activity.name} to {alternative.name}",
                emission_reduction_kg=metrics.absolute_reduction,
                reduction_percentage=metrics.percentage_reduction,
                implementation_difficulty=difficulty,
                timeframe=timeframe,
                additional_benefits=benefits
            )
            
            recommendations.append(recommendation)
        
        # Sort by emission reduction (highest first)
        recommendations.sort(
            key=lambda x: x.emission_reduction_kg,
            reverse=True
        )
        
        return recommendations
    
    def format_recommendations(
        self,
        recommendations: List[Recommendation],
        include_details: bool = True
    ) -> str:
        """
        Format recommendations into human-readable text.
        
        Args:
            recommendations: List of Recommendation objects
            include_details: Whether to include detailed information
            
        Returns:
            Formatted text string
        """
        if not recommendations:
            return "No recommendations available."
        
        lines = []
        lines.append("=== CO2 Reduction Recommendations ===\n")
        
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec.action}")
            lines.append(f"   Reduction: {rec.emission_reduction_kg:.2f} kg CO2/day ({rec.reduction_percentage:.1f}%)")
            lines.append(f"   Annual Savings: {rec.emission_reduction_kg * 365:.2f} kg CO2/year")
            
            if include_details:
                lines.append(f"   Difficulty: {rec.implementation_difficulty}")
                lines.append(f"   Timeframe: {rec.timeframe}")
                
                if rec.additional_benefits:
                    lines.append(f"   Benefits: {', '.join(rec.additional_benefits)}")
            
            lines.append("")  # Empty line between recommendations
        
        return "\n".join(lines)
    
    def generate_recommendations_for_activity(
        self,
        activity: Activity,
        max_recommendations: int = 3
    ) -> List[Recommendation]:
        """
        Generate complete recommendations for a single activity.
        
        Args:
            activity: Activity to generate recommendations for
            max_recommendations: Maximum number of recommendations
            
        Returns:
            List of ranked Recommendation objects
        """
        # Find alternatives
        alternatives = self.generate_alternatives(activity, max_alternatives=max_recommendations)
        
        if not alternatives:
            return []
        
        # Rank and create recommendations
        recommendations = self.rank_recommendations(activity, alternatives)
        
        return recommendations[:max_recommendations]
    
    def generate_recommendations_for_multiple(
        self,
        activities: List[Activity],
        max_per_activity: int = 2
    ) -> List[Recommendation]:
        """
        Generate recommendations for multiple activities.
        
        Args:
            activities: List of activities
            max_per_activity: Maximum recommendations per activity
            
        Returns:
            Combined list of recommendations, prioritized by impact
        """
        all_recommendations = []
        
        for activity in activities:
            recs = self.generate_recommendations_for_activity(
                activity,
                max_recommendations=max_per_activity
            )
            all_recommendations.extend(recs)
        
        # Sort all recommendations by emission reduction
        all_recommendations.sort(
            key=lambda x: x.emission_reduction_kg,
            reverse=True
        )
        
        return all_recommendations
    
    def _determine_difficulty(
        self,
        current: Activity,
        alternative: Activity,
        reduction_percentage: float
    ) -> str:
        """
        Determine implementation difficulty based on activity change.
        
        Args:
            current: Current activity
            alternative: Alternative activity
            reduction_percentage: Percentage reduction
            
        Returns:
            Difficulty level: "Easy", "Medium", or "Hard"
        """
        # Transport category logic
        if current.category == Category.TRANSPORT:
            if "walk" in alternative.name.lower() or "bicycle" in alternative.name.lower():
                return "Medium"  # Requires behavior change
            elif "public" in alternative.name.lower() or "bus" in alternative.name.lower():
                return "Easy"  # Usually accessible
            elif "electric" in alternative.name.lower():
                return "Hard"  # Requires investment
        
        # Household category logic
        elif current.category == Category.HOUSEHOLD:
            if reduction_percentage > 50:
                return "Hard"  # Major changes needed
            elif reduction_percentage > 25:
                return "Medium"
            else:
                return "Easy"
        
        # Food category logic
        elif current.category == Category.FOOD:
            if "vegan" in alternative.name.lower():
                return "Medium"  # Dietary change
            else:
                return "Easy"
        
        # Default based on reduction percentage
        if reduction_percentage > 50:
            return "Medium"
        else:
            return "Easy"
    
    def _determine_timeframe(
        self,
        current: Activity,
        alternative: Activity
    ) -> str:
        """
        Determine implementation timeframe.
        
        Args:
            current: Current activity
            alternative: Alternative activity
            
        Returns:
            Timeframe: "Immediate", "Short-term", or "Long-term"
        """
        # Check if it requires major investment or infrastructure
        if "electric" in alternative.name.lower() or "solar" in alternative.name.lower():
            return "Long-term"
        
        # Check if it's a simple behavior change
        if current.category == Category.TRANSPORT:
            if "walk" in alternative.name.lower() or "bicycle" in alternative.name.lower():
                return "Immediate"
            elif "public" in alternative.name.lower():
                return "Immediate"
        
        if current.category == Category.FOOD:
            return "Immediate"  # Diet changes can start immediately
        
        if current.category == Category.LIFESTYLE:
            return "Immediate"  # Lifestyle changes are usually immediate
        
        # Default to short-term
        return "Short-term"
    
    def _generate_benefits(
        self,
        category: Category,
        alternative: Activity
    ) -> List[str]:
        """
        Generate additional benefits beyond CO2 reduction.
        
        Args:
            category: Activity category
            alternative: Alternative activity
            
        Returns:
            List of benefit strings
        """
        benefits = []
        alt_name_lower = alternative.name.lower()
        
        # Transport benefits
        if category == Category.TRANSPORT:
            if "walk" in alt_name_lower or "bicycle" in alt_name_lower:
                benefits.extend(["Improved health and fitness", "Cost savings on fuel"])
            elif "public" in alt_name_lower or "bus" in alt_name_lower or "train" in alt_name_lower:
                benefits.extend(["Cost savings", "Reduced traffic stress"])
            elif "electric" in alt_name_lower:
                benefits.extend(["Lower operating costs", "Quieter operation"])
        
        # Household benefits
        elif category == Category.HOUSEHOLD:
            if "efficient" in alt_name_lower or "led" in alt_name_lower:
                benefits.extend(["Lower energy bills", "Longer lifespan"])
            elif "solar" in alt_name_lower:
                benefits.extend(["Energy independence", "Long-term savings"])
        
        # Food benefits
        elif category == Category.FOOD:
            if "vegetarian" in alt_name_lower or "vegan" in alt_name_lower:
                benefits.extend(["Health benefits", "Lower food costs"])
            elif "local" in alt_name_lower:
                benefits.extend(["Fresher produce", "Support local economy"])
        
        # Lifestyle benefits
        elif category == Category.LIFESTYLE:
            if "recycle" in alt_name_lower:
                benefits.extend(["Reduced waste", "Environmental stewardship"])
        
        return benefits
    
    def create_custom_recommendation(
        self,
        action: str,
        current_emission: float,
        alternative_emission: float,
        difficulty: str = "Medium",
        timeframe: str = "Short-term",
        benefits: Optional[List[str]] = None
    ) -> Recommendation:
        """
        Create a custom recommendation with specified parameters.
        
        Args:
            action: Recommended action description
            current_emission: Current emission in kg/day
            alternative_emission: Alternative emission in kg/day
            difficulty: Implementation difficulty
            timeframe: Implementation timeframe
            benefits: Additional benefits
            
        Returns:
            Recommendation object
        """
        metrics = self.calculator.calculate_reduction(
            current_emission,
            alternative_emission
        )
        
        return Recommendation(
            action=action,
            emission_reduction_kg=metrics.absolute_reduction,
            reduction_percentage=metrics.percentage_reduction,
            implementation_difficulty=difficulty,
            timeframe=timeframe,
            additional_benefits=benefits or []
        )
