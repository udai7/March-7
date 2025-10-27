"""
Multi-factor recommendation ranking system.

Ranks recommendations based on multiple weighted criteria including
emission reduction, cost-effectiveness, ease of implementation, and co-benefits.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RankingWeights:
    """
    Configurable weights for recommendation scoring.
    
    All weights should sum to 1.0 for normalized scoring.
    """
    emission_reduction: float = 0.35  # Primary goal
    cost_effectiveness: float = 0.25  # Affordability
    ease_of_implementation: float = 0.20  # Practicality
    time_to_impact: float = 0.15  # Quick wins
    co_benefits: float = 0.05  # Additional benefits
    
    def __post_init__(self):
        """Validate weights sum to approximately 1.0."""
        total = (self.emission_reduction + self.cost_effectiveness + 
                self.ease_of_implementation + self.time_to_impact + 
                self.co_benefits)
        
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


class RecommendationRanker:
    """
    Ranks recommendations using multi-factor weighted scoring.
    
    Combines emission reduction, cost, difficulty, timeframe, and
    co-benefits to produce a comprehensive relevance score.
    """
    
    def __init__(self, weights: Optional[RankingWeights] = None):
        """
        Initialize the recommendation ranker.
        
        Args:
            weights: Optional custom ranking weights
        """
        self.weights = weights or RankingWeights()
    
    def rank_recommendations(
        self,
        recommendations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rank recommendations by calculating weighted scores.
        
        Args:
            recommendations: List of recommendation dictionaries
            
        Returns:
            Sorted list of recommendations with scores added
        """
        if not recommendations:
            return []
        
        # Calculate score for each recommendation
        for rec in recommendations:
            rec['ranking_score'] = self.score_recommendation(rec)
            rec['score_breakdown'] = self._get_score_breakdown(rec)
        
        # Sort by score (highest first)
        ranked = sorted(
            recommendations,
            key=lambda x: x.get('ranking_score', 0),
            reverse=True
        )
        
        return ranked
    
    def score_recommendation(self, recommendation: Dict[str, Any]) -> float:
        """
        Calculate weighted score for a single recommendation.
        
        Args:
            recommendation: Single recommendation dictionary
            
        Returns:
            Score from 0 to 1
        """
        # Calculate individual component scores
        emission_score = self._calculate_emission_score(recommendation)
        cost_score = self._calculate_cost_effectiveness_score(recommendation)
        ease_score = self._calculate_ease_score(recommendation)
        time_score = self._calculate_time_score(recommendation)
        coben_score = self._calculate_cobenefit_score(recommendation)
        
        # Weighted sum
        total_score = (
            emission_score * self.weights.emission_reduction +
            cost_score * self.weights.cost_effectiveness +
            ease_score * self.weights.ease_of_implementation +
            time_score * self.weights.time_to_impact +
            coben_score * self.weights.co_benefits
        )
        
        return min(max(total_score, 0.0), 1.0)  # Clamp to [0, 1]
    
    def _calculate_emission_score(self, rec: Dict[str, Any]) -> float:
        """
        Calculate emission reduction score (0-1).
        
        Higher percentage reduction = higher score.
        """
        reduction_pct = rec.get('reduction_percentage', 0)
        
        # Normalize to 0-1 scale
        # 100% reduction = 1.0, 0% = 0.0
        score = min(reduction_pct / 100.0, 1.0)
        
        # Also consider absolute savings
        annual_savings = rec.get('annual_savings_kg', 0)
        
        # Bonus for high absolute savings (>500 kg/year)
        if annual_savings > 500:
            score = min(score + 0.1, 1.0)
        if annual_savings > 1000:
            score = min(score + 0.1, 1.0)
        
        return score
    
    def _calculate_cost_effectiveness_score(self, rec: Dict[str, Any]) -> float:
        """
        Calculate cost-effectiveness score (0-1).
        
        Lower cost per kg CO2 saved = higher score.
        """
        # Get cost category
        cost_category = rec.get('cost_category', 'medium').lower()
        annual_savings = rec.get('annual_savings_kg', 1)  # Avoid division by zero
        
        # Estimate annual cost from category
        cost_estimates = {
            'low': 100,      # $100/year or less
            'medium': 500,   # $500/year
            'high': 2000     # $2000/year
        }
        
        estimated_cost = cost_estimates.get(cost_category, 500)
        
        # Calculate cost per kg CO2 saved
        cost_per_kg = estimated_cost / max(annual_savings, 1)
        
        # Normalize using sigmoid-like function
        # $0.10/kg = excellent, $1/kg = good, $10/kg = poor
        score = 1.0 / (1.0 + cost_per_kg / 1.0)
        
        # Bonus for no-cost or negative-cost (money-saving) options
        if cost_category == 'low':
            score = min(score + 0.2, 1.0)
        
        # Check for savings in co-benefits
        co_benefits = rec.get('co_benefits', '').lower()
        if 'savings' in co_benefits or 'save money' in co_benefits:
            score = min(score + 0.15, 1.0)
        
        return score
    
    def _calculate_ease_score(self, rec: Dict[str, Any]) -> float:
        """
        Calculate ease of implementation score (0-1).
        
        Easier implementation = higher score.
        """
        difficulty = rec.get('difficulty', 'Medium').lower()
        
        ease_map = {
            'easy': 1.0,
            'medium': 0.6,
            'hard': 0.3
        }
        
        score = ease_map.get(difficulty, 0.5)
        
        # Check prerequisites
        prerequisites = rec.get('prerequisites', '').lower()
        
        # Penalty for complex prerequisites
        if len(prerequisites) > 100:  # Long prerequisite list
            score *= 0.9
        
        if any(word in prerequisites for word in 
               ['approval', 'permission', 'contractor', 'professional']):
            score *= 0.85
        
        return score
    
    def _calculate_time_score(self, rec: Dict[str, Any]) -> float:
        """
        Calculate time-to-impact score (0-1).
        
        Faster impact = higher score.
        """
        timeframe = rec.get('timeframe', 'medium').lower()
        
        time_map = {
            'immediate': 1.0,
            'short-term': 0.7,
            'short term': 0.7,
            'medium-term': 0.5,
            'medium term': 0.5,
            'long-term': 0.4,
            'long term': 0.4
        }
        
        # Try exact match first
        score = time_map.get(timeframe, 0.5)
        
        # If no exact match, check for partial matches
        if score == 0.5:
            if 'immediate' in timeframe:
                score = 1.0
            elif 'short' in timeframe:
                score = 0.7
            elif 'long' in timeframe:
                score = 0.4
        
        return score
    
    def _calculate_cobenefit_score(self, rec: Dict[str, Any]) -> float:
        """
        Calculate co-benefits score (0-1).
        
        More benefits = higher score.
        """
        co_benefits = rec.get('co_benefits', '')
        
        if not co_benefits:
            return 0.2  # Minimal score if no co-benefits listed
        
        co_benefits_lower = co_benefits.lower()
        
        # Count different types of benefits
        benefit_keywords = {
            'health': ['health', 'fitness', 'exercise', 'mental'],
            'financial': ['savings', 'save money', 'cheaper', 'cost'],
            'time': ['time saving', 'faster', 'convenient'],
            'social': ['community', 'social', 'family'],
            'comfort': ['comfort', 'quiet', 'pleasant'],
            'safety': ['safe', 'security']
        }
        
        benefit_count = 0
        for category, keywords in benefit_keywords.items():
            if any(keyword in co_benefits_lower for keyword in keywords):
                benefit_count += 1
        
        # Normalize: 0 benefits = 0.2, 3+ benefits = 1.0
        score = 0.2 + (min(benefit_count, 5) / 5) * 0.8
        
        return score
    
    def _get_score_breakdown(self, rec: Dict[str, Any]) -> Dict[str, float]:
        """
        Get detailed breakdown of score components.
        
        Args:
            recommendation: Single recommendation
            
        Returns:
            Dictionary with component scores
        """
        return {
            'emission_reduction': self._calculate_emission_score(rec),
            'cost_effectiveness': self._calculate_cost_effectiveness_score(rec),
            'ease_of_implementation': self._calculate_ease_score(rec),
            'time_to_impact': self._calculate_time_score(rec),
            'co_benefits': self._calculate_cobenefit_score(rec)
        }
    
    def get_top_n(
        self,
        recommendations: List[Dict[str, Any]],
        n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get top N recommendations by score.
        
        Args:
            recommendations: List of recommendations
            n: Number of top recommendations to return
            
        Returns:
            Top N recommendations
        """
        ranked = self.rank_recommendations(recommendations)
        return ranked[:n]
    
    def filter_by_minimum_score(
        self,
        recommendations: List[Dict[str, Any]],
        min_score: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Filter recommendations by minimum score threshold.
        
        Args:
            recommendations: List of recommendations
            min_score: Minimum score threshold (0-1)
            
        Returns:
            Filtered recommendations above threshold
        """
        ranked = self.rank_recommendations(recommendations)
        return [rec for rec in ranked if rec.get('ranking_score', 0) >= min_score]
