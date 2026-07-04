"""
Context-aware recommendation filtering based on user circumstances.

This module provides context management for personalized CO2 reduction
recommendations based on user location, budget, lifestyle, and timeframe.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from enum import Enum


class Budget(str, Enum):
    """Budget categories for filtering recommendations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ANY = "any"


class Lifestyle(str, Enum):
    """Lifestyle/location categories."""
    URBAN = "urban"
    SUBURBAN = "suburban"
    RURAL = "rural"
    ANY = "any"


class Timeframe(str, Enum):
    """Implementation timeframe categories."""
    IMMEDIATE = "immediate"
    SHORT_TERM = "short-term"
    LONG_TERM = "long-term"
    ANY = "any"


@dataclass
class UserContext:
    """
    User context information for filtering recommendations.
    
    Attributes:
        location: User's country/region for region-specific tips
        budget: Budget category (low/medium/high)
        timeframe: Desired implementation timeframe
        lifestyle: Urban/suburban/rural lifestyle
        household_size: Number of people in household
        has_car: Whether user owns a car
        has_garden: Whether user has garden/yard space
        dietary_preference: Vegetarian, vegan, omnivore, etc.
    """
    location: Optional[str] = None
    budget: Budget = Budget.ANY
    timeframe: Timeframe = Timeframe.ANY
    lifestyle: Lifestyle = Lifestyle.ANY
    household_size: Optional[int] = None
    has_car: Optional[bool] = None
    has_garden: Optional[bool] = None
    dietary_preference: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "location": self.location,
            "budget": self.budget.value if self.budget else None,
            "timeframe": self.timeframe.value if self.timeframe else None,
            "lifestyle": self.lifestyle.value if self.lifestyle else None,
            "household_size": self.household_size,
            "has_car": self.has_car,
            "has_garden": self.has_garden,
            "dietary_preference": self.dietary_preference
        }


class ContextAwareRecommender:
    """
    Filters recommendations based on user context.
    
    Applies context-based filtering to ensure recommendations are
    practical and relevant for the user's specific situation.
    """
    
    def __init__(self):
        """Initialize the context-aware recommender."""
        pass
    
    def filter_recommendations(
        self,
        recommendations: List[Dict[str, Any]],
        context: UserContext
    ) -> List[Dict[str, Any]]:
        """
        Filter recommendations based on user context.
        
        Args:
            recommendations: List of recommendation dictionaries
            context: User context information
            
        Returns:
            Filtered list of recommendations that match user context
        """
        if not recommendations:
            return []
        
        filtered = []
        
        for rec in recommendations:
            # Check if recommendation passes all context filters
            if self._passes_context_filters(rec, context):
                # Add context relevance score
                rec['context_score'] = self._calculate_context_score(rec, context)
                filtered.append(rec)
        
        # Sort by context score (higher is better)
        filtered.sort(key=lambda x: x.get('context_score', 0), reverse=True)
        
        return filtered
    
    def _passes_context_filters(
        self,
        recommendation: Dict[str, Any],
        context: UserContext
    ) -> bool:
        """
        Check if a recommendation passes context filters.
        
        Args:
            recommendation: Single recommendation dictionary
            context: User context
            
        Returns:
            True if recommendation is appropriate for context
        """
        # Budget filter
        if context.budget != Budget.ANY:
            rec_cost = recommendation.get('cost_category', 'medium').lower()
            
            if context.budget == Budget.LOW and rec_cost == 'high':
                return False
            
            if context.budget == Budget.MEDIUM and rec_cost == 'high':
                # Allow medium budget users to see some high-cost items
                # if they have very high emission reduction
                if recommendation.get('reduction_percentage', 0) < 60:
                    return False
        
        # Lifestyle filter
        if context.lifestyle != Lifestyle.ANY:
            prerequisites = recommendation.get('prerequisites', '').lower()
            
            # Rural users - filter out public transport
            if context.lifestyle == Lifestyle.RURAL:
                if 'public transport' in prerequisites or 'bus' in prerequisites:
                    return False
                if 'metro' in prerequisites or 'subway' in prerequisites:
                    return False
            
            # Urban users - prioritize public transport and walkability
            if context.lifestyle == Lifestyle.URBAN:
                if 'large property' in prerequisites or 'backyard' in prerequisites:
                    return False
        
        # Timeframe filter
        if context.timeframe != Timeframe.ANY:
            rec_timeframe = recommendation.get('timeframe', 'medium').lower()
            
            if context.timeframe == Timeframe.IMMEDIATE:
                if 'long-term' in rec_timeframe:
                    return False
            
            if context.timeframe == Timeframe.SHORT_TERM:
                if 'long-term' in rec_timeframe:
                    # Allow some long-term recommendations if high impact
                    if recommendation.get('reduction_percentage', 0) < 70:
                        return False
        
        # Car ownership filter
        if context.has_car is False:
            category = recommendation.get('category', '').lower()
            if category == 'transport':
                # Filter out car-specific recommendations
                if any(keyword in recommendation.get('name', '').lower() 
                       for keyword in ['car', 'vehicle', 'driving', 'tire', 'fuel']):
                    return False
        
        # Garden filter
        if context.has_garden is False:
            if any(keyword in recommendation.get('name', '').lower()
                   for keyword in ['garden', 'compost', 'grow', 'outdoor']):
                return False
        
        return True
    
    def _calculate_context_score(
        self,
        recommendation: Dict[str, Any],
        context: UserContext
    ) -> float:
        """
        Calculate a context relevance score (0-1).
        
        Args:
            recommendation: Single recommendation
            context: User context
            
        Returns:
            Score from 0 to 1 indicating context relevance
        """
        score = 0.5  # Base score
        
        # Boost for matching budget
        rec_cost = recommendation.get('cost_category', 'medium').lower()
        if context.budget != Budget.ANY:
            if context.budget.value == rec_cost:
                score += 0.2
            elif context.budget == Budget.LOW and rec_cost == 'low':
                score += 0.3
        
        # Boost for matching timeframe
        rec_timeframe = recommendation.get('timeframe', '').lower()
        if context.timeframe != Timeframe.ANY:
            if context.timeframe.value in rec_timeframe:
                score += 0.15
        
        # Boost for lifestyle match
        if context.lifestyle == Lifestyle.URBAN:
            if any(keyword in recommendation.get('name', '').lower()
                   for keyword in ['public', 'bus', 'train', 'walk', 'bike']):
                score += 0.15
        
        # Boost for household size relevance
        if context.household_size and context.household_size > 1:
            if any(keyword in recommendation.get('co_benefits', '').lower()
                   for keyword in ['family', 'household', 'sharing']):
                score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def extract_context_from_query(self, query: str) -> UserContext:
        """
        Extract user context hints from query text.
        
        Args:
            query: User's query string
            
        Returns:
            UserContext object with extracted information
        """
        query_lower = query.lower()
        context = UserContext()
        
        # Extract budget hints
        if any(word in query_lower for word in ['cheap', 'free', 'low cost', 'affordable']):
            context.budget = Budget.LOW
        elif any(word in query_lower for word in ['expensive', 'invest', 'long term']):
            context.budget = Budget.HIGH
        
        # Extract timeframe hints
        if any(word in query_lower for word in ['now', 'immediate', 'today', 'quick']):
            context.timeframe = Timeframe.IMMEDIATE
        elif any(word in query_lower for word in ['eventually', 'future', 'planning']):
            context.timeframe = Timeframe.LONG_TERM
        
        # Extract lifestyle hints
        if any(word in query_lower for word in ['city', 'urban', 'apartment']):
            context.lifestyle = Lifestyle.URBAN
        elif any(word in query_lower for word in ['rural', 'countryside', 'farm']):
            context.lifestyle = Lifestyle.RURAL
        elif any(word in query_lower for word in ['suburban', 'suburb']):
            context.lifestyle = Lifestyle.SUBURBAN
        
        # Extract car ownership
        if any(word in query_lower for word in ['my car', 'i drive', 'my vehicle']):
            context.has_car = True
        elif any(word in query_lower for word in ["don't have a car", 'no car', 'without car']):
            context.has_car = False
        
        return context
