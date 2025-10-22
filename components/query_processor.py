"""
Query processor for extracting activities, intent, and parameters from user queries.

This module analyzes user queries to identify activities, determine query intent,
and extract relevant parameters for CO2 emission calculations.
"""

import re
from typing import List, Dict, Any, Optional
from enum import Enum


class QueryIntent(str, Enum):
    """Types of query intents."""
    SINGLE_ACTIVITY = "single_activity"  # Query about one specific activity
    COMPARISON = "comparison"  # Comparing multiple activities
    GENERAL_ADVICE = "general_advice"  # General sustainability advice
    DATASET_ANALYSIS = "dataset_analysis"  # Analyzing uploaded data
    UNKNOWN = "unknown"  # Cannot determine intent


class QueryProcessor:
    """Processes user queries to extract activities, intent, and parameters."""
    
    # Common activity keywords by category
    TRANSPORT_KEYWORDS = [
        "drive", "driving", "car", "vehicle", "petrol", "diesel", "electric car",
        "bus", "train", "metro", "subway", "bike", "bicycle", "cycling", "walk",
        "walking", "flight", "fly", "airplane", "scooter", "motorcycle", "motorbike"
    ]
    
    HOUSEHOLD_KEYWORDS = [
        "electricity", "power", "heating", "cooling", "air conditioning", "ac",
        "water", "shower", "bath", "laundry", "washing", "dishwasher", "lights",
        "lighting", "appliance", "refrigerator", "fridge", "oven", "stove"
    ]
    
    FOOD_KEYWORDS = [
        "eat", "eating", "food", "meal", "diet", "meat", "beef", "chicken",
        "pork", "fish", "vegetarian", "vegan", "dairy", "milk", "cheese",
        "local", "organic", "processed", "restaurant"
    ]
    
    LIFESTYLE_KEYWORDS = [
        "shopping", "clothes", "fashion", "online", "delivery", "package",
        "waste", "recycle", "recycling", "compost", "plastic", "paper"
    ]
    
    # Comparison keywords
    COMPARISON_KEYWORDS = [
        "vs", "versus", "compare", "comparison", "better", "worse", "alternative",
        "instead", "replace", "switch", "or", "between"
    ]
    
    # General advice keywords
    ADVICE_KEYWORDS = [
        "how to", "how can", "what can", "tips", "advice", "suggest", "recommend",
        "help", "reduce", "lower", "decrease", "improve", "ways to"
    ]
    
    def __init__(self):
        """Initialize the QueryProcessor."""
        pass
    
    def extract_activities(self, query: str) -> List[str]:
        """
        Extract activity mentions from user query.
        
        Args:
            query: User query text
            
        Returns:
            List of identified activity strings
        """
        query_lower = query.lower()
        activities = []
        
        # Check for transport activities
        for keyword in self.TRANSPORT_KEYWORDS:
            if keyword in query_lower:
                # Extract context around the keyword
                pattern = rf'\b[\w\s]*{re.escape(keyword)}[\w\s]*\b'
                matches = re.findall(pattern, query_lower)
                activities.extend([m.strip() for m in matches if m.strip()])
        
        # Check for household activities
        for keyword in self.HOUSEHOLD_KEYWORDS:
            if keyword in query_lower:
                pattern = rf'\b[\w\s]*{re.escape(keyword)}[\w\s]*\b'
                matches = re.findall(pattern, query_lower)
                activities.extend([m.strip() for m in matches if m.strip()])
        
        # Check for food activities
        for keyword in self.FOOD_KEYWORDS:
            if keyword in query_lower:
                pattern = rf'\b[\w\s]*{re.escape(keyword)}[\w\s]*\b'
                matches = re.findall(pattern, query_lower)
                activities.extend([m.strip() for m in matches if m.strip()])
        
        # Check for lifestyle activities
        for keyword in self.LIFESTYLE_KEYWORDS:
            if keyword in query_lower:
                pattern = rf'\b[\w\s]*{re.escape(keyword)}[\w\s]*\b'
                matches = re.findall(pattern, query_lower)
                activities.extend([m.strip() for m in matches if m.strip()])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_activities = []
        for activity in activities:
            if activity not in seen and len(activity) > 2:
                seen.add(activity)
                unique_activities.append(activity)
        
        return unique_activities[:5]  # Limit to top 5 activities
    
    def identify_intent(self, query: str) -> QueryIntent:
        """
        Determine the type of query (single activity, comparison, general advice).
        
        Args:
            query: User query text
            
        Returns:
            QueryIntent enum value
        """
        query_lower = query.lower()
        
        # Check for comparison intent
        comparison_count = sum(1 for keyword in self.COMPARISON_KEYWORDS if keyword in query_lower)
        if comparison_count > 0:
            # Check if multiple activities are mentioned
            activities = self.extract_activities(query)
            if len(activities) >= 2 or comparison_count >= 2:
                return QueryIntent.COMPARISON
        
        # Check for general advice intent
        advice_count = sum(1 for keyword in self.ADVICE_KEYWORDS if keyword in query_lower)
        if advice_count > 0:
            return QueryIntent.GENERAL_ADVICE
        
        # Check for single activity intent
        activities = self.extract_activities(query)
        if len(activities) >= 1:
            # Check if specific parameters are mentioned (distance, frequency, etc.)
            has_parameters = self._has_numeric_parameters(query)
            if has_parameters:
                return QueryIntent.SINGLE_ACTIVITY
            # Even without parameters, if only one activity is clearly mentioned
            if len(activities) == 1:
                return QueryIntent.SINGLE_ACTIVITY
        
        # If we found activities but no clear intent, default to single activity
        if activities:
            return QueryIntent.SINGLE_ACTIVITY
        
        # Default to general advice if no clear pattern
        if any(word in query_lower for word in ["reduce", "lower", "help", "co2", "carbon", "emission"]):
            return QueryIntent.GENERAL_ADVICE
        
        return QueryIntent.UNKNOWN
    
    def extract_parameters(self, query: str) -> Dict[str, Any]:
        """
        Extract numerical parameters like distances, frequencies, quantities.
        
        Args:
            query: User query text
            
        Returns:
            Dictionary of extracted parameters
        """
        parameters = {}
        query_lower = query.lower()
        
        # Extract distance (km, miles, meters)
        distance_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:km|kilometer|kilometres)',
            r'(\d+(?:\.\d+)?)\s*(?:mile|miles)',
            r'(\d+(?:\.\d+)?)\s*(?:meter|metres|m)\b'
        ]
        
        for pattern in distance_patterns:
            match = re.search(pattern, query_lower)
            if match:
                distance = float(match.group(1))
                if 'mile' in pattern:
                    distance = distance * 1.60934  # Convert miles to km
                elif 'meter' in pattern or r'\bm\b' in pattern:
                    distance = distance / 1000  # Convert meters to km
                parameters['distance_km'] = distance
                break
        
        # Extract frequency (daily, weekly, monthly, times per day/week)
        frequency_patterns = [
            (r'(\d+)\s*(?:time|times)\s*(?:per|a|each)\s*day', 'times_per_day'),
            (r'(\d+)\s*(?:time|times)\s*(?:per|a|each)\s*week', 'times_per_week'),
            (r'(\d+)\s*(?:time|times)\s*(?:per|a|each)\s*month', 'times_per_month'),
            (r'daily', 'frequency'),
            (r'weekly', 'frequency'),
            (r'monthly', 'frequency')
        ]
        
        for pattern, param_name in frequency_patterns:
            match = re.search(pattern, query_lower)
            if match:
                if match.groups():
                    parameters[param_name] = int(match.group(1))
                else:
                    parameters[param_name] = pattern  # Store the frequency word
                break
        
        # Extract duration (hours, minutes)
        duration_patterns = [
            (r'(\d+(?:\.\d+)?)\s*(?:hour|hours|hr|hrs)', 'hours'),
            (r'(\d+(?:\.\d+)?)\s*(?:minute|minutes|min|mins)', 'minutes')
        ]
        
        for pattern, param_name in duration_patterns:
            match = re.search(pattern, query_lower)
            if match:
                parameters[param_name] = float(match.group(1))
        
        # Extract quantity/amount
        quantity_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:kg|kilogram|kilograms)',
            r'(\d+(?:\.\d+)?)\s*(?:liter|liters|litre|litres|l)\b',
            r'(\d+(?:\.\d+)?)\s*(?:kwh|kilowatt)',
        ]
        
        for pattern in quantity_patterns:
            match = re.search(pattern, query_lower)
            if match:
                quantity = float(match.group(1))
                if 'kg' in pattern or 'kilogram' in pattern:
                    parameters['quantity_kg'] = quantity
                elif 'liter' in pattern or 'litre' in pattern:
                    parameters['quantity_liters'] = quantity
                elif 'kwh' in pattern or 'kilowatt' in pattern:
                    parameters['quantity_kwh'] = quantity
                break
        
        # Extract vehicle type
        vehicle_types = ['petrol', 'diesel', 'electric', 'hybrid', 'gas']
        for vehicle_type in vehicle_types:
            if vehicle_type in query_lower:
                parameters['vehicle_type'] = vehicle_type
                break
        
        # Extract diet type
        diet_types = ['vegan', 'vegetarian', 'meat', 'omnivore']
        for diet_type in diet_types:
            if diet_type in query_lower:
                parameters['diet_type'] = diet_type
                break
        
        return parameters
    
    def _has_numeric_parameters(self, query: str) -> bool:
        """
        Check if query contains numeric parameters.
        
        Args:
            query: User query text
            
        Returns:
            True if numeric parameters are found
        """
        # Look for numbers followed by units
        pattern = r'\d+(?:\.\d+)?'
        return bool(re.search(pattern, query))
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process query and extract all relevant information.
        
        Args:
            query: User query text
            
        Returns:
            Dictionary with activities, intent, and parameters
        """
        if not query or not query.strip():
            return {
                'activities': [],
                'intent': QueryIntent.UNKNOWN,
                'parameters': {},
                'original_query': query
            }
        
        activities = self.extract_activities(query)
        intent = self.identify_intent(query)
        parameters = self.extract_parameters(query)
        
        return {
            'activities': activities,
            'intent': intent,
            'parameters': parameters,
            'original_query': query.strip()
        }
    
    def get_query_summary(self, query: str) -> str:
        """
        Get a human-readable summary of the processed query.
        
        Args:
            query: User query text
            
        Returns:
            Summary string
        """
        result = self.process_query(query)
        
        summary_parts = []
        summary_parts.append(f"Intent: {result['intent'].value}")
        
        if result['activities']:
            summary_parts.append(f"Activities: {', '.join(result['activities'])}")
        
        if result['parameters']:
            param_strs = [f"{k}={v}" for k, v in result['parameters'].items()]
            summary_parts.append(f"Parameters: {', '.join(param_strs)}")
        
        return " | ".join(summary_parts)
