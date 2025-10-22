"""
Response parser for extracting structured data from LLM outputs
"""
import re
from typing import List, Optional, Dict, Any
from models.data_models import Recommendation


class ResponseParser:
    """Parser for extracting structured information from LLM text responses"""
    
    @staticmethod
    def parse_llm_response(response_text: str) -> List[Recommendation]:
        """
        Parse LLM response text into structured Recommendation objects
        
        Args:
            response_text: Raw text response from LLM
            
        Returns:
            List of Recommendation objects
        """
        recommendations = []
        
        # Split response into recommendation blocks
        # Look for patterns like "RECOMMENDATION 1:", "RECOMMENDATION 2:", etc.
        recommendation_blocks = re.split(
            r'RECOMMENDATION\s+\d+:', 
            response_text, 
            flags=re.IGNORECASE
        )
        
        # Skip the first split (text before first recommendation)
        for block in recommendation_blocks[1:]:
            try:
                rec = ResponseParser._parse_recommendation_block(block)
                if rec:
                    recommendations.append(rec)
            except Exception:
                # Skip malformed recommendations
                continue
        
        # If no structured recommendations found, try alternative parsing
        if not recommendations:
            recommendations = ResponseParser._parse_unstructured_response(response_text)
        
        return recommendations
    
    @staticmethod
    def _parse_recommendation_block(block: str) -> Optional[Recommendation]:
        """
        Parse a single recommendation block into a Recommendation object
        
        Args:
            block: Text block containing one recommendation
            
        Returns:
            Recommendation object or None if parsing fails
        """
        # Extract fields using regex patterns
        action = ResponseParser._extract_field(block, r'Action:\s*(.+?)(?:\n|$)')
        emission_reduction = ResponseParser._extract_emission(
            block, r'Emission Reduction:\s*([0-9.]+)\s*kg'
        )
        reduction_percentage = ResponseParser._extract_percentage(
            block, r'Reduction Percentage:\s*([0-9.]+)\s*%'
        )
        difficulty = ResponseParser._extract_field(
            block, r'Difficulty:\s*(Easy|Medium|Hard)', 
            default="Medium"
        )
        timeframe = ResponseParser._extract_field(
            block, r'Timeframe:\s*(Immediate|Short-term|Long-term)',
            default="Short-term"
        )
        additional_benefits = ResponseParser._extract_benefits(block)
        
        # Validate required fields
        if not action or emission_reduction is None:
            return None
        
        # Calculate reduction percentage if not provided
        if reduction_percentage is None:
            reduction_percentage = 0.0
        
        try:
            return Recommendation(
                action=action,
                emission_reduction_kg=emission_reduction,
                reduction_percentage=reduction_percentage,
                implementation_difficulty=difficulty,
                timeframe=timeframe,
                additional_benefits=additional_benefits
            )
        except Exception:
            return None
    
    @staticmethod
    def _parse_unstructured_response(response_text: str) -> List[Recommendation]:
        """
        Attempt to parse recommendations from unstructured text
        
        Args:
            response_text: Raw text response
            
        Returns:
            List of Recommendation objects (may be empty)
        """
        recommendations = []
        
        # Look for numbered lists or bullet points
        lines = response_text.split('\n')
        current_rec = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line starts a new recommendation
            if re.match(r'^[\d\-\*•]\s*\.?\s*', line):
                # Save previous recommendation if exists
                if current_rec:
                    try:
                        rec = ResponseParser._create_fallback_recommendation(current_rec)
                        if rec:
                            recommendations.append(rec)
                    except Exception:
                        pass
                
                # Start new recommendation
                current_rec = re.sub(r'^[\d\-\*•]\s*\.?\s*', '', line)
            elif current_rec:
                # Continue building current recommendation
                current_rec += " " + line
        
        # Don't forget the last recommendation
        if current_rec:
            try:
                rec = ResponseParser._create_fallback_recommendation(current_rec)
                if rec:
                    recommendations.append(rec)
            except Exception:
                pass
        
        return recommendations
    
    @staticmethod
    def _create_fallback_recommendation(text: str) -> Optional[Recommendation]:
        """
        Create a recommendation with fallback values from unstructured text
        
        Args:
            text: Recommendation text
            
        Returns:
            Recommendation object or None
        """
        # Extract any emission values mentioned
        emission_match = re.search(r'([0-9.]+)\s*kg', text)
        emission_reduction = float(emission_match.group(1)) if emission_match else 1.0
        
        # Extract percentage if mentioned
        percentage_match = re.search(r'([0-9.]+)\s*%', text)
        reduction_percentage = float(percentage_match.group(1)) if percentage_match else 10.0
        
        return Recommendation(
            action=text[:200],  # Limit action text length
            emission_reduction_kg=emission_reduction,
            reduction_percentage=reduction_percentage,
            implementation_difficulty="Medium",
            timeframe="Short-term",
            additional_benefits=[]
        )
    
    @staticmethod
    def _extract_field(
        text: str, 
        pattern: str, 
        default: Optional[str] = None
    ) -> Optional[str]:
        """
        Extract a field value using regex pattern
        
        Args:
            text: Text to search
            pattern: Regex pattern
            default: Default value if not found
            
        Returns:
            Extracted value or default
        """
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
        return default
    
    @staticmethod
    def _extract_emission(text: str, pattern: str) -> Optional[float]:
        """
        Extract emission value as float
        
        Args:
            text: Text to search
            pattern: Regex pattern
            
        Returns:
            Emission value or None
        """
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None
    
    @staticmethod
    def _extract_percentage(text: str, pattern: str) -> Optional[float]:
        """
        Extract percentage value as float
        
        Args:
            text: Text to search
            pattern: Regex pattern
            
        Returns:
            Percentage value or None
        """
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1))
                # Ensure percentage is between 0 and 100
                return min(max(value, 0.0), 100.0)
            except ValueError:
                return None
        return None
    
    @staticmethod
    def _extract_benefits(text: str) -> List[str]:
        """
        Extract additional benefits from text
        
        Args:
            text: Text to search
            
        Returns:
            List of benefit strings
        """
        # Look for "Additional Benefits:" section
        match = re.search(
            r'Additional Benefits:\s*(.+?)(?:\n\n|\n[A-Z]|$)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        
        if match:
            benefits_text = match.group(1).strip()
            # Split by commas or newlines
            benefits = re.split(r'[,\n]', benefits_text)
            # Clean and filter
            benefits = [b.strip().strip('-•*') for b in benefits if b.strip()]
            return [b for b in benefits if len(b) > 2]
        
        return []
    
    @staticmethod
    def parse_emission_value(text: str) -> Optional[float]:
        """
        Extract a single emission value from text
        
        Args:
            text: Text containing emission value
            
        Returns:
            Emission value in kg or None
        """
        # Look for patterns like "X kg", "X.X kg/day", etc.
        patterns = [
            r'([0-9.]+)\s*kg\s*/?\s*day',
            r'([0-9.]+)\s*kg',
            r'([0-9.]+)\s*kilograms?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        return None
    
    @staticmethod
    def parse_activity_extraction(response_text: str) -> Dict[str, Any]:
        """
        Parse activity extraction response into structured data
        
        Args:
            response_text: LLM response from extraction prompt
            
        Returns:
            Dictionary with activity, quantity, category, details
        """
        result = {
            'activity': None,
            'quantity': None,
            'category': None,
            'details': None
        }
        
        # Extract each field
        result['activity'] = ResponseParser._extract_field(
            response_text, r'Activity:\s*(.+?)(?:\n|$)'
        )
        result['quantity'] = ResponseParser._extract_field(
            response_text, r'Quantity:\s*(.+?)(?:\n|$)'
        )
        result['category'] = ResponseParser._extract_field(
            response_text, r'Category:\s*(.+?)(?:\n|$)'
        )
        result['details'] = ResponseParser._extract_field(
            response_text, r'Details:\s*(.+?)(?:\n|$)'
        )
        
        return result


def parse_llm_response(response_text: str) -> List[Recommendation]:
    """
    Convenience function for parsing LLM responses
    
    Args:
        response_text: Raw text response from LLM
        
    Returns:
        List of Recommendation objects
    """
    parser = ResponseParser()
    return parser.parse_llm_response(response_text)
