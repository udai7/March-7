"""
Response validation and fact-checking for CO2 recommendations.

Validates emission values, calculations, and recommendation consistency
to ensure accuracy of AI-generated responses.
"""

import re
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd


class ResponseValidator:
    """
    Validates AI-generated recommendations for accuracy.
    
    Checks emission values against reference data, validates calculations,
    and ensures response consistency.
    """
    
    def __init__(self, reference_data_path: Optional[str] = None):
        """
        Initialize the response validator.
        
        Args:
            reference_data_path: Path to reference activities CSV
        """
        self.reference_data = {}
        
        if reference_data_path:
            self.load_reference_data(reference_data_path)
    
    def load_reference_data(self, filepath: str) -> None:
        """
        Load reference emission data for validation.
        
        Args:
            filepath: Path to CSV file with reference activities
        """
        try:
            df = pd.read_csv(filepath)
            
            # Create lookup dictionary
            for _, row in df.iterrows():
                activity_key = row['Activity'].lower().strip()
                self.reference_data[activity_key] = {
                    'emission': float(row['Avg_CO2_Emission(kg/day)']),
                    'category': row['Category']
                }
        except Exception as e:
            print(f"Warning: Could not load reference data: {e}")
    
    def validate_emission_value(
        self,
        activity: str,
        claimed_emission: float,
        tolerance: float = 0.25
    ) -> Tuple[bool, str]:
        """
        Validate if emission value is reasonable.
        
        Args:
            activity: Activity name
            claimed_emission: Claimed emission in kg CO2/day
            tolerance: Allowed variance as fraction (default 25%)
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Basic sanity checks
        if claimed_emission < 0:
            return False, f"Negative emissions not valid: {claimed_emission}"
        
        if claimed_emission > 100:
            return False, f"Unusually high daily emission: {claimed_emission} kg/day"
        
        # Check against reference data if available
        activity_key = activity.lower().strip()
        
        # Try exact match first
        if activity_key in self.reference_data:
            ref_emission = self.reference_data[activity_key]['emission']
            variance = abs(claimed_emission - ref_emission) / max(ref_emission, 0.001)
            
            if variance > tolerance:
                return False, (
                    f"Emission value {claimed_emission:.2f} kg/day differs significantly "
                    f"from reference {ref_emission:.2f} kg/day (variance: {variance*100:.1f}%)"
                )
        
        # Try fuzzy matching for partial matches
        else:
            for ref_key, ref_data in self.reference_data.items():
                # Check if activity contains key words from reference
                if any(word in activity_key for word in ref_key.split() if len(word) > 3):
                    ref_emission = ref_data['emission']
                    variance = abs(claimed_emission - ref_emission) / max(ref_emission, 0.001)
                    
                    # More lenient for fuzzy matches
                    if variance > tolerance * 1.5:
                        return False, (
                            f"Emission value {claimed_emission:.2f} kg/day may be incorrect. "
                            f"Similar activity '{ref_key}' has {ref_emission:.2f} kg/day"
                        )
        
        return True, "Valid"
    
    def validate_reduction_calculation(
        self,
        current_emission: float,
        alternative_emission: float,
        claimed_reduction_pct: float,
        tolerance: float = 5.0
    ) -> Tuple[bool, str]:
        """
        Validate reduction percentage calculation.
        
        Args:
            current_emission: Current emission in kg/day
            alternative_emission: Alternative emission in kg/day
            claimed_reduction_pct: Claimed reduction percentage
            tolerance: Allowed error in percentage points
            
        Returns:
            Tuple of (is_valid, message)
        """
        if current_emission <= 0:
            return False, f"Invalid current emission: {current_emission}"
        
        if alternative_emission < 0:
            return False, f"Invalid alternative emission: {alternative_emission}"
        
        # Calculate expected reduction
        expected_reduction = ((current_emission - alternative_emission) / current_emission) * 100
        
        # Check if claimed reduction is within tolerance
        error = abs(expected_reduction - claimed_reduction_pct)
        
        if error > tolerance:
            return False, (
                f"Reduction percentage error: claimed {claimed_reduction_pct:.1f}%, "
                f"expected {expected_reduction:.1f}% (error: {error:.1f} percentage points)"
            )
        
        return True, "Valid"
    
    def validate_annual_savings(
        self,
        daily_reduction: float,
        claimed_annual: float,
        tolerance: float = 50.0
    ) -> Tuple[bool, str]:
        """
        Validate annual savings calculation.
        
        Args:
            daily_reduction: Daily CO2 reduction in kg
            claimed_annual: Claimed annual savings in kg
            tolerance: Allowed error in kg
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Calculate expected annual savings
        expected_annual = daily_reduction * 365
        
        error = abs(expected_annual - claimed_annual)
        
        if error > tolerance:
            return False, (
                f"Annual savings error: claimed {claimed_annual:.0f} kg/year, "
                f"expected {expected_annual:.0f} kg/year (error: {error:.0f} kg)"
            )
        
        return True, "Valid"
    
    def validate_recommendation(
        self,
        recommendation: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate a complete recommendation.
        
        Args:
            recommendation: Recommendation dictionary with all fields
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required fields
        required_fields = [
            'activity', 'current_emission', 'alternative_emission',
            'reduction_percentage', 'annual_savings_kg'
        ]
        
        for field in required_fields:
            if field not in recommendation or recommendation[field] is None:
                issues.append(f"Missing required field: {field}")
        
        # If missing required fields, return early
        if issues:
            return False, issues
        
        # Validate current emission
        current_emission = recommendation.get('current_emission', 0)
        activity = recommendation.get('activity', '')
        
        is_valid, msg = self.validate_emission_value(activity, current_emission)
        if not is_valid:
            issues.append(f"Current emission: {msg}")
        
        # Validate alternative emission
        alternative_emission = recommendation.get('alternative_emission', 0)
        alternative_name = recommendation.get('alternative_name', 'alternative')
        
        is_valid, msg = self.validate_emission_value(alternative_name, alternative_emission)
        if not is_valid:
            issues.append(f"Alternative emission: {msg}")
        
        # Validate reduction percentage
        reduction_pct = recommendation.get('reduction_percentage', 0)
        
        is_valid, msg = self.validate_reduction_calculation(
            current_emission,
            alternative_emission,
            reduction_pct
        )
        if not is_valid:
            issues.append(msg)
        
        # Validate annual savings
        daily_reduction = current_emission - alternative_emission
        annual_savings = recommendation.get('annual_savings_kg', 0)
        
        is_valid, msg = self.validate_annual_savings(daily_reduction, annual_savings)
        if not is_valid:
            issues.append(msg)
        
        # Check for logical consistency
        if alternative_emission > current_emission:
            issues.append(
                f"Alternative emission ({alternative_emission:.2f}) is higher than "
                f"current emission ({current_emission:.2f})"
            )
        
        if reduction_pct < 0 or reduction_pct > 100:
            issues.append(f"Reduction percentage out of range: {reduction_pct}%")
        
        return len(issues) == 0, issues
    
    def extract_numbers_from_text(self, text: str) -> Dict[str, Optional[float]]:
        """
        Extract emission numbers from LLM response text.
        
        Args:
            text: Response text from LLM
            
        Returns:
            Dictionary with extracted numbers
        """
        result = {
            'current_emission': None,
            'alternative_emission': None,
            'reduction_percentage': None,
            'annual_savings': None
        }
        
        # Pattern for current emission
        current_pattern = r'current.*?(\d+\.?\d*)\s*kg\s*co2?/day'
        match = re.search(current_pattern, text.lower())
        if match:
            result['current_emission'] = float(match.group(1))
        
        # Pattern for alternative/new emission
        alt_pattern = r'(?:alternative|new).*?(\d+\.?\d*)\s*kg\s*co2?/day'
        match = re.search(alt_pattern, text.lower())
        if match:
            result['alternative_emission'] = float(match.group(1))
        
        # Pattern for reduction percentage
        pct_pattern = r'(\d+\.?\d*)%\s*reduction'
        match = re.search(pct_pattern, text.lower())
        if match:
            result['reduction_percentage'] = float(match.group(1))
        
        # Pattern for annual savings
        annual_pattern = r'(\d+\.?\d*)\s*kg\s*co2?/year'
        match = re.search(annual_pattern, text.lower())
        if match:
            result['annual_savings'] = float(match.group(1))
        
        return result
    
    def auto_correct_calculations(
        self,
        recommendation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Auto-correct calculation errors in recommendations.
        
        Args:
            recommendation: Recommendation with potential errors
            
        Returns:
            Corrected recommendation
        """
        corrected = recommendation.copy()
        
        current = recommendation.get('current_emission', 0)
        alternative = recommendation.get('alternative_emission', 0)
        
        if current > 0 and alternative >= 0:
            # Recalculate reduction percentage
            correct_reduction = ((current - alternative) / current) * 100
            corrected['reduction_percentage'] = round(correct_reduction, 1)
            
            # Recalculate annual savings
            daily_reduction = current - alternative
            correct_annual = daily_reduction * 365
            corrected['annual_savings_kg'] = round(correct_annual, 0)
            
            # Add flag indicating correction was made
            if abs(correct_reduction - recommendation.get('reduction_percentage', 0)) > 1:
                corrected['_corrected'] = True
                corrected['_original_reduction'] = recommendation.get('reduction_percentage')
        
        return corrected
    
    def validate_batch(
        self,
        recommendations: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Validate a batch of recommendations.
        
        Args:
            recommendations: List of recommendations to validate
            
        Returns:
            Tuple of (valid_recommendations, invalid_recommendations_with_issues)
        """
        valid = []
        invalid = []
        
        for rec in recommendations:
            is_valid, issues = self.validate_recommendation(rec)
            
            if is_valid:
                valid.append(rec)
            else:
                rec['validation_issues'] = issues
                invalid.append(rec)
        
        return valid, invalid
