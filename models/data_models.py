"""
Data models for the CO2 Reduction AI Agent.

This module defines Pydantic models for activities, recommendations,
agent responses, and dataset analysis results.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from enum import Enum


class Category(str, Enum):
    """Activity categories for CO2 emissions."""
    TRANSPORT = "Transport"
    HOUSEHOLD = "Household"
    FOOD = "Food"
    LIFESTYLE = "Lifestyle"


class Activity(BaseModel):
    """Represents an activity with its CO2 emission data."""
    name: str = Field(..., description="Name of the activity")
    emission_kg_per_day: float = Field(..., ge=0, description="CO2 emission in kg per day")
    category: Category = Field(..., description="Category of the activity")
    description: Optional[str] = Field(None, description="Optional description of the activity")

    @field_validator('emission_kg_per_day')
    @classmethod
    def validate_emission(cls, v: float) -> float:
        """Validate that emission values are non-negative."""
        if v < 0:
            raise ValueError("Emission value must be greater than or equal to 0")
        return v


class Recommendation(BaseModel):
    """Represents a recommendation for reducing CO2 emissions."""
    action: str = Field(..., description="Recommended action to take")
    emission_reduction_kg: float = Field(..., ge=0, description="Emission reduction in kg per day")
    reduction_percentage: float = Field(..., ge=0, le=100, description="Reduction as percentage")
    implementation_difficulty: str = Field(..., description="Difficulty level: Easy, Medium, or Hard")
    timeframe: str = Field(..., description="Implementation timeframe: Immediate, Short-term, or Long-term")
    additional_benefits: List[str] = Field(default_factory=list, description="Additional benefits beyond CO2 reduction")

    @field_validator('emission_reduction_kg')
    @classmethod
    def validate_reduction(cls, v: float) -> float:
        """Validate that reduction values are non-negative."""
        if v < 0:
            raise ValueError("Emission reduction must be greater than or equal to 0")
        return v


class AgentResponse(BaseModel):
    """Represents the agent's response to a user query."""
    current_emission: float = Field(..., ge=0, description="Current CO2 emission in kg per day")
    recommendations: List[Recommendation] = Field(..., description="List of recommendations")
    total_potential_reduction: float = Field(..., ge=0, description="Total potential reduction in kg per day")
    annual_savings_kg: float = Field(..., ge=0, description="Annual CO2 savings in kg")
    summary: str = Field(..., description="Summary of the analysis and recommendations")

    @field_validator('current_emission', 'total_potential_reduction', 'annual_savings_kg')
    @classmethod
    def validate_emissions(cls, v: float) -> float:
        """Validate that emission values are non-negative."""
        if v < 0:
            raise ValueError("Emission values must be greater than or equal to 0")
        return v


class DatasetAnalysis(BaseModel):
    """Represents analysis results for an uploaded dataset."""
    total_daily_emission: float = Field(..., ge=0, description="Total daily CO2 emission in kg")
    total_annual_emission: float = Field(..., ge=0, description="Total annual CO2 emission in kg")
    top_emitters: List[Activity] = Field(..., description="Top emission activities")
    category_breakdown: Dict[str, float] = Field(..., description="Emissions by category")
    recommendations: List[Recommendation] = Field(..., description="Prioritized recommendations")

    @field_validator('total_daily_emission', 'total_annual_emission')
    @classmethod
    def validate_totals(cls, v: float) -> float:
        """Validate that total emission values are non-negative."""
        if v < 0:
            raise ValueError("Total emission values must be greater than or equal to 0")
        return v


class ValidationResult(BaseModel):
    """Represents the result of data validation."""
    is_valid: bool = Field(..., description="Whether the data is valid")
    errors: List[str] = Field(default_factory=list, description="List of validation errors")
    warnings: List[str] = Field(default_factory=list, description="List of validation warnings")


class ReductionMetrics(BaseModel):
    """Represents metrics for emission reduction calculations."""
    current_emission: float = Field(..., ge=0, description="Current emission in kg per day")
    alternative_emission: float = Field(..., ge=0, description="Alternative emission in kg per day")
    absolute_reduction: float = Field(..., description="Absolute reduction in kg per day")
    percentage_reduction: float = Field(..., description="Percentage reduction")
    annual_savings: float = Field(..., description="Annual savings in kg")

    @field_validator('current_emission', 'alternative_emission')
    @classmethod
    def validate_emissions(cls, v: float) -> float:
        """Validate that emission values are non-negative."""
        if v < 0:
            raise ValueError("Emission values must be greater than or equal to 0")
        return v
