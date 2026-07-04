"""
Data models for the Environmental Impact AI Agent.

This module defines Pydantic models for activities, recommendations,
agent responses, and dataset analysis results covering multiple environmental metrics.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from enum import Enum


class Category(str, Enum):
    """Activity categories for environmental impact."""
    TRANSPORT = "Transport"
    HOUSEHOLD = "Household"
    FOOD = "Food"
    LIFESTYLE = "Lifestyle"
    ENERGY = "Energy"
    WATER = "Water"
    WASTE = "Waste"


class EnvironmentalMetrics(BaseModel):
    """Comprehensive environmental impact metrics for an activity."""
    co2_kg_per_day: float = Field(0.0, ge=0, description="CO2 emission in kg per day")
    water_liters_per_day: float = Field(0.0, ge=0, description="Water consumption in liters per day")
    energy_kwh_per_day: float = Field(0.0, ge=0, description="Energy consumption in kWh per day")
    waste_kg_per_day: float = Field(0.0, ge=0, description="Waste generation in kg per day")
    land_use_m2: float = Field(0.0, ge=0, description="Land use in square meters")
    pollution_index: float = Field(0.0, ge=0, le=100, description="Air/water pollution index (0-100)")
    
    @field_validator('co2_kg_per_day', 'water_liters_per_day', 'energy_kwh_per_day', 'waste_kg_per_day', 'land_use_m2')
    @classmethod
    def validate_positive(cls, v: float) -> float:
        """Validate that metric values are non-negative."""
        if v < 0:
            raise ValueError("Environmental metric values must be greater than or equal to 0")
        return v


class Activity(BaseModel):
    """Represents an activity with its comprehensive environmental impact data."""
    name: str = Field(..., description="Name of the activity")
    emission_kg_per_day: float = Field(..., ge=0, description="CO2 emission in kg per day (legacy support)")
    category: Category = Field(..., description="Category of the activity")
    description: Optional[str] = Field(None, description="Optional description of the activity")
    environmental_metrics: Optional[EnvironmentalMetrics] = Field(None, description="Comprehensive environmental metrics")

    @field_validator('emission_kg_per_day')
    @classmethod
    def validate_emission(cls, v: float) -> float:
        """Validate that emission values are non-negative."""
        if v < 0:
            raise ValueError("Emission value must be greater than or equal to 0")
        return v


class Recommendation(BaseModel):
    """Represents a recommendation for reducing environmental impact."""
    action: str = Field(..., description="Recommended action to take")
    emission_reduction_kg: float = Field(..., ge=0, description="CO2 emission reduction in kg per day")
    reduction_percentage: float = Field(..., ge=0, le=100, description="Reduction as percentage")
    implementation_difficulty: str = Field(..., description="Difficulty level: Easy, Medium, or Hard")
    timeframe: str = Field(..., description="Implementation timeframe: Immediate, Short-term, or Long-term")
    additional_benefits: List[str] = Field(default_factory=list, description="Additional benefits beyond CO2 reduction")
    
    # Extended environmental impact reductions
    water_reduction_liters: float = Field(0.0, ge=0, description="Water savings in liters per day")
    energy_reduction_kwh: float = Field(0.0, ge=0, description="Energy savings in kWh per day")
    waste_reduction_kg: float = Field(0.0, ge=0, description="Waste reduction in kg per day")
    cost_savings_annual: float = Field(0.0, description="Estimated annual cost savings in USD")
    health_benefits: List[str] = Field(default_factory=list, description="Health-related benefits")

    @field_validator('emission_reduction_kg', 'water_reduction_liters', 'energy_reduction_kwh', 'waste_reduction_kg')
    @classmethod
    def validate_reduction(cls, v: float) -> float:
        """Validate that reduction values are non-negative."""
        if v < 0:
            raise ValueError("Reduction values must be greater than or equal to 0")
        return v


class AgentResponse(BaseModel):
    """Represents the agent's response to a user query with comprehensive environmental metrics."""
    current_emission: float = Field(..., ge=0, description="Current CO2 emission in kg per day")
    recommendations: List[Recommendation] = Field(..., description="List of recommendations")
    total_potential_reduction: float = Field(..., ge=0, description="Total potential CO2 reduction in kg per day")
    annual_savings_kg: float = Field(..., ge=0, description="Annual CO2 savings in kg")
    summary: str = Field(..., description="Summary of the analysis and recommendations")
    
    # Extended environmental metrics
    current_water_usage: float = Field(0.0, ge=0, description="Current water usage in liters per day")
    current_energy_usage: float = Field(0.0, ge=0, description="Current energy usage in kWh per day")
    current_waste_generation: float = Field(0.0, ge=0, description="Current waste generation in kg per day")
    total_water_savings: float = Field(0.0, ge=0, description="Total potential water savings in liters per day")
    total_energy_savings: float = Field(0.0, ge=0, description="Total potential energy savings in kWh per day")
    total_waste_reduction: float = Field(0.0, ge=0, description="Total potential waste reduction in kg per day")
    environmental_score: float = Field(0.0, ge=0, le=100, description="Overall environmental impact score (0-100)")

    @field_validator('current_emission', 'total_potential_reduction', 'annual_savings_kg')
    @classmethod
    def validate_emissions(cls, v: float) -> float:
        """Validate that emission values are non-negative."""
        if v < 0:
            raise ValueError("Emission values must be greater than or equal to 0")
        return v


class DatasetAnalysis(BaseModel):
    """Represents analysis results for an uploaded dataset with comprehensive environmental metrics."""
    total_daily_emission: float = Field(..., ge=0, description="Total daily CO2 emission in kg")
    total_annual_emission: float = Field(..., ge=0, description="Total annual CO2 emission in kg")
    top_emitters: List[Activity] = Field(..., description="Top emission activities")
    category_breakdown: Dict[str, float] = Field(..., description="Emissions by category")
    recommendations: List[Recommendation] = Field(..., description="Prioritized recommendations")
    
    # Extended environmental metrics
    total_daily_water: float = Field(0.0, ge=0, description="Total daily water usage in liters")
    total_daily_energy: float = Field(0.0, ge=0, description="Total daily energy usage in kWh")
    total_daily_waste: float = Field(0.0, ge=0, description="Total daily waste generation in kg")
    environmental_breakdown: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, 
        description="Multi-metric breakdown by category"
    )
    sustainability_grade: str = Field("N/A", description="Overall sustainability grade (A+ to F)")

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
