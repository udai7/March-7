"""
Financial Impact Calculator for Environmental Sustainability.

This module calculates cost savings, ROI for green investments,
utility cost comparisons, and carbon tax/credit estimations.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import math


class InvestmentType(str, Enum):
    """Types of green investments."""
    SOLAR_PANELS = "solar_panels"
    ELECTRIC_VEHICLE = "electric_vehicle"
    HEAT_PUMP = "heat_pump"
    LED_LIGHTING = "led_lighting"
    INSULATION = "insulation"
    SMART_THERMOSTAT = "smart_thermostat"
    ELECTRIC_BIKE = "electric_bike"
    RAINWATER_HARVESTING = "rainwater_harvesting"
    SOLAR_WATER_HEATER = "solar_water_heater"
    ENERGY_EFFICIENT_APPLIANCES = "energy_efficient_appliances"


@dataclass
class CostSavings:
    """Represents cost savings from environmental actions."""
    daily_savings: float
    monthly_savings: float
    annual_savings: float
    lifetime_savings: float  # Over typical product lifetime
    currency: str = "USD"
    description: str = ""


@dataclass
class ROIResult:
    """Represents ROI calculation results."""
    initial_cost: float
    annual_savings: float
    payback_years: float
    total_roi_percent: float
    net_present_value: float
    internal_rate_of_return: float
    lifetime_years: int
    total_lifetime_savings: float
    environmental_savings: Dict[str, float]  # CO2, water, energy saved


@dataclass
class CarbonCredit:
    """Represents carbon credit/tax estimation."""
    annual_co2_kg: float
    carbon_price_per_ton: float
    annual_credit_value: float
    annual_tax_liability: float
    net_position: float  # Positive = earning credits, Negative = paying tax


class FinancialCalculator:
    """Calculate financial impacts of environmental choices."""
    
    # Average utility rates (can be customized by region)
    DEFAULT_RATES = {
        "electricity_kwh": 0.15,  # USD per kWh
        "natural_gas_therm": 1.20,  # USD per therm
        "water_gallon": 0.005,  # USD per gallon
        "water_liter": 0.0013,  # USD per liter
        "gasoline_gallon": 3.50,  # USD per gallon
        "gasoline_liter": 0.92,  # USD per liter
        "diesel_gallon": 4.00,  # USD per gallon
        "carbon_price_per_ton": 50.0,  # USD per metric ton CO2
    }
    
    # Green investment data
    INVESTMENT_DATA = {
        InvestmentType.SOLAR_PANELS: {
            "name": "Residential Solar Panel System (6kW)",
            "cost_range": (12000, 20000),
            "avg_cost": 16000,
            "annual_savings_range": (1200, 2400),
            "avg_annual_savings": 1800,
            "lifetime_years": 25,
            "co2_savings_kg_year": 4000,
            "maintenance_annual": 150,
            "incentives_percent": 30,  # Federal tax credit
        },
        InvestmentType.ELECTRIC_VEHICLE: {
            "name": "Electric Vehicle (vs Gas Car)",
            "cost_range": (35000, 55000),
            "avg_cost": 45000,
            "gas_car_equivalent": 32000,
            "annual_fuel_savings": 1500,
            "annual_maintenance_savings": 500,
            "avg_annual_savings": 2000,
            "lifetime_years": 12,
            "co2_savings_kg_year": 3500,
            "incentives_max": 7500,
        },
        InvestmentType.HEAT_PUMP: {
            "name": "Heat Pump HVAC System",
            "cost_range": (4000, 12000),
            "avg_cost": 8000,
            "annual_savings_range": (500, 1500),
            "avg_annual_savings": 1000,
            "lifetime_years": 15,
            "co2_savings_kg_year": 2000,
            "maintenance_annual": 100,
            "incentives_percent": 10,
        },
        InvestmentType.LED_LIGHTING: {
            "name": "LED Lighting Upgrade (Whole Home)",
            "cost_range": (200, 500),
            "avg_cost": 350,
            "annual_savings_range": (100, 200),
            "avg_annual_savings": 150,
            "lifetime_years": 15,
            "co2_savings_kg_year": 200,
            "maintenance_annual": 0,
        },
        InvestmentType.INSULATION: {
            "name": "Home Insulation Upgrade",
            "cost_range": (2000, 8000),
            "avg_cost": 5000,
            "annual_savings_range": (300, 800),
            "avg_annual_savings": 550,
            "lifetime_years": 40,
            "co2_savings_kg_year": 800,
            "maintenance_annual": 0,
        },
        InvestmentType.SMART_THERMOSTAT: {
            "name": "Smart Thermostat",
            "cost_range": (150, 300),
            "avg_cost": 225,
            "annual_savings_range": (100, 200),
            "avg_annual_savings": 150,
            "lifetime_years": 10,
            "co2_savings_kg_year": 300,
            "maintenance_annual": 0,
        },
        InvestmentType.ELECTRIC_BIKE: {
            "name": "Electric Bike (vs Car Commute)",
            "cost_range": (1000, 3000),
            "avg_cost": 2000,
            "annual_savings_range": (500, 2000),
            "avg_annual_savings": 1200,
            "lifetime_years": 8,
            "co2_savings_kg_year": 1500,
            "maintenance_annual": 100,
        },
        InvestmentType.RAINWATER_HARVESTING: {
            "name": "Rainwater Harvesting System",
            "cost_range": (1500, 5000),
            "avg_cost": 3000,
            "annual_savings_range": (100, 400),
            "avg_annual_savings": 250,
            "lifetime_years": 20,
            "water_savings_liters_year": 50000,
            "maintenance_annual": 50,
        },
        InvestmentType.SOLAR_WATER_HEATER: {
            "name": "Solar Water Heater",
            "cost_range": (3000, 6000),
            "avg_cost": 4500,
            "annual_savings_range": (200, 500),
            "avg_annual_savings": 350,
            "lifetime_years": 20,
            "co2_savings_kg_year": 1000,
            "maintenance_annual": 50,
        },
        InvestmentType.ENERGY_EFFICIENT_APPLIANCES: {
            "name": "Energy Star Appliance Upgrade",
            "cost_range": (500, 2000),
            "avg_cost": 1200,
            "annual_savings_range": (100, 300),
            "avg_annual_savings": 200,
            "lifetime_years": 12,
            "co2_savings_kg_year": 400,
            "maintenance_annual": 0,
        },
    }
    
    def __init__(self, rates: Optional[Dict[str, float]] = None):
        """
        Initialize the financial calculator.
        
        Args:
            rates: Custom utility rates (optional)
        """
        self.rates = {**self.DEFAULT_RATES, **(rates or {})}
    
    def calculate_activity_cost_savings(
        self,
        current_activity: str,
        alternative_activity: str,
        daily_usage: float,
        usage_unit: str = "kwh"
    ) -> CostSavings:
        """
        Calculate cost savings from switching activities.
        
        Args:
            current_activity: Current activity name
            alternative_activity: Alternative activity name
            daily_usage: Daily usage amount
            usage_unit: Unit of measurement (kwh, liters, gallons, km, miles)
            
        Returns:
            CostSavings object with financial impact
        """
        # Activity cost mappings (per unit)
        activity_costs = {
            # Transport (per km)
            "driving_petrol": 0.12,
            "driving_diesel": 0.10,
            "driving_electric": 0.04,
            "public_transit": 0.08,
            "cycling": 0.01,
            "walking": 0.0,
            "ebike": 0.02,
            
            # Energy (per kWh equivalent)
            "electricity_grid": self.rates["electricity_kwh"],
            "electricity_solar": 0.02,
            "natural_gas": 0.08,
            "heat_pump": 0.05,
            
            # Water (per liter)
            "tap_water": self.rates["water_liter"],
            "bottled_water": 0.50,
            "rainwater": 0.001,
        }
        
        current_cost = activity_costs.get(current_activity.lower().replace(" ", "_"), 0.10)
        alt_cost = activity_costs.get(alternative_activity.lower().replace(" ", "_"), 0.05)
        
        daily_savings = (current_cost - alt_cost) * daily_usage
        
        return CostSavings(
            daily_savings=round(daily_savings, 2),
            monthly_savings=round(daily_savings * 30, 2),
            annual_savings=round(daily_savings * 365, 2),
            lifetime_savings=round(daily_savings * 365 * 10, 2),
            description=f"Switching from {current_activity} to {alternative_activity}"
        )
    
    def calculate_energy_cost_savings(
        self,
        current_kwh_daily: float,
        reduced_kwh_daily: float,
        electricity_rate: Optional[float] = None
    ) -> CostSavings:
        """
        Calculate cost savings from energy reduction.
        
        Args:
            current_kwh_daily: Current daily kWh usage
            reduced_kwh_daily: Reduced daily kWh usage
            electricity_rate: Custom electricity rate (optional)
            
        Returns:
            CostSavings object
        """
        rate = electricity_rate or self.rates["electricity_kwh"]
        savings_kwh = current_kwh_daily - reduced_kwh_daily
        daily_savings = savings_kwh * rate
        
        return CostSavings(
            daily_savings=round(daily_savings, 2),
            monthly_savings=round(daily_savings * 30, 2),
            annual_savings=round(daily_savings * 365, 2),
            lifetime_savings=round(daily_savings * 365 * 20, 2),
            description=f"Reducing energy use by {savings_kwh:.1f} kWh/day"
        )
    
    def calculate_water_cost_savings(
        self,
        current_liters_daily: float,
        reduced_liters_daily: float,
        water_rate: Optional[float] = None
    ) -> CostSavings:
        """
        Calculate cost savings from water reduction.
        
        Args:
            current_liters_daily: Current daily water usage in liters
            reduced_liters_daily: Reduced daily water usage in liters
            water_rate: Custom water rate per liter (optional)
            
        Returns:
            CostSavings object
        """
        rate = water_rate or self.rates["water_liter"]
        savings_liters = current_liters_daily - reduced_liters_daily
        daily_savings = savings_liters * rate
        
        return CostSavings(
            daily_savings=round(daily_savings, 3),
            monthly_savings=round(daily_savings * 30, 2),
            annual_savings=round(daily_savings * 365, 2),
            lifetime_savings=round(daily_savings * 365 * 20, 2),
            description=f"Reducing water use by {savings_liters:.0f} liters/day"
        )
    
    def calculate_transport_cost_savings(
        self,
        daily_km: float,
        current_mode: str = "petrol_car",
        alternative_mode: str = "electric_car"
    ) -> CostSavings:
        """
        Calculate cost savings from transport mode change.
        
        Args:
            daily_km: Daily kilometers traveled
            current_mode: Current transport mode
            alternative_mode: Alternative transport mode
            
        Returns:
            CostSavings object
        """
        # Cost per km for different modes
        mode_costs = {
            "petrol_car": 0.15,
            "diesel_car": 0.12,
            "electric_car": 0.05,
            "hybrid_car": 0.08,
            "motorcycle": 0.08,
            "public_transit": 0.10,
            "cycling": 0.01,
            "ebike": 0.02,
            "walking": 0.0,
            "carpool": 0.06,
        }
        
        current_cost = mode_costs.get(current_mode, 0.15) * daily_km
        alt_cost = mode_costs.get(alternative_mode, 0.05) * daily_km
        daily_savings = current_cost - alt_cost
        
        return CostSavings(
            daily_savings=round(daily_savings, 2),
            monthly_savings=round(daily_savings * 30, 2),
            annual_savings=round(daily_savings * 365, 2),
            lifetime_savings=round(daily_savings * 365 * 10, 2),
            description=f"Switching from {current_mode} to {alternative_mode} for {daily_km}km/day"
        )
    
    def calculate_investment_roi(
        self,
        investment_type: InvestmentType,
        custom_cost: Optional[float] = None,
        custom_annual_savings: Optional[float] = None,
        discount_rate: float = 0.05
    ) -> ROIResult:
        """
        Calculate ROI for green investments.
        
        Args:
            investment_type: Type of green investment
            custom_cost: Custom initial cost (optional)
            custom_annual_savings: Custom annual savings (optional)
            discount_rate: Discount rate for NPV calculation
            
        Returns:
            ROIResult object with comprehensive ROI analysis
        """
        data = self.INVESTMENT_DATA[investment_type]
        
        initial_cost = custom_cost or data["avg_cost"]
        annual_savings = custom_annual_savings or data["avg_annual_savings"]
        lifetime = data["lifetime_years"]
        maintenance = data.get("maintenance_annual", 0)
        
        # Apply incentives if available
        incentive_percent = data.get("incentives_percent", 0)
        incentive_max = data.get("incentives_max", 0)
        
        if incentive_percent > 0:
            incentive = initial_cost * (incentive_percent / 100)
            initial_cost -= incentive
        elif incentive_max > 0:
            initial_cost -= incentive_max
        
        # Net annual savings (after maintenance)
        net_annual_savings = annual_savings - maintenance
        
        # Payback period
        payback_years = initial_cost / net_annual_savings if net_annual_savings > 0 else float('inf')
        
        # Total lifetime savings
        total_lifetime_savings = (net_annual_savings * lifetime) - initial_cost
        
        # Total ROI
        total_roi = (total_lifetime_savings / initial_cost) * 100 if initial_cost > 0 else 0
        
        # Net Present Value (NPV)
        npv = -initial_cost
        for year in range(1, lifetime + 1):
            npv += net_annual_savings / ((1 + discount_rate) ** year)
        
        # Internal Rate of Return (IRR) - simplified approximation
        irr = self._calculate_irr(initial_cost, net_annual_savings, lifetime)
        
        # Environmental savings
        env_savings = {
            "co2_kg_lifetime": data.get("co2_savings_kg_year", 0) * lifetime,
            "co2_kg_annual": data.get("co2_savings_kg_year", 0),
            "water_liters_lifetime": data.get("water_savings_liters_year", 0) * lifetime,
        }
        
        return ROIResult(
            initial_cost=round(initial_cost, 2),
            annual_savings=round(net_annual_savings, 2),
            payback_years=round(payback_years, 1),
            total_roi_percent=round(total_roi, 1),
            net_present_value=round(npv, 2),
            internal_rate_of_return=round(irr * 100, 1),
            lifetime_years=lifetime,
            total_lifetime_savings=round(total_lifetime_savings, 2),
            environmental_savings=env_savings
        )
    
    def _calculate_irr(
        self,
        initial_cost: float,
        annual_cash_flow: float,
        years: int,
        precision: float = 0.0001
    ) -> float:
        """Calculate Internal Rate of Return using Newton's method."""
        if annual_cash_flow <= 0:
            return 0.0
        
        # Initial guess based on simple payback
        rate = annual_cash_flow / initial_cost
        
        for _ in range(100):  # Max iterations
            npv = -initial_cost
            npv_derivative = 0
            
            for year in range(1, years + 1):
                discount = (1 + rate) ** year
                npv += annual_cash_flow / discount
                npv_derivative -= year * annual_cash_flow / ((1 + rate) ** (year + 1))
            
            if abs(npv_derivative) < 1e-10:
                break
                
            new_rate = rate - npv / npv_derivative
            
            if abs(new_rate - rate) < precision:
                return max(0, new_rate)
            
            rate = new_rate
        
        return max(0, rate)
    
    def calculate_carbon_credit(
        self,
        annual_co2_reduction_kg: float,
        annual_co2_emissions_kg: float,
        carbon_price: Optional[float] = None
    ) -> CarbonCredit:
        """
        Calculate carbon credit/tax position.
        
        Args:
            annual_co2_reduction_kg: Annual CO2 reduction in kg
            annual_co2_emissions_kg: Annual CO2 emissions in kg
            carbon_price: Price per metric ton of CO2 (optional)
            
        Returns:
            CarbonCredit object with financial position
        """
        price_per_ton = carbon_price or self.rates["carbon_price_per_ton"]
        
        # Convert to metric tons
        reduction_tons = annual_co2_reduction_kg / 1000
        emissions_tons = annual_co2_emissions_kg / 1000
        
        credit_value = reduction_tons * price_per_ton
        tax_liability = emissions_tons * price_per_ton
        net_position = credit_value - tax_liability
        
        return CarbonCredit(
            annual_co2_kg=annual_co2_emissions_kg - annual_co2_reduction_kg,
            carbon_price_per_ton=price_per_ton,
            annual_credit_value=round(credit_value, 2),
            annual_tax_liability=round(tax_liability, 2),
            net_position=round(net_position, 2)
        )
    
    def compare_utility_costs(
        self,
        current_usage: Dict[str, float],
        optimized_usage: Dict[str, float]
    ) -> Dict[str, any]:
        """
        Compare utility costs between current and optimized usage.
        
        Args:
            current_usage: Dict with electricity_kwh, gas_therms, water_liters
            optimized_usage: Dict with optimized values
            
        Returns:
            Comparison results with savings breakdown
        """
        def calc_monthly_cost(usage: Dict[str, float]) -> float:
            electricity = usage.get("electricity_kwh", 0) * 30 * self.rates["electricity_kwh"]
            gas = usage.get("gas_therms", 0) * 30 * self.rates["natural_gas_therm"]
            water = usage.get("water_liters", 0) * 30 * self.rates["water_liter"]
            return electricity + gas + water
        
        current_monthly = calc_monthly_cost(current_usage)
        optimized_monthly = calc_monthly_cost(optimized_usage)
        
        return {
            "current_monthly_cost": round(current_monthly, 2),
            "optimized_monthly_cost": round(optimized_monthly, 2),
            "monthly_savings": round(current_monthly - optimized_monthly, 2),
            "annual_savings": round((current_monthly - optimized_monthly) * 12, 2),
            "savings_percentage": round(((current_monthly - optimized_monthly) / current_monthly) * 100, 1) if current_monthly > 0 else 0,
            "breakdown": {
                "electricity_savings": round(
                    (current_usage.get("electricity_kwh", 0) - optimized_usage.get("electricity_kwh", 0)) * 30 * self.rates["electricity_kwh"], 2
                ),
                "gas_savings": round(
                    (current_usage.get("gas_therms", 0) - optimized_usage.get("gas_therms", 0)) * 30 * self.rates["natural_gas_therm"], 2
                ),
                "water_savings": round(
                    (current_usage.get("water_liters", 0) - optimized_usage.get("water_liters", 0)) * 30 * self.rates["water_liter"], 2
                ),
            }
        }
    
    def get_all_investment_options(self) -> List[Dict]:
        """Get all available green investment options with summary data."""
        options = []
        for inv_type, data in self.INVESTMENT_DATA.items():
            roi = self.calculate_investment_roi(inv_type)
            options.append({
                "type": inv_type.value,
                "name": data["name"],
                "cost_range": data["cost_range"],
                "avg_cost": data["avg_cost"],
                "payback_years": roi.payback_years,
                "annual_savings": roi.annual_savings,
                "lifetime_savings": roi.total_lifetime_savings,
                "roi_percent": roi.total_roi_percent,
                "co2_savings_annual": data.get("co2_savings_kg_year", 0),
            })
        
        # Sort by payback period
        options.sort(key=lambda x: x["payback_years"])
        return options
