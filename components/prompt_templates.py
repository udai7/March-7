"""
Prompt templates for LLM interactions with enhanced accuracy constraints
for comprehensive environmental impact analysis (CO2, water, energy, waste).
"""
from typing import List, Dict, Any, Optional
from models.data_models import Activity


class PromptTemplates:
    """Collection of structured prompts for different LLM tasks with improved accuracy"""
    
    @staticmethod
    def enhanced_recommendation_prompt(
        activity: str,
        emission: float,
        category: str,
        alternatives: List[str],
        user_context: Optional[Dict[str, Any]] = None,
        water_usage: float = 0.0,
        energy_usage: float = 0.0,
        waste_generation: float = 0.0
    ) -> str:
        """
        Generate enhanced prompt with accuracy constraints and user context.
        
        Args:
            activity: Current activity name
            emission: Current CO2 emission in kg/day
            category: Activity category
            alternatives: List of alternative activities from knowledge base
            user_context: Optional user context (budget, location, etc.)
            water_usage: Water consumption in liters/day
            energy_usage: Energy consumption in kWh/day
            waste_generation: Waste generation in kg/day
            
        Returns:
            Formatted prompt string with strict accuracy requirements
        """
        alternatives_text = "\n".join([f"- {alt}" for alt in alternatives]) if alternatives else "No specific alternatives provided"
        
        context_str = ""
        if user_context:
            context_str = f"""
USER CONTEXT:
- Location: {user_context.get('location', 'Not specified')}
- Budget: {user_context.get('budget', 'Not specified')}
- Timeframe: {user_context.get('timeframe', 'Not specified')}
- Lifestyle: {user_context.get('lifestyle', 'Not specified')}
"""
        
        prompt = f"""You are an expert environmental sustainability advisor. Provide accurate, data-driven recommendations based ONLY on the verified information below. Consider ALL environmental impacts: CO₂ emissions, water usage, energy consumption, and waste generation.

{context_str}
CURRENT ACTIVITY:
Activity: {activity}
Category: {category}
Current CO₂ Emission: {emission} kg/day ({emission * 365:.1f} kg/year)
Water Usage: {water_usage} liters/day
Energy Consumption: {energy_usage} kWh/day
Waste Generation: {waste_generation} kg/day

VERIFIED SUSTAINABILITY ALTERNATIVES:
{alternatives_text}

CRITICAL INSTRUCTIONS - ACCURACY REQUIREMENTS:
1. Use EXACT emission values from the alternatives above
2. Double-check ALL calculations before responding:
   - Reduction (kg/day) = Current - Alternative
   - Reduction % = ((Current - Alternative) / Current) × 100
   - Annual Savings = Daily Reduction × 365
3. Provide 3-5 alternatives ranked by OVERALL environmental impact
4. Include specific numbers with units (kg CO₂/day, liters water, kWh, % reduction, kg/year)
5. Consider user context when applicable
6. Mark difficulty: Easy/Medium/Hard based on implementation complexity
7. Specify timeframe: Immediate (<1 week)/Short-term (1-3 months)/Long-term (>3 months)
8. Add realistic cost estimates or note "Minimal cost" / "Cost savings"
9. List 2-4 co-benefits (health, financial savings, time, comfort, water savings, etc.)
10. If data is insufficient, state "Limited data available" instead of guessing
11. NEVER make up emission values - use only the provided alternatives

RESPONSE FORMAT (strictly follow):

**Current Activity:** {activity}
**Current Environmental Impact:**
- CO₂: {emission} kg/day ({emission * 365:.0f} kg/year)
- Water: {water_usage} L/day
- Energy: {energy_usage} kWh/day
- Waste: {waste_generation} kg/day

**Recommendations (ranked by overall impact):**

1. **[Alternative Name]**
   - CO₂ Emission: [X.X] kg/day (Saves [Y.Y] kg/day, [Z]% reduction)
   - Water Savings: [L/day]
   - Energy Savings: [kWh/day]
   - Waste Reduction: [kg/day]
   - Annual CO₂ Savings: [A] kg/year
   - Cost: [Estimate or "Minimal" or "Cost savings: $X/month"]
   - Difficulty: [Easy/Medium/Hard]
   - Timeframe: [Immediate/Short-term/Long-term]
   - Co-benefits: [Benefit 1], [Benefit 2], [Benefit 3]
   - Health Benefits: [Any health-related benefits]
   - Implementation: [2-3 sentence practical steps]

2. **[Alternative Name]**
   [Same format]

[Continue for remaining recommendations]

**Key Insight:** [1-2 sentence summary of the most impactful action considering all metrics]

VERIFICATION CHECKLIST (complete before responding):
☐ All emission values from verified alternatives
☐ Reduction percentages calculated correctly
☐ Annual savings = daily reduction × 365
☐ Recommendations ranked by overall environmental impact
☐ Water, energy, and waste impacts considered
☐ Cost estimates realistic and helpful
☐ Implementation steps are specific and actionable"""
        
        return prompt
    
    @staticmethod
    def recommendation_prompt(
        activity: str,
        emission: float,
        category: str,
        alternatives: List[str]
    ) -> str:
        """
        Generate prompt for creating alternative recommendations (legacy version)
        
        Args:
            activity: Current activity name
            emission: Current CO2 emission in kg/day
            category: Activity category
            alternatives: List of alternative activities from knowledge base
            
        Returns:
            Formatted prompt string
        """
        # Call enhanced version with no context
        return PromptTemplates.enhanced_recommendation_prompt(
            activity, emission, category, alternatives, None
        )
    
    @staticmethod
    def analysis_prompt(activities: List[Activity]) -> str:
        """
        Generate prompt for analyzing a dataset of activities
        
        Args:
            activities: List of Activity objects to analyze
            
        Returns:
            Formatted prompt string
        """
        # Calculate totals
        total_daily = sum(a.emission_kg_per_day for a in activities)
        total_annual = total_daily * 365
        
        # Format activities list
        activities_text = "\n".join([
            f"- {a.name} ({a.category.value}): {a.emission_kg_per_day} kg/day"
            for a in activities
        ])
        
        # Identify top emitters
        sorted_activities = sorted(activities, key=lambda x: x.emission_kg_per_day, reverse=True)
        top_3 = sorted_activities[:3]
        top_emitters_text = "\n".join([
            f"{i+1}. {a.name}: {a.emission_kg_per_day} kg/day ({a.emission_kg_per_day/total_daily*100:.1f}% of total)"
            for i, a in enumerate(top_3)
        ])
        
        prompt = f"""You are a CO₂ reduction advisor analyzing a user's daily activities.

ACTIVITY DATASET:
{activities_text}

SUMMARY STATISTICS:
- Total Daily Emission: {total_daily:.2f} kg/day
- Total Annual Emission: {total_annual:.1f} kg/year
- Number of Activities: {len(activities)}

TOP EMITTERS:
{top_emitters_text}

Provide a comprehensive analysis with:
1. Overall assessment of the carbon footprint
2. Key insights about emission patterns
3. Priority areas for reduction (focus on top emitters)
4. Specific recommendations for the highest-impact changes
5. Estimated total potential reduction if recommendations are followed

Be specific, actionable, and encouraging. Focus on practical steps the user can take."""
        
        return prompt
    
    @staticmethod
    def comparison_prompt(
        current: Activity,
        alternatives: List[Activity]
    ) -> str:
        """
        Generate prompt for comparing current activity with alternatives
        
        Args:
            current: Current activity
            alternatives: List of alternative activities
            
        Returns:
            Formatted prompt string
        """
        alternatives_text = "\n".join([
            f"- {a.name}: {a.emission_kg_per_day} kg/day "
            f"(Reduction: {current.emission_kg_per_day - a.emission_kg_per_day:.2f} kg/day, "
            f"{(current.emission_kg_per_day - a.emission_kg_per_day) / current.emission_kg_per_day * 100:.1f}%)"
            for a in alternatives
        ])
        
        prompt = f"""You are a CO₂ reduction advisor comparing different activity options.

CURRENT ACTIVITY:
- Name: {current.name}
- Category: {current.category.value}
- CO₂ Emission: {current.emission_kg_per_day} kg/day
- Annual Emission: {current.emission_kg_per_day * 365:.1f} kg/year

ALTERNATIVE OPTIONS:
{alternatives_text}

Provide a detailed comparison that includes:
1. Clear explanation of emission differences
2. Practical considerations for switching (cost, convenience, feasibility)
3. Recommended best alternative with justification
4. Implementation steps for making the switch
5. Long-term impact projection (annual savings)

Be balanced and realistic about the trade-offs involved."""
        
        return prompt
    
    @staticmethod
    def general_advice_prompt(query: str, context: List[str]) -> str:
        """
        Generate prompt for general CO₂ reduction advice
        
        Args:
            query: User's question or request
            context: Retrieved sustainability tips from knowledge base
            
        Returns:
            Formatted prompt string
        """
        context_text = "\n\n".join([f"Tip {i+1}:\n{tip}" for i, tip in enumerate(context)])
        
        prompt = f"""You are a CO₂ reduction advisor helping users understand and reduce their carbon footprint.

RELEVANT SUSTAINABILITY KNOWLEDGE:
{context_text}

USER QUERY:
{query}

Provide a helpful, informative response that:
1. Directly addresses the user's question
2. Uses the provided sustainability knowledge
3. Offers specific, actionable advice
4. Includes quantitative information when possible (emission values, percentages)
5. Encourages positive action

Be conversational, supportive, and practical."""
        
        return prompt
    
    @staticmethod
    def extraction_prompt(query: str) -> str:
        """
        Generate prompt for extracting activities and parameters from user query
        
        Args:
            query: User's natural language query
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Extract activity information from the following user query.

USER QUERY: {query}

Identify:
1. Activity name (e.g., "driving", "eating beef", "using air conditioning")
2. Quantity/frequency if mentioned (e.g., "20 km", "daily", "3 times per week")
3. Category (Transport/Household/Food/Lifestyle)
4. Any specific details (e.g., "petrol car", "electric vehicle")

Format your response as:
Activity: [activity name]
Quantity: [quantity/frequency or "not specified"]
Category: [category]
Details: [any additional details]

If multiple activities are mentioned, list each separately."""
        
        return prompt
