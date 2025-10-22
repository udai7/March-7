"""
Prompt templates for LLM interactions
"""
from typing import List, Dict, Any
from models.data_models import Activity


class PromptTemplates:
    """Collection of structured prompts for different LLM tasks"""
    
    @staticmethod
    def recommendation_prompt(
        activity: str,
        emission: float,
        category: str,
        alternatives: List[str]
    ) -> str:
        """
        Generate prompt for creating alternative recommendations
        
        Args:
            activity: Current activity name
            emission: Current CO2 emission in kg/day
            category: Activity category
            alternatives: List of alternative activities from knowledge base
            
        Returns:
            Formatted prompt string
        """
        alternatives_text = "\n".join([f"- {alt}" for alt in alternatives]) if alternatives else "No specific alternatives provided"
        
        prompt = f"""You are a CO₂ reduction advisor. A user is currently doing the following activity:

Activity: {activity}
Category: {category}
Current CO₂ Emission: {emission} kg/day
Annual Emission: {emission * 365:.1f} kg/year

Based on the following alternative suggestions:
{alternatives_text}

Provide 3-5 specific, actionable recommendations to reduce CO₂ emissions. For each recommendation:
1. Describe the alternative action clearly
2. Estimate the emission reduction in kg/day
3. Calculate the reduction percentage
4. Specify implementation difficulty (Easy/Medium/Hard)
5. Indicate timeframe (Immediate/Short-term/Long-term)
6. List additional benefits (cost savings, health, etc.)

Format your response as follows:

RECOMMENDATION 1:
Action: [specific action]
Emission Reduction: [X.X kg/day]
Reduction Percentage: [XX%]
Difficulty: [Easy/Medium/Hard]
Timeframe: [Immediate/Short-term/Long-term]
Additional Benefits: [benefit 1], [benefit 2]

RECOMMENDATION 2:
[same format]

Continue for all recommendations."""
        
        return prompt
    
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
