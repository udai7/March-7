# CO₂ Reduction AI Agent - User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Formulating Effective Queries](#formulating-effective-queries)
4. [Uploading and Formatting Datasets](#uploading-and-formatting-datasets)
5. [Interpreting Results](#interpreting-results)
6. [Best Practices](#best-practices)
7. [Advanced Usage](#advanced-usage)
8. [FAQ](#faq)

## Introduction

The CO₂ Reduction AI Agent is designed to help you understand and reduce your carbon footprint through AI-powered recommendations. This guide will help you get the most out of the system.

### What Can the Agent Do?

- Analyze individual activities and suggest lower-emission alternatives
- Compare multiple options to help you make informed decisions
- Process your activity datasets to identify your biggest emission sources
- Provide quantitative emission reductions and annual savings projections
- Offer practical, actionable recommendations tailored to your situation

## Getting Started

### Launching the Application

1. Ensure Ollama is running with a model loaded (e.g., `ollama pull llama3`)
2. Activate your virtual environment
3. Run: `streamlit run app.py`
4. Open your browser to `http://localhost:8501`

### Interface Overview

The application has two main interaction modes:

**Query Mode**: Ask questions in natural language

- Located in the main content area
- Type your question and click Submit
- View AI-generated recommendations

**Dataset Upload Mode**: Analyze your activity data

- Located in the sidebar
- Upload CSV or Excel files
- View comprehensive analysis

## Formulating Effective Queries

### Query Types

The agent understands several types of queries:

#### 1. Single Activity Queries

Ask about a specific activity you want to optimize.

**Good Examples:**

- "I drive 20 km daily using a petrol car. How can I reduce my carbon footprint?"
- "I eat beef 3 times a week. What are lower-emission alternatives?"
- "My household uses 15 kWh of electricity per day. How can I reduce this?"

**Tips:**

- Include specific quantities (distance, frequency, amount)
- Mention the current method (petrol car, beef, electricity)
- Be specific about timeframes (daily, weekly, monthly)

**Poor Examples:**

- "How do I save the planet?" (too vague)
- "Cars" (no context or question)
- "Tell me about emissions" (no specific activity)

#### 2. Comparison Queries

Compare two or more options to make informed decisions.

**Good Examples:**

- "What's better for the environment: driving a petrol car or taking public transport for my 15 km commute?"
- "Should I switch from beef to chicken or go fully vegetarian?"
- "Compare the carbon footprint of heating with natural gas versus electric heating"

**Tips:**

- Clearly state both/all options you want to compare
- Include relevant context (distance, frequency, usage)
- Ask for specific comparisons

**Poor Examples:**

- "Which is better?" (no options specified)
- "Car vs bus" (no context or quantities)

#### 3. General Advice Queries

Get broad recommendations for reducing emissions.

**Good Examples:**

- "What are the top 3 things I can do to reduce my household carbon emissions?"
- "How can I make my daily commute more sustainable?"
- "Give me some easy wins for reducing my carbon footprint in the kitchen"

**Tips:**

- Specify a category or area (household, transport, food)
- Ask for prioritized or ranked recommendations
- Indicate your preference (easy wins, biggest impact, etc.)

**Poor Examples:**

- "Help me" (too vague)
- "What should I do?" (no context)

#### 4. Category-Specific Queries

Focus on a particular emission category.

**Categories:**

- **Transport**: Commuting, travel, vehicle choices
- **Household**: Energy use, heating, appliances
- **Food**: Diet choices, meal planning, food sourcing
- **Lifestyle**: Shopping, waste, consumption habits

**Good Examples:**

- "What transport changes would have the biggest impact on my emissions?"
- "How can I reduce my household energy consumption?"
- "What dietary changes are most effective for reducing CO₂?"

### Query Best Practices

**DO:**

- ✅ Be specific with numbers and quantities
- ✅ Provide context about your current situation
- ✅ Ask clear, focused questions
- ✅ Mention timeframes (daily, weekly, monthly)
- ✅ Specify categories when relevant

**DON'T:**

- ❌ Use overly vague or general questions
- ❌ Ask multiple unrelated questions in one query
- ❌ Omit important context or quantities
- ❌ Use ambiguous terms without clarification

### Example Query Progression

**Vague → Better → Best:**

1. "Cars and emissions"
   → "How do cars affect emissions?"
   → "I drive 30 km daily in a petrol car. What are lower-emission alternatives?"

2. "Food carbon footprint"
   → "How does food affect my carbon footprint?"
   → "I eat beef 4 times a week. What dietary changes would reduce my emissions the most?"

3. "Save energy"
   → "How can I save energy at home?"
   → "My household uses 20 kWh daily. What are the most effective ways to reduce this?"

## Uploading and Formatting Datasets

### When to Upload a Dataset

Upload a dataset when you want to:

- Analyze multiple activities at once
- Identify your top emission sources
- Get prioritized recommendations across all activities
- Track your carbon footprint over time

### Dataset Format Requirements

Your dataset must include these three columns:

| Column Name              | Description                 | Data Type   | Example                   |
| ------------------------ | --------------------------- | ----------- | ------------------------- |
| Activity                 | Description of the activity | Text        | "Driving petrol car 20km" |
| Avg_CO2_Emission(kg/day) | Daily CO₂ emission          | Number (≥0) | 4.6                       |
| Category                 | Activity category           | Text        | "Transport"               |

**Valid Categories:**

- Transport
- Household
- Food
- Lifestyle

### Creating Your Dataset

#### Option 1: CSV File

Create a file named `my_activities.csv`:

```csv
Activity,Avg_CO2_Emission(kg/day),Category
Driving petrol car 20km,4.6,Transport
Eating beef 3 times/week,3.3,Food
Electric heating,2.5,Household
Online shopping weekly,1.2,Lifestyle
Taking bus 10km,0.8,Transport
Eating chicken 4 times/week,1.5,Food
Using LED lights,0.3,Household
Recycling waste,0.1,Lifestyle
```

#### Option 2: Excel File

Create an Excel file (.xlsx or .xls) with the same structure:

| Activity                 | Avg_CO2_Emission(kg/day) | Category  |
| ------------------------ | ------------------------ | --------- |
| Driving petrol car 20km  | 4.6                      | Transport |
| Eating beef 3 times/week | 3.3                      | Food      |
| Electric heating         | 2.5                      | Household |

### Finding Emission Values

If you don't know the emission values for your activities:

1. **Use the Reference Dataset**: Check `data/reference_activities.csv` for common activities
2. **Online Calculators**: Use carbon footprint calculators to estimate values
3. **Ask the Agent**: Query the agent about specific activities to get estimates
4. **Utility Bills**: Some utility companies provide CO₂ emission data

### Common Dataset Errors and Fixes

#### Error: "Missing required columns"

**Problem**: Column names don't match exactly

**Fix**: Ensure column names are:

- `Activity` (not "activity" or "Activity Name")
- `Avg_CO2_Emission(kg/day)` (exact spelling with parentheses)
- `Category` (not "category" or "Type")

#### Error: "Invalid category value"

**Problem**: Category doesn't match allowed values

**Fix**: Use only these exact values:

- Transport
- Household
- Food
- Lifestyle

#### Error: "Invalid emission value"

**Problem**: Emission values are not numbers or are negative

**Fix**:

- Use numbers only (no text like "high" or "low")
- Ensure values are ≥ 0
- Remove any currency symbols or units from the number field

#### Error: "File format not supported"

**Problem**: File is not CSV or Excel

**Fix**:

- Save as .csv, .xlsx, or .xls
- If using Google Sheets, download as CSV or Excel

### Dataset Best Practices

**DO:**

- ✅ Include all regular activities (daily, weekly, monthly)
- ✅ Use consistent units (kg CO₂ per day)
- ✅ Be specific in activity descriptions
- ✅ Double-check emission values for accuracy
- ✅ Categorize activities correctly

**DON'T:**

- ❌ Mix different time periods (some daily, some monthly)
- ❌ Include duplicate activities
- ❌ Use vague activity descriptions
- ❌ Leave cells empty (use 0 if needed)
- ❌ Include activities with unknown emissions (estimate or research first)

## Interpreting Results

### Understanding the Response Structure

When you submit a query or upload a dataset, the agent provides:

#### 1. Current Emission Summary

**What it shows:**

- Your baseline CO₂ output for the activity/activities
- Daily emission in kg CO₂/day
- Annual projection (daily × 365)

**Example:**

```
Current Emission: 4.6 kg CO₂/day
Annual Emission: 1,679 kg CO₂/year
```

**How to interpret:**

- This is your starting point before any changes
- Annual values help you see long-term impact
- Compare to average per-person emissions (varies by country)

#### 2. Recommendations

Each recommendation includes:

**Action**: What to do

- Clear, specific alternative or change
- Example: "Switch to public transport for daily commute"

**Emission Reduction**: Quantitative impact

- Absolute: kg CO₂/day saved
- Percentage: % reduction from current
- Example: "Reduces emissions by 3.8 kg CO₂/day (83% reduction)"

**Implementation Difficulty**: Effort required

- **Easy**: Minimal effort, immediate changes
- **Medium**: Some planning or investment needed
- **Hard**: Significant lifestyle change or major investment

**Timeframe**: When you can implement

- **Immediate**: Can start today
- **Short-term**: Within weeks or months
- **Long-term**: Requires planning or major changes

**Additional Benefits**: Beyond CO₂ reduction

- Cost savings
- Health benefits
- Quality of life improvements
- Example: "Saves $50/month on fuel, improves fitness"

#### 3. Annual Savings Projection

**What it shows:**

- Total potential CO₂ reduction per year if you implement all recommendations
- Cumulative impact of multiple changes

**Example:**

```
Total Annual Savings: 1,387 kg CO₂/year
```

**How to interpret:**

- This assumes you implement all recommendations
- You can pick and choose based on feasibility
- Even partial implementation has significant impact

### Comparing Recommendations

Recommendations are typically ranked by:

1. **Emission Reduction Potential**: Highest impact first
2. **Implementation Difficulty**: Easier changes highlighted
3. **Timeframe**: Immediate actions prioritized

**How to prioritize:**

- Start with "Easy" + "Immediate" recommendations
- Focus on highest emission reductions
- Consider additional benefits (cost, health)
- Balance quick wins with long-term changes

### Dataset Analysis Results

When you upload a dataset, you'll see:

#### Top Emitters

**What it shows:**

- Your 3 highest emission activities
- Percentage of total emissions each represents

**Example:**

```
Top Emitters:
1. Driving petrol car 20km: 4.6 kg CO₂/day (45% of total)
2. Eating beef 3x/week: 3.3 kg CO₂/day (32% of total)
3. Electric heating: 2.5 kg CO₂/day (24% of total)
```

**How to use:**

- Focus on these activities first for maximum impact
- Even small changes to top emitters have big results
- Consider which are easiest to change

#### Category Breakdown

**What it shows:**

- Total emissions by category
- Helps identify which areas of life have highest impact

**Example:**

```
Category Breakdown:
- Transport: 5.4 kg CO₂/day (53%)
- Food: 4.8 kg CO₂/day (47%)
- Household: 2.8 kg CO₂/day (27%)
- Lifestyle: 1.3 kg CO₂/day (13%)
```

**How to use:**

- Identify categories where you can make the most difference
- Balance changes across categories
- Some categories may be easier to change than others

### Understanding Emission Values

**Context for CO₂ Emissions:**

- **Average per-person emissions** (varies by country):

  - USA: ~45 kg CO₂/day
  - EU: ~20 kg CO₂/day
  - Global average: ~12 kg CO₂/day

- **Common activity emissions:**

  - Driving 1 km (petrol car): ~0.2 kg CO₂
  - Eating 1 kg beef: ~27 kg CO₂
  - 1 kWh electricity: ~0.5 kg CO₂ (varies by grid)
  - 1 hour flight: ~90 kg CO₂

- **Reduction targets:**
  - Paris Agreement goal: ~6 kg CO₂/day per person by 2030
  - Net-zero goal: ~2 kg CO₂/day per person by 2050

### Realistic Expectations

**What the agent can do:**

- Provide accurate emission calculations based on reference data
- Suggest evidence-based alternatives
- Quantify potential reductions
- Offer practical implementation guidance

**What the agent cannot do:**

- Account for every personal circumstance
- Guarantee exact emission values (estimates vary)
- Make decisions for you
- Implement changes automatically

**Remember:**

- Every reduction counts, no matter how small
- Perfect is the enemy of good - start somewhere
- Sustainable changes are better than unsustainable perfection
- Focus on what you can control

## Best Practices

### For Query Mode

1. **Start Simple**: Begin with one activity or question
2. **Be Specific**: Include numbers, frequencies, and context
3. **Iterate**: Refine your query based on initial results
4. **Ask Follow-ups**: Dig deeper into specific recommendations
5. **Compare Options**: Use comparison queries to make informed decisions

### For Dataset Mode

1. **Start Small**: Begin with 5-10 main activities
2. **Be Accurate**: Research emission values when possible
3. **Update Regularly**: Track changes over time
4. **Focus on Top Emitters**: Prioritize high-impact activities
5. **Categorize Correctly**: Ensures relevant recommendations

### For Implementation

1. **Start with Easy Wins**: Build momentum with simple changes
2. **Track Progress**: Monitor your emission reductions
3. **Be Consistent**: Small, sustained changes beat sporadic efforts
4. **Combine Changes**: Multiple small changes add up
5. **Share Knowledge**: Help others reduce their footprint too

## Advanced Usage

### Combining Query and Dataset Modes

1. Upload your dataset to identify top emitters
2. Use query mode to explore specific alternatives for each top emitter
3. Compare multiple options before deciding
4. Update your dataset with new activities
5. Re-analyze to track progress

### Customizing for Your Situation

**Regional Differences:**

- Electricity emissions vary by grid (coal vs. renewable)
- Public transport availability differs by location
- Climate affects heating/cooling needs

**Personal Circumstances:**

- Budget constraints
- Physical abilities
- Family size
- Living situation (rent vs. own)
- Work requirements

**Mention these in your queries for more relevant recommendations.**

### Tracking Progress Over Time

1. Create a baseline dataset of current activities
2. Implement recommendations gradually
3. Update dataset monthly with new activities
4. Compare total emissions over time
5. Celebrate reductions and identify new opportunities

## FAQ

### Q: How accurate are the emission calculations?

A: Emission values are based on research and reference data, but actual emissions vary based on many factors (vehicle efficiency, electricity grid mix, etc.). Treat values as estimates for comparison purposes.

### Q: Can I use this for business/organizational carbon footprinting?

A: The system is designed for individual/household use. For organizational carbon accounting, consider specialized tools that meet reporting standards (GHG Protocol, ISO 14064).

### Q: What if my activity isn't in the reference dataset?

A: The agent can still provide recommendations based on similar activities in its knowledge base. You can also research emission values and add them to your dataset.

### Q: How often should I update my dataset?

A: Monthly updates are ideal for tracking progress. Update immediately after making significant lifestyle changes to see the impact.

### Q: Can I export my results?

A: Currently, results are displayed in the web interface. You can take screenshots or copy text. Future versions may include export functionality.

### Q: What if I disagree with a recommendation?

A: Recommendations are suggestions, not requirements. Consider your personal circumstances, values, and constraints. The agent provides information to help you make informed decisions.

### Q: How do I know which recommendations to prioritize?

A: Focus on:

1. Highest emission reductions
2. Easiest to implement
3. Lowest cost or highest savings
4. Best fit for your lifestyle
5. Additional benefits beyond CO₂

### Q: Can I add my own sustainability tips to the knowledge base?

A: Yes! Edit `data/sustainability_tips.txt` and run `python scripts/init_vector_store.py` to update the vector database.

### Q: What if the agent gives inconsistent answers?

A: LLM responses can vary slightly between queries. For critical decisions, run the query multiple times or use dataset mode for more structured analysis.

### Q: How can I improve response quality?

A:

- Be more specific in your queries
- Include relevant context and quantities
- Try different LLM models in config.py
- Adjust temperature settings for more/less creative responses
- Update the knowledge base with better information

---

## Need More Help?

- Check the main README.md for installation and troubleshooting
- Review example queries in `data/example_queries.txt`
- Open an issue on GitHub for bugs or feature requests
- Consult the design document for technical details

**Remember**: Every step toward reducing your carbon footprint matters. Start where you are, use what you have, do what you can.
