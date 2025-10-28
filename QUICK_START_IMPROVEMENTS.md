# Quick Start Guide - Improved Recommendation System

## What's New?

Your CO₂ Reduction AI Agent now has **significantly improved accuracy** with:

✅ **Validated calculations** - All emission reductions are mathematically verified  
✅ **Context-aware recommendations** - Personalized based on your budget and lifestyle  
✅ **Better ranking** - Most impactful and practical solutions shown first  
✅ **Source citations** - All data backed by EPA, IEA, and research studies  
✅ **User feedback** - Help improve the system by rating recommendations

## Getting Started

### 1. First Time Setup

```bash
# Install/update dependencies (if you haven't already)
pip install -r requirements.txt

# Reinitialize vector store with enhanced data
python scripts/init_vector_store.py
```

### 2. Running the Application

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Set your LLM provider
$env:LLM_PROVIDER="groq"  # Windows PowerShell
$env:GROQ_API_KEY="your_key_here"

# Run the app
streamlit run app.py
```

### 3. Using New Features

#### Context-Aware Queries

The system now understands context from your questions:

**Budget-aware:**

- "I need a **cheap** way to reduce car emissions" → Shows only low-cost options
- "I can **invest** in reducing emissions" → Includes long-term solutions

**Lifestyle-aware:**

- "I live in the **city**" → Emphasizes public transport, walking, cycling
- "I'm in a **rural** area" → Filters out public transport options

**Timeframe-aware:**

- "What can I do **right now**?" → Shows immediate actions only
- "I'm planning for the **future**" → Includes long-term investments

#### Example Queries

Try these improved queries:

```
1. "I drive 20 km daily in a petrol car. Need low-cost options."
   → Shows public transport, carpooling, cycling (not expensive EVs)

2. "I live in an urban apartment. How to reduce emissions?"
   → Shows space-appropriate options (no large garden solutions)

3. "What are immediate steps to reduce food emissions?"
   → Shows only quick-win food changes (not long-term projects)
```

#### Rating Recommendations

After getting recommendations, you'll see:

- **👍 Helpful** - Quick thumbs up
- **👎 Not Helpful** - Opens detailed feedback form
- **⚠️ Inaccurate** - Report specific issues

**Your feedback helps improve the system for everyone!**

---

## New Components Explained

### 1. Response Validator

**What it does:** Checks all emission calculations for accuracy

**How it helps you:**

- Ensures reduction percentages are correct
- Verifies annual savings calculations
- Catches LLM hallucinations

**Behind the scenes:**

```python
# Every recommendation is validated before showing you
is_valid = validator.validate_recommendation(rec)
if not is_valid:
    rec = validator.auto_correct_calculations(rec)
```

### 2. Recommendation Ranker

**What it does:** Scores recommendations on 5 factors:

1. **Emission reduction (35%)** - How much CO₂ you'll save
2. **Cost-effectiveness (25%)** - Bang for your buck
3. **Ease of implementation (20%)** - How hard to do
4. **Time to impact (15%)** - How quickly you see results
5. **Co-benefits (5%)** - Health, money, time savings

**How it helps you:**

- Best recommendations appear first
- Balanced between impact and practicality
- No overwhelming suggestions

### 3. Context-Aware Filtering

**What it does:** Removes impractical recommendations

**Examples:**

- Rural user → No public transport suggestions
- Low budget → No expensive equipment
- Apartment dweller → No backyard projects
- Want immediate action → No multi-month projects

### 4. Enhanced Knowledge Base

**What changed:**

- Every tip now includes:
  - Exact emission values (from EPA, IEA)
  - Cost estimates (not just "cheap" or "expensive")
  - Implementation timeframes
  - Prerequisites needed
  - Co-benefits (health, money savings)
  - Source citations

**Example:**

**Before:**

```
Use public transport - reduces emissions by ~70%
```

**After:**

```
Use public transportation instead of driving
- Current: 4.6 kg CO₂/day (petrol car 20km)
- Alternative: 1.2 kg CO₂/day (bus)
- Reduction: 3.4 kg CO₂/day (74%)
- Annual savings: 1,241 kg CO₂
- Cost: $50-150/month transit pass
- Timeframe: Immediate (1 week)
- Prerequisites: Available routes, schedule flexibility
- Co-benefits: Save $300-500/month, productive commute time
- Source: EPA 2024, APTA Public Transportation Fact Book
```

---

## Testing the Improvements

### Try These Comparison Queries

**Before:** "reduce car emissions"
**Now:** "I need low-cost immediate ways to reduce car emissions in the city"

→ You'll see more relevant, practical options ranked by actual impact

### Validate Calculations

Ask for specific emission reductions and verify:

**Query:** "If I switch from a petrol car to an electric car, how much CO₂ will I save?"

**Check the math:**

- Current: 4.6 kg/day
- Alternative: 1.2 kg/day
- Reduction: 3.4 kg/day ✓
- Percentage: (3.4/4.6) × 100 = 73.9% ✓
- Annual: 3.4 × 365 = 1,241 kg ✓

### Test Context Understanding

**Urban + Low Budget:**

```
"I live in the city and need cheap ways to reduce transport emissions"
```

Should see: Public transport, cycling, walking, carpooling
Should NOT see: Electric cars, solar panels, rural solutions

**Rural + High Budget:**

```
"I live in the countryside and can invest in emission reduction"
```

Should see: Electric vehicle, solar panels, heat pumps
Should NOT see: Public transport, metro options

---

## Configuration Options

### Adjust Ranking Priorities

Edit `components/recommendation_ranker.py` if you want different priorities:

```python
# Example: Prioritize cost over emission reduction
weights = RankingWeights(
    emission_reduction=0.25,      # Default: 0.35
    cost_effectiveness=0.40,      # Default: 0.25 (INCREASED)
    ease_of_implementation=0.20,  # Default: 0.20
    time_to_impact=0.10,          # Default: 0.15
    co_benefits=0.05              # Default: 0.05
)
```

### Adjust Validation Strictness

Edit `components/response_validator.py`:

```python
# More lenient (allow 30% variance in emission values)
tolerance=0.30  # Default: 0.25

# Stricter calculation checks (allow only 3% error)
tolerance=3.0  # Default: 5.0 percentage points
```

### Context Filter Customization

Edit `components/context_manager.py` in `_passes_context_filters()`:

```python
# Example: Allow high-cost items for medium budget if very high impact
if context.budget == Budget.MEDIUM and rec_cost == 'high':
    if recommendation.get('reduction_percentage', 0) < 80:  # Default: 60
        return False
```

---

## Troubleshooting

### Issue: Not seeing any recommendations

**Solution:** Context filters might be too strict

```python
# In query, be less specific
# Instead of: "I need free immediate urban solutions"
# Try: "reduce transport emissions"
```

### Issue: Recommendations seem off

**Use the feedback button!**

1. Click **⚠️ Inaccurate**
2. Describe the issue
3. We'll review and improve

### Issue: Math doesn't add up

**This should be rare now!** But if you spot errors:

1. Check if validation is enabled in agent.py
2. Report via feedback system
3. Validator will be updated

---

## Advanced Usage

### For Developers

#### Running Tests

```bash
# Install pytest
pip install pytest

# Run all tests
pytest tests/test_recommendations.py -v

# Run specific test
pytest tests/test_recommendations.py::TestResponseValidator -v
```

#### Analyzing Feedback

```python
from components.feedback_collector import FeedbackCollector

collector = FeedbackCollector()

# Get statistics
stats = collector.get_statistics()
print(f"Average rating: {stats['average_rating']:.2f}/5")
print(f"Accuracy rate: {stats['accurate_percentage']:.1f}%")

# Find problem areas
low_rated = collector.get_low_rated_queries(threshold=2)
inaccurate = collector.get_inaccuracy_reports()

# Export for analysis
collector.export_feedback("analysis.json")
```

#### Manual Validation

```python
from components.response_validator import ResponseValidator

validator = ResponseValidator("data/reference_activities.csv")

# Validate a recommendation
is_valid, issues = validator.validate_recommendation({
    'current_emission': 4.6,
    'alternative_emission': 1.2,
    'reduction_percentage': 74.0,
    'annual_savings_kg': 1241.0,
    ...
})

if not is_valid:
    print("Issues found:", issues)
    corrected = validator.auto_correct_calculations(rec)
```

---

## What's Next?

The system is now much more accurate, but we're continuously improving:

### Upcoming Features

- [ ] A/B testing different prompt variations
- [ ] Machine learning from user feedback to optimize ranking
- [ ] Regional customization (US vs EU vs Asia emission factors)
- [ ] Cost-benefit calculator
- [ ] Recommendation explanations ("Why this ranks #1")

### How You Can Help

1. **Use the feedback buttons** - Every rating helps
2. **Report inaccuracies** - We'll fix them in the knowledge base
3. **Share edge cases** - Unusual queries help us improve
4. **Suggest features** - What would make this more useful?

---

## Support

### Having Issues?

1. **Check the logs** - Look for validation errors in terminal
2. **Review feedback** - See what others have reported
3. **Run tests** - `pytest tests/ -v` to check system health
4. **Read docs** - See [ACCURACY_IMPROVEMENTS.md](ACCURACY_IMPROVEMENTS.md) for details

### Want to Contribute?

- Update knowledge base with better data
- Improve prompt templates
- Add more test cases
- Submit feedback from your usage

---

## Summary

You now have a **significantly more accurate** CO₂ reduction recommendation system that:

✅ **Validates all calculations** automatically  
✅ **Personalizes recommendations** to your situation  
✅ **Ranks by actual impact** and practicality  
✅ **Sources all data** from reputable studies  
✅ **Learns from feedback** to improve over time

**Start using it now** and see the difference in recommendation quality!

```bash
streamlit run app.py
```

Then try a context-rich query like:

```
"I live in an urban apartment, have a low budget, and drive 10 km daily to work.
What are immediate ways I can reduce my carbon footprint?"
```

You'll see relevant, accurate, actionable recommendations ranked by real impact. 🌱
