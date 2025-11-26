# 🌿 Environmental Impact AI Agent - New Features

This document outlines the new features added to enhance the Environmental Impact AI Agent, transforming it from a CO₂-only calculator into a comprehensive sustainability platform.

---

## 📊 Environmental Impact Dashboard

A fully interactive dashboard for tracking, comparing, and understanding your environmental footprint.

### Features

#### 🔢 Impact Calculator

Calculate your personalized daily environmental footprint by entering:

- **Transportation**: Car usage, public transit, flights, biking/walking
- **Home Energy**: Electricity consumption, renewable energy %, gas usage, heating type
- **Food & Diet**: Diet type (vegan to heavy meat), local food percentage, food waste level
- **Water Usage**: Shower duration, fixture types, garden watering

**Output**: Detailed breakdown showing:

- Daily CO₂ emissions (kg)
- Water usage (liters)
- Energy consumption (kWh)
- Annual carbon footprint (tons)
- Real-world equivalents (trees needed, driving km equivalent)

#### 📊 Activity Comparison

Compare the environmental impact of everyday activities across categories:

| Category          | What You Can Compare                                        |
| ----------------- | ----------------------------------------------------------- |
| 🚗 Transportation | Petrol car vs Electric vs Bus vs Bike (CO₂, cost, time)     |
| 🍽️ Food & Meals   | Beef vs Chicken vs Vegetarian vs Vegan (CO₂, water, land)   |
| 🏠 Household      | Shower vs Bath, Dishwasher vs Handwash (energy, water)      |
| 🛍️ Shopping       | New vs Refurbished electronics, Fast fashion vs Second-hand |
| 🎮 Entertainment  | Streaming vs Gaming vs Reading (energy consumption)         |

#### 🎯 Goal Tracker

Set and monitor daily sustainability targets:

- Set targets for CO₂, water, energy, and waste
- Log your actual daily usage
- Visual progress bars and color-coded status
- Overall sustainability score (0-100)

#### 🌍 Footprint Analyzer

Quick annual carbon footprint estimate based on:

- Country of residence
- Household size
- Housing type
- Car usage patterns
- Diet preferences
- Flight frequency

Compares your footprint to world and country averages with personalized reduction suggestions.

#### 💡 Eco Tips

Category-specific actionable tips with:

- Impact estimates (savings in $ or CO₂)
- Difficulty ratings (🟢 Easy, 🟡 Medium, 🔴 Advanced)
- Weekly eco-challenges to stay motivated

---

## 💰 Financial Impact Calculator

Calculate the financial benefits of making eco-friendly choices.

### Features

#### 💵 Cost Savings Calculator

Calculate savings from lifestyle changes:

| Category              | Examples                                                |
| --------------------- | ------------------------------------------------------- |
| 🚗 Transport          | Switching from petrol car to EV, e-bike, public transit |
| ⚡ Energy             | Reducing electricity usage by X%                        |
| 💧 Water              | Installing low-flow fixtures, reducing usage            |
| 🍽️ Food & Groceries   | Home cooking vs restaurants, meal prep savings          |
| 🌡️ Heating & Cooling  | Heat pump vs gas furnace efficiency comparison          |
| 🔌 Appliance Upgrades | Old refrigerator vs Energy Star model                   |
| 📱 Subscriptions      | Reducing streaming/gaming subscriptions                 |

**Output**: Daily, monthly, annual, and 10-year savings projections.

#### 📈 Green Investment ROI Calculator

Analyze returns on eco-friendly investments:

**Investment Categories**:

- ⚡ Energy & Power: Solar panels, battery storage, solar water heater
- 🏠 Home Improvement: Insulation, double-glazed windows, heat pump, green roof
- 🚗 Transportation: Electric vehicle, e-bike, e-scooter, home EV charger
- 💡 Efficiency Upgrades: LED lighting, smart thermostat, efficient appliances
- 💧 Water Conservation: Rainwater harvesting, smart irrigation
- ♻️ Waste Management: Composting system

**Calculations Include**:

- Initial cost and payback period
- Net Present Value (NPV)
- Internal Rate of Return (IRR)
- Total lifetime savings
- CO₂ reduction over lifetime
- Comparison tables across investment options

#### 🏠 Utility Cost Comparison

Compare utility costs across different scenarios:

| Comparison Type        | What It Compares                                                |
| ---------------------- | --------------------------------------------------------------- |
| 📊 General Utility     | Current vs optimized electricity/gas/water usage                |
| ⚡ Electricity Sources | Grid vs Solar vs Wind vs Green energy plans                     |
| 🔥 Heating Systems     | Gas furnace vs Heat pump vs Geothermal (by home size & climate) |
| ❄️ Cooling Systems     | Window AC vs Central AC vs Mini-split (by SEER rating)          |
| 💧 Water Systems       | Standard vs Low-flow vs Rainwater harvesting                    |
| 🚗 Transportation Fuel | Gas cars vs Hybrids vs EVs (annual fuel costs)                  |
| 🏡 Home Energy Audit   | Quick audit with personalized improvement recommendations       |

#### 🌱 Carbon Credits Calculator

Estimate carbon credit values based on your emissions or reductions:

- Calculate annual CO₂ emissions
- Estimate carbon credit value at current market prices
- Project potential earnings from carbon offset programs

---

## 🧾 Receipt Scanner

Analyze shopping receipts to understand the environmental impact of your purchases.

### How It Works

1. **Upload Receipt**: Take a photo or upload an image of your shopping receipt
2. **AI Analysis**: Uses Groq's LLaMA 4 Scout vision model to:
   - Extract product names and prices
   - Categorize items (Food, Electronics, Clothing, etc.)
   - Identify eco-friendly products
3. **Impact Assessment**: For each product category, calculates:
   - CO₂ footprint (kg)
   - Water usage (liters)
   - Waste generated (grams)
   - Eco-friendliness score

### Product Categories Analyzed

| Category          | Environmental Factors Considered           |
| ----------------- | ------------------------------------------ |
| 🥬 Fresh Produce  | Local vs imported, organic vs conventional |
| 🥩 Meat & Seafood | Type of meat, sourcing method              |
| 🥛 Dairy          | Processing, packaging                      |
| 🍞 Packaged Food  | Packaging waste, processing energy         |
| 🧴 Personal Care  | Chemicals, plastic packaging               |
| 🧹 Cleaning       | Chemical content, packaging                |
| 👕 Clothing       | Material type, manufacturing               |
| 📱 Electronics    | E-waste potential, manufacturing footprint |
| 🏠 Home & Garden  | Material sustainability                    |

### Output Includes

- **Receipt Summary**: Total items, eco-friendly count, total impact
- **Per-Item Analysis**: Individual product impacts
- **Recommendations**: Eco-friendly alternatives for high-impact items
- **Sustainability Score**: Overall rating for your shopping trip

---

## 🚀 Quick Start

1. **Run the app**:

   ```bash
   streamlit run app.py
   ```

2. **Navigate to features**:

   - Use the main tabs to switch between:
     - 🤖 Ask Question (AI chat)
     - 📁 Upload Dataset (analyze CSV files)
     - 📈 Dashboard (Environmental Impact Dashboard)
     - 💰 Financial Calculator
     - 🧾 Receipt Scanner

3. **Set up API key** (for Receipt Scanner):
   - Create a `.env` file with: `GROQ_API_KEY=your_api_key_here`
   - Get a free API key from [Groq Console](https://console.groq.com)

---

## 📋 Requirements

New dependencies added:

```
groq>=0.4.0          # For LLM vision analysis
Pillow>=10.0.0       # For image processing
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 🎯 Use Cases

| Goal                                 | Recommended Feature                       |
| ------------------------------------ | ----------------------------------------- |
| Understand my daily carbon footprint | Dashboard → Impact Calculator             |
| Compare beef vs plant-based meals    | Dashboard → Activity Comparison           |
| Should I buy solar panels?           | Financial Calculator → ROI Calculator     |
| How much can I save switching to EV? | Financial Calculator → Cost Savings       |
| Analyze my grocery shopping impact   | Receipt Scanner                           |
| Set and track sustainability goals   | Dashboard → Goal Tracker                  |
| Get personalized eco tips            | Dashboard → Eco Tips                      |
| Compare heating system options       | Financial Calculator → Utility Comparison |

---

## 📈 Future Enhancements

- [ ] Historical tracking of goals and progress
- [ ] Integration with smart home devices
- [ ] Social sharing and community challenges
- [ ] Gamification with badges and achievements
- [ ] Export reports to PDF
- [ ] Multi-language support

---

_Built with ❤️ for a sustainable future_
