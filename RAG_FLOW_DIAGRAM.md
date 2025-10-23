# RAG Implementation - Visual Flow Diagrams

## 1. Complete RAG Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INPUT                                  │
│                  "I drive to college daily"                         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    QUERY PROCESSING                                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ QueryProcessor.process_query()                               │   │
│  │  • Extract intent: SINGLE_ACTIVITY                           │   │
│  │  • Extract activities: ["drive", "college"]                  │   │
│  │  • Extract parameters: {}                                    │   │
│  └──────────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    RETRIEVAL PHASE (RAG)                            │
│                                                                     │
│  ┌─────────────────────────┐    ┌─────────────────────────┐         │
│  │  Vector Store Search    │    │  Reference Data Search  │         │
│  │  ─────────────────────  │    │  ─────────────────────  │         │
│  │  Query: "reduce         │    │  Query: "driving car"   │         │
│  │  transport emissions"   │    │                         │         │
│  │                         │    │  Strategy 1: Keywords   │         │
│  │  Embedding Model:       │    │  "drive" → "driving car"│         │
│  │  paraphrase-MiniLM-L6   │    │                         │         │
│  │                         │    │  Strategy 2: Fuzzy Match│         │
│  │  Top-K: 3 documents     │    │  Cutoff: 0.3            │         │
│  │                         │    │                         │         │
│  │  Results:               │    │  Results:               │         │
│  │  1. "Switch to EV..."   │    │  1. Driving petrol car  │         │
│  │  2. "Use public trans"  │    │     4.6 kg CO2/day      │         │
│  │  3. "Carpool tips..."   │    │  2. Driving diesel car  │         │
│  │                         │    │     4.2 kg CO2/day      │         │
│  └─────────────────────────┘    └─────────────────────────┘         │
│                             │                │                      │
└─────────────────────────────┼────────────────┼──────────────────────┘
                              │                │
                              └────────┬───────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    AUGMENTATION PHASE                               │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Prompt Construction                                          │   │
│  │                                                              │   │
│  │ System: "You are a CO₂ reduction advisor..."                 │   │
│  │                                                              │   │
│  │ User Query: "I drive to college daily"                       │   │
│  │                                                              │   │
│  │ Current Activity: Driving petrol car (20 km)                 │   │
│  │ Current Emission: 4.6 kg CO2/day                             │   │
│  │ Category: Transport                                          │   │
│  │                                                              │   │
│  │ Retrieved Context:                                           │   │
│  │ Context 1: Switch to electric vehicle...                     │   │
│  │ Context 2: Use public transportation...                      │   │
│  │ Context 3: Carpool with colleagues...                        │   │
│  │                                                              │   │
│  │ Task: Provide 3-5 specific recommendations                   │   │
│  └──────────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    GENERATION PHASE                                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ LLM Client (Groq API)                                        │   │
│  │                                                              │   │
│  │ Model: llama-3.1-8b-instant                                  │   │
│  │ Temperature: 0.4                                             │   │
│  │ Max Tokens: 500                                              │   │
│  │                                                              │   │
│  │ Processing...                                                │   │
│  │ ⚡ Ultra-fast inference (~500ms)                             │   │
│  │                                                              │   │
│  │ Generated Response:                                          │   │
│  │ "1. Carpool or use public transport - Share rides with       │   │
│  │     classmates to reduce emissions by 50%...                 │   │
│  │  2. Switch to electric vehicle - Consider EV for zero        │   │
│  │     direct emissions...                                      │   │
│  │  3. Optimize driving habits - Maintain steady speeds...      │   │
│  │  4. Bike or walk for short distances...                      │   │
│  │  5. Use park-and-ride facilities..."                         │   │
│  └──────────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    POST-PROCESSING                                  │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Response Parser                                              │   │
│  │                                                              │   │
│  │ Parse LLM output into structured format:                     │   │
│  │                                                              │   │
│  │ Recommendation 1:                                            │   │
│  │   action: "Carpool or use public transport"                  │   │
│  │   emission_reduction_kg: 2.5                                 │   │
│  │   reduction_percentage: 54.3                                 │   │
│  │   difficulty: "Easy"                                         │   │
│  │   timeframe: "Immediate"                                     │   │
│  │   benefits: ["Cost savings", "Social interaction"]           │   │
│  │                                                              │   │
│  │ Recommendation 2: ...                                        │   │
│  │                                                              │   │
│  │ Calculate totals:                                            │   │
│  │   Total reduction: 8.5 kg CO2/day                            │   │
│  │   Annual savings: 3102.5 kg CO2/year                         │   │
│  └──────────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    USER RESPONSE                                    │
│                                                                     │
│  Current Activity: Driving petrol car (20 km)                       │
│  Current Emission: 4.6 kg CO2/day (1,679 kg/year)                   │
│                                                                     │
│  Top Recommendations:                                               │
│  1. ✓ Carpool or use public transport                               │
│     Reduction: 2.5 kg CO2/day | Difficulty: Easy                    │
│                                                                     │
│  2. ✓ Switch to electric vehicle                                    │
│     Reduction: 3.4 kg CO2/day | Difficulty: Hard                    │
│                                                                     │
│  3. ✓ Optimize driving habits                                       │
│     Reduction: 0.8 kg CO2/day | Difficulty: Easy                    │
│                                                                     │
│  Potential Annual Savings: 3,102.5 kg CO2                           │
└─────────────────────────────────────────────────────────────────────┘
```

## 2. Fallback Mechanism Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  User Query: "I charge my phone 10 times a day"                     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Query Processing                                                    │
│  Intent: GENERAL_ADVICE (no specific activity matched)              │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Try Activity Matching                                               │
│  ❌ No match in reference data (phone charging not in database)     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FALLBACK: Vector Store Search                                      │
│  Query: "phone charging emissions reduce"                           │
│  Retrieved: 5 general energy/electronics documents                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LLM Fallback Generation (RAG)                                      │
│                                                                       │
│  Prompt:                                                             │
│  "User asks: 'I charge my phone 10 times a day...'                  │
│   Context: [5 energy-saving documents]                              │
│   Provide helpful advice even though this is not in our database"   │
│                                                                       │
│  LLM generates context-aware response:                               │
│  1. Use solar power bank                                             │
│  2. Switch to eco-friendly charger                                   │
│  3. Limit charging frequency                                         │
│  4. Enable power-saving mode                                         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Response to User (with LLM-generated recommendations)              │
└─────────────────────────────────────────────────────────────────────┘
```

## 3. Dataset Upload with Unknown Categories

```
┌─────────────────────────────────────────────────────────────────────┐
│  User uploads CSV:                                                   │
│  Activity,Avg_CO2_Emission(kg/day),Category                         │
│  Charging laptop,4.69,Energy                                        │
│  Using heater,4.63,Energy                                           │
│  Driving car,1.24,Transport                                         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Validation                                                          │
│  ⚠️  Warning: Unknown category "Energy" found                       │
│  ✓ Will use AI fallback for recommendations                         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Category Mapping                                                    │
│  "Energy" → Category.HOUSEHOLD (internal mapping)                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Analysis                                                            │
│  Total: 10.56 kg CO2/day                                            │
│  Top Emitters:                                                       │
│    1. Charging laptop (4.69 kg/day)                                 │
│    2. Using heater (4.63 kg/day)                                    │
│    3. Driving car (1.24 kg/day)                                     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Recommendation Strategy Decision                                    │
│  ❓ Has unknown/energy categories? YES                              │
│  → Use LLM-based recommendations (RAG)                              │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Vector Store Search                                                 │
│  Query: "reduce emissions charging laptop heater"                   │
│  Retrieved: 3 relevant documents about energy efficiency            │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LLM Generation for Dataset                                          │
│                                                                       │
│  Prompt:                                                             │
│  "Dataset Summary:                                                   │
│   - Total: 10.56 kg CO2/day                                         │
│   - Top: Charging laptop (4.69), Using heater (4.63)               │
│                                                                       │
│   Context: [3 energy efficiency documents]                          │
│                                                                       │
│   Provide 5 recommendations..."                                      │
│                                                                       │
│  LLM Response:                                                       │
│  1. Switch to renewable energy sources                               │
│  2. Use energy-efficient heating alternatives                        │
│  3. Optimize device charging schedules                               │
│  4. Improve home insulation                                          │
│  5. Use smart power management                                       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Display Results with LLM-Generated Recommendations                 │
└─────────────────────────────────────────────────────────────────────┘
```

## 4. Component Interaction Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Streamlit UI                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐         │
│  │  Text Input    │  │  File Upload   │  │  Visualizations│         │
│  └────────┬───────┘  └────────┬───────┘  └────────▲───────┘         │
└───────────┼──────────────────┼──────────────────┼────────────────────┘
            │                  │                  │
            │                  │                  │
┌───────────▼──────────────────▼──────────────────┼────────────────────┐
│                      CO2ReductionAgent           │                    │
│  ┌──────────────────────────────────────────────┼─────────────────┐  │
│  │  process_query()                             │                 │  │
│  │  analyze_dataset()                           │                 │  │
│  │  generate_recommendations()                  │                 │  │
│  └──┬───────┬───────┬───────┬───────┬──────────┼─────────────────┘  │
└─────┼───────┼───────┼───────┼───────┼──────────┼────────────────────┘
      │       │       │       │       │          │
      │       │       │       │       │          │
┌─────▼──┐ ┌─▼─────┐ ┌▼─────┐ ┌▼────┐ ┌▼────────▼──┐
│ Query  │ │Vector │ │Refer │ │ LLM │ │ Dataset    │
│Process │ │Store  │ │Data  │ │Client│ │ Analyzer   │
└────┬───┘ └───┬───┘ └──┬───┘ └──┬──┘ └─────┬──────┘
     │         │        │        │          │
     │         │        │        │          │
     │    ┌────▼────────▼────────▼──────────▼────┐
     │    │      Recommendation Generator        │
     │    └──────────────────┬───────────────────┘
     │                       │
     └───────────────────────┴──────────────────────►
                             │
                    ┌────────▼────────┐
                    │  AgentResponse  │
                    │  - Emission     │
                    │  - Recommends   │
                    │  - Summary      │
                    └─────────────────┘
```

## 5. RAG vs Non-RAG Comparison

### Without RAG (Traditional Approach)

```
User Query → Activity Matching → Static Recommendations
                                        ↓
                              Limited to database
                              No context awareness
                              Generic responses
```

### With RAG (Our Implementation)

```
User Query → Activity Matching → Vector Search → Context Retrieval
                                        ↓
                              LLM with Context → Informed Generation
                                        ↓
                              Specific, contextual recommendations
                              Handles unknown queries
                              Personalized responses
```

## 6. Data Flow Summary

```
┌─────────────┐
│   User      │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  Input Layer                             │
│  • Text queries                          │
│  • CSV/Excel files                       │
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  Processing Layer                        │
│  • Query analysis                        │
│  • Intent classification                 │
│  • Activity extraction                   │
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  Retrieval Layer (RAG)                   │
│  • Vector store search                   │
│  • Reference data lookup                 │
│  • Context aggregation                   │
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  Generation Layer (RAG)                  │
│  • Prompt construction                   │
│  • LLM inference                         │
│  • Response parsing                      │
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  Output Layer                            │
│  • Structured recommendations            │
│  • Emission calculations                 │
│  • Visualizations                        │
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────┐
│   User      │
└─────────────┘
```

---

## Key Takeaways

1. **RAG = Retrieval + Augmentation + Generation**

   - Retrieval: Find relevant information
   - Augmentation: Add context to query
   - Generation: Create informed response

2. **Multi-Strategy Approach**

   - Primary: Reference data matching
   - Fallback: Vector store + LLM
   - Hybrid: Combine both for best results

3. **Graceful Degradation**

   - Known queries → Fast, structured responses
   - Unknown queries → LLM-powered fallback
   - Always provide helpful information

4. **Context is King**
   - More context = Better recommendations
   - Vector store provides relevant knowledge
   - LLM synthesizes context into actionable advice
