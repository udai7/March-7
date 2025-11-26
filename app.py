"""
Environmental Impact AI Agent - Streamlit Application

Main entry point for the Streamlit web interface that helps users
understand and reduce their environmental footprint through comprehensive
analysis of CO₂, water, energy, and waste impacts.
"""

import streamlit as st
import pandas as pd
from typing import Optional, List
import config
from components.agent import CO2ReductionAgent
from components.llm_client import LLMClient
from components.vector_store import VectorStore
from components.reference_data import ReferenceDataManager
from components.feedback_collector import FeedbackCollector
from components.environmental_scorer import EnvironmentalScorer
from components.financial_calculator import FinancialCalculator, InvestmentType, CostSavings, ROIResult
from components.receipt_scanner import ReceiptScanner, ProductCategory, ReceiptAnalysis
from models.data_models import AgentResponse, DatasetAnalysis


# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def initialize_agent() -> Optional[CO2ReductionAgent]:
    """
    Initialize and cache the CO2 Reduction Agent with all components.
    
    Returns:
        Initialized CO2ReductionAgent or None if initialization fails
    """
    # Initialize LLM client
    llm_client = LLMClient(
        provider=config.LLM_PROVIDER,
        model_name=config.LLM_MODEL,
        base_url=config.LLM_BASE_URL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS
    )
    
    # Initialize vector store
    vector_store = VectorStore(
        collection_name="sustainability_tips",
        persist_directory=config.VECTOR_DB_PATH,
        embedding_model=config.EMBEDDING_MODEL
    )
    
    # Initialize reference data manager
    reference_manager = ReferenceDataManager(
        filepath=config.REFERENCE_DATA_PATH
    )
    
    # Create agent
    agent = CO2ReductionAgent(
        llm_client=llm_client,
        vector_store=vector_store,
        reference_data_manager=reference_manager
    )
    
    return agent


def initialize_session_state():
    """Initialize session state variables for conversation history."""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""
    
    if 'last_response' not in st.session_state:
        st.session_state.last_response = None
    
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    
    if 'dataset_analysis' not in st.session_state:
        st.session_state.dataset_analysis = None


def render_query_interface(agent: CO2ReductionAgent):
    """
    Render the query input interface with text input and example queries.
    
    Args:
        agent: Initialized CO2ReductionAgent
    """
    st.subheader("🌍 Ask About Environmental Impact")
    st.markdown("Enter your question about reducing your environmental footprint - CO₂, water, energy, or waste.")
    
    # Example queries in an expander
    with st.expander("💡 Example Questions"):
        st.markdown("""
        **Carbon Emissions:**
        - I drive 20 km daily using a petrol car. How can I reduce emissions?
        - Compare emissions from driving vs taking the bus
        - What's the CO₂ impact of eating beef vs chicken?
        
        **Water Conservation:**
        - How can I reduce water usage at home?
        - What's the water footprint of different foods?
        - Compare water usage of shower vs bath
        
        **Energy Efficiency:**
        - How can I reduce my household energy consumption?
        - What are the best ways to save electricity?
        - Compare energy efficiency of LED vs incandescent lights
        
        **Waste Reduction:**
        - How can I reduce plastic waste?
        - What are the best recycling practices?
        - Compare environmental impact of reusable vs disposable items
        
        **Comprehensive:**
        - What are the best ways to reduce my overall environmental footprint?
        - Compare the total environmental impact of different lifestyle choices
        """)
    
    # Query input with form for better UX
    with st.form(key="query_form", clear_on_submit=False):
        user_query = st.text_area(
            "Your Question:",
            value=st.session_state.last_query,
            height=100,
            placeholder="e.g., I drive 30 km daily in a petrol car. How can I reduce my environmental impact?",
            help="Describe your activity or ask a question about environmental impact reduction"
        )
        
        submit_button = st.form_submit_button("🔍 Get Recommendations")
    
    # Process query when submitted
    if submit_button and user_query.strip():
        st.session_state.last_query = user_query
        
        with st.spinner("🤔 Analyzing your query and generating recommendations..."):
            try:
                response = agent.process_query(user_query)
                st.session_state.last_response = response
                
                # Add to conversation history
                st.session_state.conversation_history.append({
                    'query': user_query,
                    'response': response
                })
                
            except RuntimeError as e:
                # LLM-related errors
                if "Failed to generate response" in str(e):
                    show_error(
                        f"LLM service error: {str(e)}",
                        error_type="llm_unavailable",
                        show_retry=True
                    )
                else:
                    show_error(
                        f"Failed to process query: {str(e)}",
                        error_type="query_processing",
                        show_retry=True
                    )
                return
            except ValueError as e:
                show_error(
                    f"Invalid query: {str(e)}",
                    error_type="query_processing"
                )
                return
            except Exception as e:
                show_error(
                    f"Unexpected error: {str(e)}",
                    error_type="general",
                    show_retry=True
                )
                return
    
    # Display results if available
    if st.session_state.last_response:
        st.markdown("---")
        render_query_results(st.session_state.last_response)


def render_file_upload_interface(agent: CO2ReductionAgent):
    """
    Render the file upload interface for dataset analysis.
    
    Args:
        agent: Initialized CO2ReductionAgent
    """
    st.subheader("Upload Your Activity Dataset")
    st.markdown("Upload a CSV or Excel file with your daily activities for comprehensive analysis.")
    
    # Upload instructions
    with st.expander("📋 File Format Requirements"):
        st.markdown("""
        Your file must include the following columns:
        - **Activity**: Name of the activity (e.g., "Driving petrol car")
        - **Avg_CO2_Emission(kg/day)**: CO₂ emission in kg per day
        - **Category**: One of: Transport, Household, Food, Lifestyle
        
        **Example:**
        ```
        Activity,Avg_CO2_Emission(kg/day),Category
        Driving petrol car,4.6,Transport
        Eating beef,3.3,Food
        Using air conditioning,2.1,Household
        ```
        
        **Limits:**
        - Maximum file size: 10 MB
        - Supported formats: CSV, Excel (.xlsx, .xls)
        """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel file with your activity data",
        key="dataset_uploader"
    )
    
    if uploaded_file is not None:
        try:
            # Read the file based on type
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.uploaded_data = df
            
            # Show preview
            st.success(f"✅ File uploaded successfully: {uploaded_file.name}")
            st.markdown("**Preview (first 5 rows):**")
            st.dataframe(df.head())
            
            # Validate the dataset
            required_columns = ["Activity", "Avg_CO2_Emission(kg/day)", "Category"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                show_error(
                    f"Missing required columns: {', '.join(missing_columns)}",
                    error_type="file_upload"
                )
                return
            
            # Show dataset statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Activities", len(df))
            with col2:
                st.metric("Total Daily CO₂", f"{df['Avg_CO2_Emission(kg/day)'].sum():.2f} kg")
            with col3:
                st.metric("Categories", df['Category'].nunique())
            
            # Analyze button
            if st.button("🔍 Analyze Dataset", key="analyze_btn"):
                with st.spinner("📊 Analyzing your dataset and generating recommendations..."):
                    try:
                        analysis = agent.analyze_dataset(df)
                        st.session_state.dataset_analysis = analysis
                        st.success("✅ Analysis complete!")
                        
                    except ValueError as e:
                        show_error(
                            f"Dataset validation failed: {str(e)}",
                            error_type="validation"
                        )
                    except RuntimeError as e:
                        if "Failed to generate response" in str(e):
                            show_error(
                                f"LLM service error during analysis: {str(e)}",
                                error_type="llm_unavailable",
                                show_retry=True
                            )
                        else:
                            show_error(
                                f"Failed to analyze dataset: {str(e)}",
                                error_type="general",
                                show_retry=True
                            )
                    except Exception as e:
                        show_error(
                            f"Failed to analyze dataset: {str(e)}",
                            error_type="general",
                            show_retry=True
                        )
            
            # Display analysis results if available
            if st.session_state.dataset_analysis:
                st.markdown("---")
                render_dataset_analysis_results(st.session_state.dataset_analysis)
                
        except Exception as e:
            show_error(
                f"Failed to read file: {str(e)}",
                error_type="file_upload"
            )
    else:
        # Show placeholder when no file is uploaded
        st.info("👆 Upload a file to get started with dataset analysis")


def render_query_results(response: AgentResponse):
    """
    Display query results with comprehensive environmental metrics and visualizations.
    
    Args:
        response: AgentResponse object with recommendations
    """
    st.success("✅ Analysis Complete!")
    
    # Display summary
    st.markdown("### 📝 Summary")
    st.info(response.summary)
    
    # Environmental Score
    if response.environmental_score > 0:
        st.markdown("### 🎯 Environmental Impact Score")
        score = response.environmental_score
        score_color = "green" if score < 30 else "orange" if score < 60 else "red"
        st.progress(min(score / 100, 1.0))
        st.markdown(f"**Score: {score:.1f}/100** (Lower is better)")
    
    # Main metrics - CO2
    st.markdown("### 🌡️ Carbon Footprint")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Current Daily Emission",
            f"{response.current_emission:.2f} kg CO₂",
            help="Your current daily CO₂ emission from this activity"
        )
    
    with col2:
        st.metric(
            "Potential Daily Reduction",
            f"{response.total_potential_reduction:.2f} kg CO₂",
            delta=f"-{response.total_potential_reduction:.2f} kg",
            delta_color="inverse",
            help="Total potential reduction from all recommendations"
        )
    
    with col3:
        st.metric(
            "Annual CO₂ Savings",
            f"{response.annual_savings_kg:.1f} kg",
            delta=f"-{response.annual_savings_kg:.1f} kg/year",
            delta_color="inverse",
            help="Projected annual CO₂ savings"
        )
    
    # Additional Environmental Metrics
    st.markdown("### 💧🔌♻️ Other Environmental Impacts")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        water_usage = getattr(response, 'current_water_usage', 0)
        water_savings = getattr(response, 'total_water_savings', 0)
        st.metric(
            "💧 Water Usage",
            f"{water_usage:.1f} L/day",
            delta=f"-{water_savings:.1f} L" if water_savings > 0 else None,
            delta_color="inverse",
            help="Daily water consumption"
        )
    
    with col2:
        energy_usage = getattr(response, 'current_energy_usage', 0)
        energy_savings = getattr(response, 'total_energy_savings', 0)
        st.metric(
            "⚡ Energy Usage",
            f"{energy_usage:.2f} kWh/day",
            delta=f"-{energy_savings:.2f} kWh" if energy_savings > 0 else None,
            delta_color="inverse",
            help="Daily energy consumption"
        )
    
    with col3:
        waste_gen = getattr(response, 'current_waste_generation', 0)
        waste_reduction = getattr(response, 'total_waste_reduction', 0)
        st.metric(
            "🗑️ Waste Generation",
            f"{waste_gen:.3f} kg/day",
            delta=f"-{waste_reduction:.3f} kg" if waste_reduction > 0 else None,
            delta_color="inverse",
            help="Daily waste generation"
        )
    
    with col4:
        # Calculate annual totals for all metrics
        annual_water = water_usage * 365
        annual_energy = energy_usage * 365
        st.metric(
            "📅 Annual Water",
            f"{annual_water:.0f} L/year",
            help="Projected annual water usage"
        )
    
    # Display recommendations
    if response.recommendations:
        st.markdown("### 💡 Recommendations")
        
        for idx, rec in enumerate(response.recommendations, 1):
            with st.expander(f"**{idx}. {rec.action}**", expanded=(idx == 1)):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**🌡️ CO₂ Impact:**")
                    st.write(f"- Reduces emissions by **{rec.emission_reduction_kg:.2f} kg CO₂/day**")
                    st.write(f"- **{rec.reduction_percentage:.1f}%** reduction")
                    st.write(f"- Annual savings: **{rec.emission_reduction_kg * 365:.1f} kg CO₂**")
                    
                    # Additional environmental benefits
                    water_red = getattr(rec, 'water_reduction_liters', 0)
                    energy_red = getattr(rec, 'energy_reduction_kwh', 0)
                    waste_red = getattr(rec, 'waste_reduction_kg', 0)
                    cost_savings = getattr(rec, 'cost_savings_annual', 0)
                    
                    if water_red > 0 or energy_red > 0 or waste_red > 0:
                        st.markdown("**🌍 Additional Environmental Benefits:**")
                        if water_red > 0:
                            st.write(f"- 💧 Water savings: **{water_red:.1f} L/day**")
                        if energy_red > 0:
                            st.write(f"- ⚡ Energy savings: **{energy_red:.2f} kWh/day**")
                        if waste_red > 0:
                            st.write(f"- ♻️ Waste reduction: **{waste_red:.3f} kg/day**")
                    
                    if cost_savings > 0:
                        st.markdown(f"**💰 Cost Savings:** ~${cost_savings:.0f}/year")
                    
                    if rec.additional_benefits:
                        st.markdown("**✨ Other Benefits:**")
                        for benefit in rec.additional_benefits:
                            st.write(f"- {benefit}")
                    
                    health_benefits = getattr(rec, 'health_benefits', [])
                    if health_benefits:
                        st.markdown("**❤️ Health Benefits:**")
                        for benefit in health_benefits:
                            st.write(f"- {benefit}")
                
                with col2:
                    st.markdown(f"**Difficulty:** {rec.implementation_difficulty}")
                    st.markdown(f"**Timeframe:** {rec.timeframe}")
        
        # Multi-metric comparison charts
        if len(response.recommendations) > 0:
            st.markdown("### 📊 Impact Comparison")
            
            # Tabs for different metrics
            tab1, tab2, tab3 = st.tabs(["🌡️ CO₂ Emissions", "💧 Water Usage", "⚡ Energy"])
            
            with tab1:
                chart_data = {
                    "Option": ["Current"],
                    "CO₂ Emission (kg/day)": [response.current_emission]
                }
                
                for idx, rec in enumerate(response.recommendations[:5], 1):
                    alternative_emission = response.current_emission - rec.emission_reduction_kg
                    chart_data["Option"].append(f"Alt {idx}")
                    chart_data["CO₂ Emission (kg/day)"].append(max(0, alternative_emission))
                
                chart_df = pd.DataFrame(chart_data)
                st.bar_chart(chart_df.set_index("Option"))
            
            with tab2:
                water_usage = getattr(response, 'current_water_usage', 0)
                water_data = {
                    "Option": ["Current"],
                    "Water (L/day)": [water_usage]
                }
                
                for idx, rec in enumerate(response.recommendations[:5], 1):
                    water_red = getattr(rec, 'water_reduction_liters', 0)
                    water_data["Option"].append(f"Alt {idx}")
                    water_data["Water (L/day)"].append(max(0, water_usage - water_red))
                
                water_df = pd.DataFrame(water_data)
                st.bar_chart(water_df.set_index("Option"))
            
            with tab3:
                energy_usage = getattr(response, 'current_energy_usage', 0)
                energy_data = {
                    "Option": ["Current"],
                    "Energy (kWh/day)": [energy_usage]
                }
                
                for idx, rec in enumerate(response.recommendations[:5], 1):
                    energy_red = getattr(rec, 'energy_reduction_kwh', 0)
                    energy_data["Option"].append(f"Alt {idx}")
                    energy_data["Energy (kWh/day)"].append(max(0, energy_usage - energy_red))
                
                energy_df = pd.DataFrame(energy_data)
                st.bar_chart(energy_df.set_index("Option"))
    else:
        st.warning("No specific recommendations available for this query.")
    
    # Add feedback section
    render_feedback_interface(response)


def render_feedback_interface(response: AgentResponse):
    """
    Render user feedback collection interface.
    
    Args:
        response: AgentResponse to collect feedback on
    """
    st.markdown("---")
    st.markdown("### 💭 Was this helpful?")
    st.markdown("Your feedback helps us improve recommendations.")
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    
    with col1:
        if st.button("👍 Helpful", key="helpful_btn"):
            collector = FeedbackCollector()
            collector.save_feedback(
                query=st.session_state.get('last_query', ''),
                response=response.summary,
                is_helpful=True,
                rating=5
            )
            st.success("Thanks for your feedback!")
    
    with col2:
        if st.button("👎 Not Helpful", key="not_helpful_btn"):
            st.session_state['show_feedback_form'] = True
    
    with col3:
        if st.button("⚠️ Inaccurate", key="inaccurate_btn"):
            st.session_state['show_inaccuracy_form'] = True
    
    # Show detailed feedback form if requested
    if st.session_state.get('show_feedback_form', False):
        with st.form("feedback_form"):
            st.markdown("**What could be improved?**")
            feedback_text = st.text_area(
                "Your feedback (optional)",
                placeholder="Tell us what was missing or could be better..."
            )
            rating = st.slider("Rate this recommendation", 1, 5, 3)
            
            col_submit, col_cancel = st.columns(2)
            with col_submit:
                if st.form_submit_button("Submit Feedback"):
                    collector = FeedbackCollector()
                    collector.save_feedback(
                        query=st.session_state.get('last_query', ''),
                        response=response.summary,
                        is_helpful=False,
                        rating=rating,
                        feedback_text=feedback_text
                    )
                    st.success("Thank you for your detailed feedback!")
                    st.session_state['show_feedback_form'] = False
                    st.rerun()
            
            with col_cancel:
                if st.form_submit_button("Cancel"):
                    st.session_state['show_feedback_form'] = False
                    st.rerun()
    
    # Show inaccuracy report form
    if st.session_state.get('show_inaccuracy_form', False):
        with st.form("inaccuracy_form"):
            st.markdown("**Report Inaccuracy**")
            st.markdown("Please describe what was inaccurate:")
            issue_description = st.text_area(
                "Issue description",
                placeholder="E.g., 'The emission value for electric cars seems too high'"
            )
            
            col_submit, col_cancel = st.columns(2)
            with col_submit:
                if st.form_submit_button("Submit Report"):
                    collector = FeedbackCollector()
                    collector.log_inaccuracy_report(
                        query=st.session_state.get('last_query', ''),
                        response=response.summary,
                        issue_description=issue_description
                    )
                    st.warning("Thank you for reporting. We'll review this recommendation.")
                    st.session_state['show_inaccuracy_form'] = False
                    st.rerun()
            
            with col_cancel:
                if st.form_submit_button("Cancel"):
                    st.session_state['show_inaccuracy_form'] = False
                    st.rerun()


def render_dataset_analysis_results(analysis: DatasetAnalysis):
    """
    Display dataset analysis results with visualizations.
    
    Args:
        analysis: DatasetAnalysis object with complete analysis
    """
    st.success("✅ Dataset Analysis Complete!")
    
    # Overall emissions
    st.markdown("### 📊 Overall Emissions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Total Daily Emission",
            f"{analysis.total_daily_emission:.2f} kg CO₂",
            help="Sum of all daily CO₂ emissions"
        )
    
    with col2:
        st.metric(
            "Total Annual Emission",
            f"{analysis.total_annual_emission:.1f} kg CO₂",
            help="Projected annual CO₂ emissions (daily × 365)"
        )
    
    # Top emitters
    if analysis.top_emitters:
        st.markdown("### 🔥 Top Emission Activities")
        
        for idx, activity in enumerate(analysis.top_emitters, 1):
            percentage = (activity.emission_kg_per_day / analysis.total_daily_emission) * 100
            st.write(
                f"{idx}. **{activity.name}** ({activity.category.value}): "
                f"{activity.emission_kg_per_day:.2f} kg CO₂/day "
                f"({percentage:.1f}% of total)"
            )
    
    # Category breakdown
    if analysis.category_breakdown:
        st.markdown("### 📈 Emissions by Category")
        
        category_df = pd.DataFrame({
            "Category": list(analysis.category_breakdown.keys()),
            "CO₂ Emission (kg/day)": list(analysis.category_breakdown.values())
        })
        
        st.bar_chart(category_df.set_index("Category"))
    
    # Recommendations
    if analysis.recommendations:
        st.markdown("### 💡 Priority Recommendations")
        st.markdown("Based on your top emission activities, here are our recommendations:")
        
        total_potential_reduction = sum(rec.emission_reduction_kg for rec in analysis.recommendations)
        annual_potential_savings = total_potential_reduction * 365
        
        # Show potential savings
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Potential Daily Reduction",
                f"{total_potential_reduction:.2f} kg CO₂",
                delta=f"-{total_potential_reduction:.2f} kg",
                delta_color="inverse"
            )
        with col2:
            st.metric(
                "Potential Annual Savings",
                f"{annual_potential_savings:.1f} kg CO₂",
                delta=f"-{annual_potential_savings:.1f} kg/year",
                delta_color="inverse"
            )
        
        # Display recommendations
        for idx, rec in enumerate(analysis.recommendations[:10], 1):
            with st.expander(f"**{idx}. {rec.action}**", expanded=(idx <= 3)):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Impact:**")
                    st.write(f"- Reduces emissions by **{rec.emission_reduction_kg:.2f} kg CO₂/day**")
                    st.write(f"- **{rec.reduction_percentage:.1f}%** reduction")
                    st.write(f"- Annual savings: **{rec.emission_reduction_kg * 365:.1f} kg CO₂**")
                    
                    if rec.additional_benefits:
                        st.markdown("**Additional Benefits:**")
                        for benefit in rec.additional_benefits:
                            st.write(f"- {benefit}")
                
                with col2:
                    st.markdown(f"**Difficulty:** {rec.implementation_difficulty}")
                    st.markdown(f"**Timeframe:** {rec.timeframe}")


def show_error(message: str, error_type: str = "general", show_retry: bool = False):
    """
    Display user-friendly error messages with context-specific guidance.
    
    Args:
        message: Error message to display
        error_type: Type of error for context-specific guidance
        show_retry: Whether to show a retry button for transient errors
    """
    st.error(f"❌ {message}")
    
    # Provide context-specific guidance
    if error_type == "llm_unavailable":
        st.info("""
        **Troubleshooting Steps:**
        1. Ensure Ollama is running: `ollama serve`
        2. Check if the model is available: `ollama list`
        3. Pull the model if needed: `ollama pull llama3`
        4. Verify the model name in config.py matches your installed model
        """)
        
        if show_retry:
            if st.button("🔄 Retry Connection", key="retry_llm"):
                st.rerun()
    
    elif error_type == "file_upload":
        st.info("""
        **File Requirements:**
        - Format: CSV or Excel (.xlsx, .xls)
        - Required columns: Activity, Avg_CO2_Emission(kg/day), Category
        - Maximum size: 10 MB
        
        **Common Issues:**
        - Check column names match exactly (case-sensitive)
        - Ensure emission values are numeric
        - Verify category values are: Transport, Household, Food, or Lifestyle
        """)
    
    elif error_type == "query_processing":
        st.info("""
        **Tips:**
        - Try rephrasing your question
        - Be specific about activities and quantities
        - Check the example questions for guidance
        - Ensure your query mentions a recognizable activity
        """)
        
        if show_retry:
            if st.button("🔄 Try Again", key="retry_query"):
                st.session_state.last_response = None
                st.rerun()
    
    elif error_type == "validation":
        st.info("""
        **Data Validation Failed:**
        - Check that all emission values are positive numbers
        - Verify all required columns are present
        - Ensure there are no empty rows
        - Remove any special characters from activity names
        """)
    
    elif error_type == "vector_store":
        st.info("""
        **Vector Store Issue:**
        - The sustainability knowledge base may not be initialized
        - Try running: `python scripts/init_vector_store.py`
        - Check that the chroma_db directory exists and is accessible
        """)
        
        if show_retry:
            if st.button("🔄 Retry", key="retry_vector"):
                st.rerun()
    
    elif error_type == "general":
        st.info("""
        **General Error:**
        - Check your internet connection
        - Verify all required services are running
        - Review the system status in the sidebar
        - Try refreshing the page
        """)
        
        if show_retry:
            if st.button("🔄 Retry", key="retry_general"):
                st.rerun()


def show_warning(message: str, details: Optional[str] = None):
    """
    Display warning messages with optional details.
    
    Args:
        message: Warning message to display
        details: Optional additional details
    """
    st.warning(f"⚠️ {message}")
    
    if details:
        with st.expander("View Details"):
            st.write(details)


def show_validation_errors(errors: List[str]):
    """
    Display validation errors with specific guidance.
    
    Args:
        errors: List of validation error messages
    """
    st.error("❌ Data Validation Failed")
    
    st.markdown("**Issues Found:**")
    for idx, error in enumerate(errors, 1):
        st.write(f"{idx}. {error}")
    
    st.info("""
    **How to Fix:**
    - Review the file format requirements
    - Correct the issues listed above
    - Re-upload your file
    """)


def render_financial_calculator():
    """Render the Financial Impact Calculator interface."""
    st.subheader("💰 Financial Impact Calculator")
    st.markdown("Calculate cost savings, ROI, and financial benefits of eco-friendly choices.")
    
    # Initialize calculator
    calculator = FinancialCalculator()
    
    # Initialize session state for financial calculator
    if 'custom_rates' not in st.session_state:
        st.session_state.custom_rates = {}
    
    # Create sub-tabs for different calculators
    fin_tab1, fin_tab2, fin_tab3, fin_tab4 = st.tabs([
        "💵 Cost Savings",
        "📈 Green Investment ROI",
        "🏠 Utility Comparison",
        "🌱 Carbon Credits"
    ])
    
    with fin_tab1:
        render_cost_savings_calculator(calculator)
    
    with fin_tab2:
        render_roi_calculator(calculator)
    
    with fin_tab3:
        render_utility_comparison(calculator)
    
    with fin_tab4:
        render_carbon_credit_calculator(calculator)


def render_cost_savings_calculator(calculator: FinancialCalculator):
    """Render the cost savings calculation interface."""
    st.markdown("### Calculate Savings from Lifestyle Changes")
    
    calc_type = st.selectbox(
        "What type of savings would you like to calculate?",
        [
            "Transport Savings", 
            "Energy Savings", 
            "Water Savings",
            "Food & Groceries",
            "Heating & Cooling",
            "Appliance Upgrades",
            "Subscription Services"
        ],
        key="savings_type"
    )
    
    if calc_type == "Transport Savings":
        st.markdown("#### 🚗 Transport Cost Savings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            daily_km = st.number_input(
                "Daily distance (km)",
                min_value=0.0,
                value=20.0,
                step=5.0,
                key="transport_km"
            )
            
            current_mode = st.selectbox(
                "Current transport mode",
                ["petrol_car", "diesel_car", "motorcycle", "public_transit", "hybrid_car"],
                format_func=lambda x: x.replace("_", " ").title(),
                key="current_transport"
            )
        
        with col2:
            alternative_mode = st.selectbox(
                "Alternative mode",
                ["electric_car", "hybrid_car", "public_transit", "cycling", "ebike", "carpool", "walking"],
                format_func=lambda x: x.replace("_", " ").title(),
                key="alt_transport"
            )
        
        if st.button("Calculate Transport Savings", key="calc_transport"):
            savings = calculator.calculate_transport_cost_savings(
                daily_km=daily_km,
                current_mode=current_mode,
                alternative_mode=alternative_mode
            )
            display_cost_savings(savings)
    
    elif calc_type == "Energy Savings":
        st.markdown("#### ⚡ Energy Cost Savings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            current_kwh = st.number_input(
                "Current daily energy use (kWh)",
                min_value=0.0,
                value=25.0,
                step=1.0,
                key="current_energy"
            )
            
            electricity_rate = st.number_input(
                "Electricity rate ($/kWh)",
                min_value=0.01,
                value=0.15,
                step=0.01,
                key="elec_rate"
            )
        
        with col2:
            reduction_percent = st.slider(
                "Expected reduction (%)",
                min_value=5,
                max_value=50,
                value=20,
                key="energy_reduction"
            )
            
            reduced_kwh = current_kwh * (1 - reduction_percent / 100)
            st.metric("Reduced usage", f"{reduced_kwh:.1f} kWh/day")
        
        if st.button("Calculate Energy Savings", key="calc_energy"):
            savings = calculator.calculate_energy_cost_savings(
                current_kwh_daily=current_kwh,
                reduced_kwh_daily=reduced_kwh,
                electricity_rate=electricity_rate
            )
            display_cost_savings(savings)
    
    elif calc_type == "Water Savings":
        st.markdown("#### 💧 Water Cost Savings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            current_liters = st.number_input(
                "Current daily water use (liters)",
                min_value=0.0,
                value=150.0,
                step=10.0,
                key="current_water"
            )
            
            water_rate = st.number_input(
                "Water rate ($/liter)",
                min_value=0.0001,
                value=0.0013,
                step=0.0001,
                format="%.4f",
                key="water_rate"
            )
        
        with col2:
            reduction_percent = st.slider(
                "Expected reduction (%)",
                min_value=5,
                max_value=50,
                value=25,
                key="water_reduction"
            )
            
            reduced_liters = current_liters * (1 - reduction_percent / 100)
            st.metric("Reduced usage", f"{reduced_liters:.0f} L/day")
        
        if st.button("Calculate Water Savings", key="calc_water"):
            savings = calculator.calculate_water_cost_savings(
                current_liters_daily=current_liters,
                reduced_liters_daily=reduced_liters,
                water_rate=water_rate
            )
            display_cost_savings(savings)
    
    elif calc_type == "Food & Groceries":
        st.markdown("#### 🍽️ Food & Grocery Cost Savings")
        st.markdown("Calculate savings from changing your meal habits.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            meals_per_week = st.number_input(
                "Meals per week to change",
                min_value=1,
                max_value=21,
                value=7,
                step=1,
                key="meals_week"
            )
            
            current_option = st.selectbox(
                "Current meal source",
                ["restaurant", "fast_food", "takeout", "meal_kit", "cafeteria"],
                format_func=lambda x: x.replace("_", " ").title(),
                key="current_meal"
            )
        
        with col2:
            alternative_option = st.selectbox(
                "Alternative meal source",
                ["home_cooked", "meal_prep", "plant_based_home", "meal_kit", "cafeteria"],
                format_func=lambda x: x.replace("_", " ").title(),
                key="alt_meal"
            )
            
            meal_costs_display = {
                "restaurant": "$18/meal",
                "fast_food": "$10/meal", 
                "takeout": "$15/meal",
                "meal_kit": "$12/meal",
                "home_cooked": "$5/meal",
                "meal_prep": "$4/meal",
                "plant_based_home": "$3.50/meal",
                "cafeteria": "$8/meal"
            }
            st.info(f"💡 {current_option.replace('_', ' ').title()}: {meal_costs_display.get(current_option, 'N/A')}")
        
        if st.button("Calculate Food Savings", key="calc_food"):
            savings = calculator.calculate_food_cost_savings(
                meals_per_week=meals_per_week,
                current_option=current_option,
                alternative_option=alternative_option
            )
            display_cost_savings(savings)
    
    elif calc_type == "Appliance Upgrades":
        st.markdown("#### 🔌 Appliance Upgrade Savings")
        st.markdown("Calculate savings from upgrading to energy-efficient appliances.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            appliance_type = st.selectbox(
                "Appliance type",
                ["refrigerator", "washing_machine", "dryer", "dishwasher", 
                 "air_conditioner", "water_heater", "television", "computer"],
                format_func=lambda x: x.replace("_", " ").title(),
                key="appliance_type"
            )
            
            current_age = st.number_input(
                "Current appliance age (years)",
                min_value=1,
                max_value=30,
                value=15,
                step=1,
                key="appliance_age"
            )
        
        with col2:
            usage_hours = st.number_input(
                "Daily usage (hours)",
                min_value=0.5,
                max_value=24.0,
                value=8.0,
                step=0.5,
                key="usage_hours"
            )
            
            # Show estimated energy use
            energy_multiplier = 1 + (current_age * 0.02)
            st.warning(f"⚠️ {current_age}-year-old appliances use ~{int(energy_multiplier*100-100)}% more energy due to wear")
        
        if st.button("Calculate Appliance Savings", key="calc_appliance"):
            savings = calculator.calculate_appliance_savings(
                appliance_type=appliance_type,
                current_age_years=current_age,
                usage_hours_daily=usage_hours
            )
            display_cost_savings(savings)
    
    elif calc_type == "Heating & Cooling":
        st.markdown("#### 🌡️ Heating & Cooling Savings")
        st.markdown("Calculate savings from HVAC system upgrades.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            current_system = st.selectbox(
                "Current system",
                ["gas_furnace", "oil_furnace", "electric_resistance", 
                 "window_ac", "central_ac"],
                format_func=lambda x: x.replace("_", " ").title(),
                key="current_hvac"
            )
            
            monthly_heating = st.number_input(
                "Monthly heating cost ($)",
                min_value=0.0,
                value=150.0,
                step=10.0,
                key="monthly_heat"
            )
        
        with col2:
            alternative_system = st.selectbox(
                "New system option",
                ["heat_pump", "geothermal", "mini_split", "evaporative_cooler"],
                format_func=lambda x: x.replace("_", " ").title(),
                key="alt_hvac"
            )
            
            monthly_cooling = st.number_input(
                "Monthly cooling cost ($)",
                min_value=0.0,
                value=100.0,
                step=10.0,
                key="monthly_cool"
            )
        
        # Show efficiency comparison
        efficiency_info = {
            "gas_furnace": "92-98% efficient",
            "oil_furnace": "80-90% efficient",
            "electric_resistance": "100% efficient (but expensive)",
            "heat_pump": "200-300% efficient (COP 2-3)",
            "geothermal": "300-500% efficient (COP 3-5)",
            "mini_split": "200-400% efficient",
            "evaporative_cooler": "Very efficient in dry climates"
        }
        st.info(f"💡 {alternative_system.replace('_', ' ').title()}: {efficiency_info.get(alternative_system, 'N/A')}")
        
        if st.button("Calculate HVAC Savings", key="calc_hvac"):
            savings = calculator.calculate_heating_cooling_savings(
                current_system=current_system,
                alternative_system=alternative_system,
                monthly_heating_cost=monthly_heating,
                monthly_cooling_cost=monthly_cooling
            )
            display_cost_savings(savings)
    
    elif calc_type == "Subscription Services":
        st.markdown("#### 📱 Subscription Service Savings")
        st.markdown("Calculate savings from reducing digital subscriptions (also saves energy from reduced data usage).")
        
        col1, col2 = st.columns(2)
        
        with col1:
            service_type = st.selectbox(
                "Subscription type",
                ["streaming", "gaming", "music", "news", "fitness", "cloud_storage", "software", "delivery"],
                format_func=lambda x: x.replace("_", " ").title(),
                key="service_type"
            )
            
            current_subs = st.number_input(
                "Current number of subscriptions",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                key="current_subs"
            )
        
        with col2:
            reduced_subs = st.number_input(
                "Subscriptions to keep",
                min_value=0,
                max_value=current_subs,
                value=min(2, current_subs),
                step=1,
                key="reduced_subs"
            )
            
            avg_costs = {
                "streaming": 15, "gaming": 12, "music": 10, "news": 15,
                "fitness": 30, "cloud_storage": 10, "software": 20, "delivery": 15
            }
            st.info(f"💡 Average {service_type} cost: ${avg_costs.get(service_type, 15)}/month each")
        
        if st.button("Calculate Subscription Savings", key="calc_subs"):
            savings = calculator.calculate_subscription_savings(
                service_type=service_type,
                current_subscriptions=current_subs,
                reduced_subscriptions=reduced_subs
            )
            display_cost_savings(savings)


def display_cost_savings(savings: CostSavings):
    """Display cost savings results."""
    st.success("💰 Potential Savings Calculated!")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Daily Savings",
            f"${savings.daily_savings:.2f}",
            delta=f"+${savings.daily_savings:.2f}"
        )
    
    with col2:
        st.metric(
            "Monthly Savings",
            f"${savings.monthly_savings:.2f}",
            delta=f"+${savings.monthly_savings:.2f}"
        )
    
    with col3:
        st.metric(
            "Annual Savings",
            f"${savings.annual_savings:.2f}",
            delta=f"+${savings.annual_savings:.2f}"
        )
    
    with col4:
        st.metric(
            "10-Year Savings",
            f"${savings.lifetime_savings:.2f}",
            delta=f"+${savings.lifetime_savings:.2f}"
        )
    
    st.info(f"📝 {savings.description}")


def render_roi_calculator(calculator: FinancialCalculator):
    """Render the green investment ROI calculator."""
    st.markdown("### 📈 Green Investment ROI Calculator")
    st.markdown("Analyze the financial returns on eco-friendly investments.")
    
    # Get all investment options
    options = calculator.get_all_investment_options()
    
    # Define investment categories
    investment_categories = {
        "⚡ Energy & Power": [
            InvestmentType.SOLAR_PANELS, InvestmentType.BATTERY_STORAGE, 
            InvestmentType.SOLAR_WATER_HEATER, InvestmentType.SMART_POWER_STRIPS,
            InvestmentType.ENERGY_MONITOR
        ],
        "🏠 Home Improvement": [
            InvestmentType.INSULATION, InvestmentType.DOUBLE_GLAZED_WINDOWS,
            InvestmentType.GREEN_ROOF, InvestmentType.HEAT_PUMP
        ],
        "🚗 Transportation": [
            InvestmentType.ELECTRIC_VEHICLE, InvestmentType.ELECTRIC_BIKE, 
            InvestmentType.ELECTRIC_SCOOTER, InvestmentType.EV_CHARGER_HOME
        ],
        "💡 Efficiency Upgrades": [
            InvestmentType.LED_LIGHTING, InvestmentType.SMART_THERMOSTAT,
            InvestmentType.ENERGY_EFFICIENT_APPLIANCES, InvestmentType.EFFICIENT_WATER_HEATER
        ],
        "💧 Water Conservation": [
            InvestmentType.RAINWATER_HARVESTING, InvestmentType.SMART_IRRIGATION
        ],
        "♻️ Waste Management": [
            InvestmentType.COMPOSTING_SYSTEM
        ]
    }
    
    # Category selection
    selected_category = st.selectbox(
        "Select Investment Category",
        list(investment_categories.keys()),
        key="investment_category"
    )
    
    # Get investments in selected category
    category_investments = investment_categories[selected_category]
    category_options = [opt for opt in options if InvestmentType(opt["type"]) in category_investments]
    
    # Investment selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_investment = st.selectbox(
            "Select Investment Type",
            [opt["type"] for opt in category_options],
            format_func=lambda x: next((opt["name"] for opt in category_options if opt["type"] == x), x),
            key="investment_type"
        )
    
    with col2:
        use_custom_values = st.checkbox("Use custom values", key="custom_values")
    
    # Get selected investment data
    investment_type = InvestmentType(selected_investment)
    investment_data = calculator.INVESTMENT_DATA[investment_type]
    
    # Show investment details
    with st.expander("📋 Investment Details", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Cost", f"${investment_data['avg_cost']:,}")
            st.metric("Typical Lifetime", f"{investment_data['lifetime_years']} years")
        with col2:
            st.metric("Annual Savings", f"${investment_data['avg_annual_savings']:,}/year")
            co2_savings = investment_data.get('co2_savings_kg_year', 0)
            st.metric("CO₂ Reduction", f"{co2_savings:,} kg/year")
        with col3:
            payback = investment_data['avg_cost'] / investment_data['avg_annual_savings']
            st.metric("Simple Payback", f"{payback:.1f} years")
            water_savings = investment_data.get('water_savings_liters_year', 0)
            if water_savings > 0:
                st.metric("Water Savings", f"{water_savings:,} L/year")
    
    # Custom inputs if enabled
    custom_cost = None
    custom_savings = None
    
    if use_custom_values:
        st.markdown("#### 🔧 Custom Values")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            custom_cost = st.number_input(
                "Initial Cost ($)",
                min_value=0.0,
                value=float(investment_data["avg_cost"]),
                step=100.0,
                key="custom_cost"
            )
        
        with col2:
            custom_savings = st.number_input(
                "Annual Savings ($)",
                min_value=0.0,
                value=float(investment_data["avg_annual_savings"]),
                step=50.0,
                key="custom_savings"
            )
        
        with col3:
            discount_rate = st.number_input(
                "Discount Rate (%)",
                min_value=0.0,
                max_value=15.0,
                value=5.0,
                step=0.5,
                key="discount_rate"
            ) / 100
    else:
        discount_rate = 0.05
    
    if st.button("Calculate ROI", key="calc_roi"):
        roi = calculator.calculate_investment_roi(
            investment_type=investment_type,
            custom_cost=custom_cost,
            custom_annual_savings=custom_savings,
            discount_rate=discount_rate
        )
        display_roi_results(roi, investment_data)
    
    # Show comparison table for selected category
    st.markdown(f"### 📊 {selected_category} Comparison")
    
    comparison_data = {
        "Investment": [],
        "Initial Cost": [],
        "Annual Savings": [],
        "Payback (Years)": [],
        "Lifetime ROI (%)": [],
        "CO₂ Saved (kg/year)": []
    }
    
    for opt in category_options:
        comparison_data["Investment"].append(opt["name"])
        comparison_data["Initial Cost"].append(f"${opt['avg_cost']:,.0f}")
        comparison_data["Annual Savings"].append(f"${opt['annual_savings']:,.0f}")
        comparison_data["Payback (Years)"].append(f"{opt['payback_years']:.1f}")
        comparison_data["Lifetime ROI (%)"].append(f"{opt['roi_percent']:.0f}%")
        comparison_data["CO₂ Saved (kg/year)"].append(f"{opt['co2_savings_annual']:,}")
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df)
    
    # Quick recommendations
    st.markdown("### 💡 Quick Recommendations")
    all_options = sorted(options, key=lambda x: x["payback_years"])
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🏆 Fastest Payback:**")
        for i, opt in enumerate(all_options[:3], 1):
            st.write(f"{i}. {opt['name']} ({opt['payback_years']:.1f} years)")
    
    with col2:
        highest_roi = sorted(options, key=lambda x: x["roi_percent"], reverse=True)
        st.markdown("**📈 Highest ROI:**")
        for i, opt in enumerate(highest_roi[:3], 1):
            st.write(f"{i}. {opt['name']} ({opt['roi_percent']:.0f}%)")


def display_roi_results(roi: ROIResult, investment_data: dict):
    """Display ROI calculation results."""
    st.success("📈 ROI Analysis Complete!")
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Initial Cost",
            f"${roi.initial_cost:,.0f}",
            help="Net cost after incentives"
        )
    
    with col2:
        st.metric(
            "Annual Savings",
            f"${roi.annual_savings:,.0f}",
            delta=f"+${roi.annual_savings:,.0f}/year"
        )
    
    with col3:
        color = "🟢" if roi.payback_years <= 5 else "🟡" if roi.payback_years <= 10 else "🔴"
        st.metric(
            f"{color} Payback Period",
            f"{roi.payback_years:.1f} years",
            help="Time to recover initial investment"
        )
    
    with col4:
        st.metric(
            "Total ROI",
            f"{roi.total_roi_percent:.0f}%",
            delta=f"+{roi.total_roi_percent:.0f}%"
        )
    
    # Financial details
    st.markdown("#### 💵 Financial Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Net Present Value (NPV)",
            f"${roi.net_present_value:,.0f}",
            help="Present value of future savings minus cost"
        )
    
    with col2:
        st.metric(
            "Internal Rate of Return",
            f"{roi.internal_rate_of_return:.1f}%",
            help="Effective annual return rate"
        )
    
    with col3:
        st.metric(
            "Lifetime Savings",
            f"${roi.total_lifetime_savings:,.0f}",
            help=f"Total savings over {roi.lifetime_years} years"
        )
    
    # Environmental impact
    st.markdown("#### 🌍 Environmental Impact")
    
    col1, col2, col3 = st.columns(3)
    
    env = roi.environmental_savings
    
    with col1:
        st.metric(
            "Annual CO₂ Reduction",
            f"{env.get('co2_kg_annual', 0):,} kg",
            delta=f"-{env.get('co2_kg_annual', 0):,} kg CO₂/year",
            delta_color="inverse"
        )
    
    with col2:
        lifetime_co2 = env.get('co2_kg_lifetime', 0)
        trees_equivalent = int(lifetime_co2 / 21)  # ~21 kg CO2 per tree per year
        st.metric(
            "Lifetime CO₂ Reduction",
            f"{lifetime_co2:,} kg",
            help=f"Equivalent to planting {trees_equivalent} trees"
        )
    
    with col3:
        if env.get('water_liters_lifetime', 0) > 0:
            st.metric(
                "Lifetime Water Savings",
                f"{env.get('water_liters_lifetime', 0):,} L"
            )
        else:
            st.metric(
                "Investment Lifetime",
                f"{roi.lifetime_years} years"
            )


def render_utility_comparison(calculator: FinancialCalculator):
    """Render utility cost comparison interface."""
    st.markdown("### 🏠 Utility Cost Comparison")
    st.markdown("Compare your current utility costs with optimized usage or alternative systems.")
    
    # Comparison type selection
    comparison_type = st.selectbox(
        "Select Comparison Type",
        [
            "📊 General Utility Comparison",
            "⚡ Electricity Sources",
            "🔥 Heating Systems",
            "❄️ Cooling Systems",
            "💧 Water Systems",
            "🚗 Transportation Fuel",
            "🏡 Home Energy Audit"
        ],
        key="comparison_type"
    )
    
    if comparison_type == "📊 General Utility Comparison":
        st.markdown("#### Compare Current vs Optimized Usage")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Current Usage (Daily)**")
            
            current_elec = st.number_input(
                "Electricity (kWh/day)",
                min_value=0.0,
                value=30.0,
                step=1.0,
                key="curr_elec"
            )
            
            current_gas = st.number_input(
                "Natural Gas (therms/day)",
                min_value=0.0,
                value=2.0,
                step=0.1,
                key="curr_gas"
            )
            
            current_water = st.number_input(
                "Water (liters/day)",
                min_value=0.0,
                value=300.0,
                step=10.0,
                key="curr_water_util"
            )
        
        with col2:
            st.markdown("**Optimized Usage (Daily)**")
            
            opt_elec = st.number_input(
                "Electricity (kWh/day)",
                min_value=0.0,
                value=22.0,
                step=1.0,
                key="opt_elec"
            )
            
            opt_gas = st.number_input(
                "Natural Gas (therms/day)",
                min_value=0.0,
                value=1.5,
                step=0.1,
                key="opt_gas"
            )
            
            opt_water = st.number_input(
                "Water (liters/day)",
                min_value=0.0,
                value=200.0,
                step=10.0,
                key="opt_water_util"
            )
        
        if st.button("Compare Utility Costs", key="compare_utility"):
            current = {
                "electricity_kwh": current_elec,
                "gas_therms": current_gas,
                "water_liters": current_water
            }
            
            optimized = {
                "electricity_kwh": opt_elec,
                "gas_therms": opt_gas,
                "water_liters": opt_water
            }
            
            comparison = calculator.compare_utility_costs(current, optimized)
            display_utility_comparison(comparison)
    
    elif comparison_type == "⚡ Electricity Sources":
        st.markdown("#### Compare Electricity Sources")
        
        col1, col2 = st.columns(2)
        
        with col1:
            monthly_kwh = st.number_input(
                "Monthly electricity usage (kWh)",
                min_value=0.0,
                value=900.0,
                step=50.0,
                key="monthly_kwh"
            )
        
        with col2:
            grid_rate = st.number_input(
                "Current grid rate ($/kWh)",
                min_value=0.0,
                value=0.14,
                step=0.01,
                format="%.2f",
                key="grid_rate"
            )
        
        if st.button("Compare Sources", key="compare_elec"):
            sources = {
                "Grid Electricity": {"rate": grid_rate, "co2_per_kwh": 0.42},
                "Solar Panels": {"rate": 0.06, "co2_per_kwh": 0.04},
                "Wind Power": {"rate": 0.08, "co2_per_kwh": 0.01},
                "Green Energy Plan": {"rate": grid_rate + 0.02, "co2_per_kwh": 0.05},
                "Solar + Battery": {"rate": 0.04, "co2_per_kwh": 0.03}
            }
            
            st.success("📊 Electricity Source Comparison")
            
            comparison_data = {
                "Source": [],
                "Monthly Cost": [],
                "Annual Cost": [],
                "CO₂/month (kg)": [],
                "Annual Savings vs Grid": []
            }
            
            grid_annual = monthly_kwh * 12 * grid_rate
            
            for source, data in sources.items():
                monthly_cost = monthly_kwh * data["rate"]
                annual_cost = monthly_cost * 12
                co2_monthly = monthly_kwh * data["co2_per_kwh"]
                savings = grid_annual - annual_cost
                
                comparison_data["Source"].append(source)
                comparison_data["Monthly Cost"].append(f"${monthly_cost:.2f}")
                comparison_data["Annual Cost"].append(f"${annual_cost:.2f}")
                comparison_data["CO₂/month (kg)"].append(f"{co2_monthly:.1f}")
                comparison_data["Annual Savings vs Grid"].append(f"${savings:.2f}" if savings > 0 else "-")
            
            df = pd.DataFrame(comparison_data)
            st.dataframe(df)
    
    elif comparison_type == "🔥 Heating Systems":
        st.markdown("#### Compare Heating Systems")
        
        col1, col2 = st.columns(2)
        
        with col1:
            heating_sqft = st.number_input(
                "Home size (sq ft)",
                min_value=0,
                value=2000,
                step=100,
                key="heating_sqft"
            )
            
            heating_months = st.slider(
                "Heating months per year",
                min_value=1,
                max_value=12,
                value=5,
                key="heating_months"
            )
        
        with col2:
            climate = st.selectbox(
                "Climate Zone",
                ["Mild", "Moderate", "Cold", "Very Cold"],
                index=1,
                key="climate_zone"
            )
        
        if st.button("Compare Heating Systems", key="compare_heating"):
            # BTU needs based on climate
            btu_per_sqft = {"Mild": 25, "Moderate": 35, "Cold": 45, "Very Cold": 55}
            annual_btu = heating_sqft * btu_per_sqft[climate] * heating_months * 30 * 8  # 8 hours/day
            
            systems = {
                "Natural Gas Furnace": {"efficiency": 0.92, "fuel_cost_per_btu": 0.000012, "co2_per_btu": 0.000053},
                "Electric Baseboard": {"efficiency": 1.0, "fuel_cost_per_btu": 0.000041, "co2_per_btu": 0.000123},
                "Heat Pump": {"efficiency": 3.0, "fuel_cost_per_btu": 0.000014, "co2_per_btu": 0.000041},
                "Geothermal Heat Pump": {"efficiency": 4.5, "fuel_cost_per_btu": 0.000009, "co2_per_btu": 0.000027},
                "Propane Furnace": {"efficiency": 0.85, "fuel_cost_per_btu": 0.000025, "co2_per_btu": 0.000063},
                "Wood Pellet Stove": {"efficiency": 0.80, "fuel_cost_per_btu": 0.000010, "co2_per_btu": 0.000010}
            }
            
            st.success("📊 Heating System Comparison")
            
            comparison_data = {
                "System": [],
                "Efficiency": [],
                "Annual Cost": [],
                "CO₂/year (kg)": [],
                "10-Year Cost": []
            }
            
            for system, data in systems.items():
                effective_btu = annual_btu / data["efficiency"]
                annual_cost = effective_btu * data["fuel_cost_per_btu"]
                annual_co2 = effective_btu * data["co2_per_btu"]
                
                comparison_data["System"].append(system)
                comparison_data["Efficiency"].append(f"{data['efficiency'] * 100:.0f}%" if data["efficiency"] <= 1 else f"{data['efficiency']:.1f} COP")
                comparison_data["Annual Cost"].append(f"${annual_cost:.0f}")
                comparison_data["CO₂/year (kg)"].append(f"{annual_co2:.0f}")
                comparison_data["10-Year Cost"].append(f"${annual_cost * 10:.0f}")
            
            df = pd.DataFrame(comparison_data)
            st.dataframe(df)
    
    elif comparison_type == "❄️ Cooling Systems":
        st.markdown("#### Compare Cooling Systems")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cooling_sqft = st.number_input(
                "Home size (sq ft)",
                min_value=0,
                value=2000,
                step=100,
                key="cooling_sqft"
            )
            
            cooling_months = st.slider(
                "Cooling months per year",
                min_value=1,
                max_value=12,
                value=4,
                key="cooling_months"
            )
        
        with col2:
            hours_per_day = st.slider(
                "Hours of AC use per day",
                min_value=1,
                max_value=24,
                value=8,
                key="ac_hours"
            )
        
        if st.button("Compare Cooling Systems", key="compare_cooling"):
            # Approximate BTU/hr needed
            btu_per_sqft_hr = 20
            total_hours = cooling_months * 30 * hours_per_day
            annual_btu = cooling_sqft * btu_per_sqft_hr * total_hours
            
            systems = {
                "Window AC (SEER 10)": {"seer": 10, "install_cost": 400},
                "Central AC (SEER 14)": {"seer": 14, "install_cost": 4000},
                "Central AC (SEER 18)": {"seer": 18, "install_cost": 6000},
                "Heat Pump (SEER 20)": {"seer": 20, "install_cost": 7500},
                "Ductless Mini-Split": {"seer": 22, "install_cost": 5000},
                "Evaporative Cooler": {"seer": 40, "install_cost": 2500}  # Very efficient in dry climates
            }
            
            elec_rate = 0.14  # $/kWh
            
            st.success("📊 Cooling System Comparison")
            
            comparison_data = {
                "System": [],
                "SEER Rating": [],
                "Annual kWh": [],
                "Annual Cost": [],
                "Install Cost": [],
                "5-Year Total": []
            }
            
            for system, data in systems.items():
                annual_kwh = annual_btu / (data["seer"] * 1000)
                annual_cost = annual_kwh * elec_rate
                five_year = (annual_cost * 5) + data["install_cost"]
                
                comparison_data["System"].append(system)
                comparison_data["SEER Rating"].append(str(data["seer"]))
                comparison_data["Annual kWh"].append(f"{annual_kwh:.0f}")
                comparison_data["Annual Cost"].append(f"${annual_cost:.0f}")
                comparison_data["Install Cost"].append(f"${data['install_cost']:,}")
                comparison_data["5-Year Total"].append(f"${five_year:,.0f}")
            
            df = pd.DataFrame(comparison_data)
            st.dataframe(df)
    
    elif comparison_type == "💧 Water Systems":
        st.markdown("#### Compare Water Systems & Conservation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            household_size = st.number_input(
                "Household size (people)",
                min_value=1,
                value=4,
                step=1,
                key="household_size"
            )
            
            water_rate = st.number_input(
                "Water rate ($/gallon)",
                min_value=0.0,
                value=0.005,
                step=0.001,
                format="%.3f",
                key="water_rate_comp"
            )
        
        with col2:
            current_gpd = st.number_input(
                "Current usage (gallons/person/day)",
                min_value=0.0,
                value=80.0,
                step=5.0,
                key="current_gpd"
            )
        
        if st.button("Compare Water Systems", key="compare_water"):
            baseline_annual = household_size * current_gpd * 365
            
            systems = {
                "Standard (No Changes)": {"reduction": 0, "install_cost": 0},
                "Low-Flow Fixtures": {"reduction": 0.20, "install_cost": 200},
                "Dual-Flush Toilets": {"reduction": 0.15, "install_cost": 800},
                "Rainwater Harvesting": {"reduction": 0.30, "install_cost": 3000},
                "Greywater System": {"reduction": 0.35, "install_cost": 5000},
                "Full Conservation Package": {"reduction": 0.50, "install_cost": 8000}
            }
            
            st.success("📊 Water System Comparison")
            
            comparison_data = {
                "System": [],
                "Annual Gallons": [],
                "Annual Cost": [],
                "Water Saved": [],
                "Annual Savings": [],
                "Payback (Years)": []
            }
            
            baseline_cost = baseline_annual * water_rate
            
            for system, data in systems.items():
                annual_gallons = baseline_annual * (1 - data["reduction"])
                annual_cost = annual_gallons * water_rate
                water_saved = baseline_annual * data["reduction"]
                annual_savings = baseline_cost - annual_cost
                payback = data["install_cost"] / annual_savings if annual_savings > 0 else 0
                
                comparison_data["System"].append(system)
                comparison_data["Annual Gallons"].append(f"{annual_gallons:,.0f}")
                comparison_data["Annual Cost"].append(f"${annual_cost:.0f}")
                comparison_data["Water Saved"].append(f"{water_saved:,.0f} gal")
                comparison_data["Annual Savings"].append(f"${annual_savings:.0f}")
                comparison_data["Payback (Years)"].append(f"{payback:.1f}" if payback > 0 else "-")
            
            df = pd.DataFrame(comparison_data)
            st.dataframe(df)
    
    elif comparison_type == "🚗 Transportation Fuel":
        st.markdown("#### Compare Transportation Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            annual_miles = st.number_input(
                "Annual miles driven",
                min_value=0,
                value=12000,
                step=1000,
                key="annual_miles"
            )
            
            gas_price = st.number_input(
                "Gas price ($/gallon)",
                min_value=0.0,
                value=3.50,
                step=0.10,
                format="%.2f",
                key="gas_price"
            )
        
        with col2:
            elec_price = st.number_input(
                "Electricity price ($/kWh)",
                min_value=0.0,
                value=0.14,
                step=0.01,
                format="%.2f",
                key="elec_price_car"
            )
        
        if st.button("Compare Transportation", key="compare_transport"):
            vehicles = {
                "Gas Car (25 MPG)": {"mpg": 25, "type": "gas", "co2_per_mile": 0.89},
                "Gas Car (35 MPG)": {"mpg": 35, "type": "gas", "co2_per_mile": 0.64},
                "Hybrid (50 MPG)": {"mpg": 50, "type": "gas", "co2_per_mile": 0.44},
                "Plug-in Hybrid (100 MPGe)": {"mpg": 100, "type": "hybrid", "co2_per_mile": 0.25},
                "Electric Vehicle (4 mi/kWh)": {"mi_per_kwh": 4, "type": "electric", "co2_per_mile": 0.12},
                "Electric Vehicle (3 mi/kWh)": {"mi_per_kwh": 3, "type": "electric", "co2_per_mile": 0.16}
            }
            
            st.success("📊 Transportation Comparison")
            
            comparison_data = {
                "Vehicle Type": [],
                "Annual Fuel Cost": [],
                "CO₂/year (lbs)": [],
                "5-Year Fuel Cost": [],
                "Savings vs 25 MPG": []
            }
            
            baseline_cost = (annual_miles / 25) * gas_price * 5
            
            for vehicle, data in vehicles.items():
                if data["type"] == "gas" or data["type"] == "hybrid":
                    annual_cost = (annual_miles / data["mpg"]) * gas_price
                else:
                    annual_cost = (annual_miles / data["mi_per_kwh"]) * elec_price
                
                annual_co2 = annual_miles * data["co2_per_mile"]
                five_year = annual_cost * 5
                savings = baseline_cost - five_year
                
                comparison_data["Vehicle Type"].append(vehicle)
                comparison_data["Annual Fuel Cost"].append(f"${annual_cost:.0f}")
                comparison_data["CO₂/year (lbs)"].append(f"{annual_co2:.0f}")
                comparison_data["5-Year Fuel Cost"].append(f"${five_year:,.0f}")
                comparison_data["Savings vs 25 MPG"].append(f"${savings:,.0f}" if savings > 0 else "-")
            
            df = pd.DataFrame(comparison_data)
            st.dataframe(df)
    
    elif comparison_type == "🏡 Home Energy Audit":
        st.markdown("#### Quick Home Energy Audit")
        
        col1, col2 = st.columns(2)
        
        with col1:
            home_sqft = st.number_input(
                "Home size (sq ft)",
                min_value=0,
                value=2000,
                step=100,
                key="audit_sqft"
            )
            
            home_age = st.selectbox(
                "Home age",
                ["New (< 5 years)", "Modern (5-15 years)", "Older (15-30 years)", "Vintage (30+ years)"],
                index=2,
                key="home_age"
            )
            
            current_bill = st.number_input(
                "Average monthly energy bill ($)",
                min_value=0.0,
                value=200.0,
                step=10.0,
                key="current_bill"
            )
        
        with col2:
            insulation = st.selectbox(
                "Insulation quality",
                ["Poor", "Average", "Good", "Excellent"],
                index=1,
                key="insulation_quality"
            )
            
            window_type = st.selectbox(
                "Window type",
                ["Single pane", "Double pane", "Triple pane", "Low-E"],
                index=1,
                key="window_type"
            )
            
            hvac_age = st.slider(
                "HVAC system age (years)",
                min_value=0,
                max_value=25,
                value=10,
                key="hvac_age"
            )
        
        if st.button("Generate Audit Report", key="audit_report"):
            # Calculate potential savings
            age_factor = {"New (< 5 years)": 0.05, "Modern (5-15 years)": 0.10, "Older (15-30 years)": 0.20, "Vintage (30+ years)": 0.35}
            insulation_factor = {"Poor": 0.20, "Average": 0.10, "Good": 0.05, "Excellent": 0.02}
            window_factor = {"Single pane": 0.15, "Double pane": 0.08, "Triple pane": 0.03, "Low-E": 0.02}
            hvac_factor = min(hvac_age * 0.01, 0.20)
            
            total_potential = age_factor[home_age] + insulation_factor[insulation] + window_factor[window_type] + hvac_factor
            monthly_savings_potential = current_bill * total_potential
            
            st.success("📊 Home Energy Audit Results")
            
            st.markdown("#### 💰 Savings Potential")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Monthly Savings", f"${monthly_savings_potential:.0f}")
            with col2:
                st.metric("Annual Savings", f"${monthly_savings_potential * 12:.0f}")
            with col3:
                st.metric("Efficiency Score", f"{100 - (total_potential * 100):.0f}/100")
            
            st.markdown("#### 🔧 Recommended Improvements")
            
            improvements = []
            if insulation in ["Poor", "Average"]:
                improvements.append(("🏠 Upgrade Insulation", insulation_factor[insulation] * current_bill * 12))
            if window_type in ["Single pane", "Double pane"]:
                improvements.append(("🪟 Upgrade Windows", window_factor[window_type] * current_bill * 12))
            if hvac_age > 10:
                improvements.append(("❄️ Replace HVAC System", hvac_factor * current_bill * 12))
            improvements.append(("💡 LED Lighting", current_bill * 0.05 * 12))
            improvements.append(("🌡️ Smart Thermostat", current_bill * 0.08 * 12))
            
            for improvement, annual_savings in sorted(improvements, key=lambda x: x[1], reverse=True):
                st.write(f"- {improvement}: Save ~${annual_savings:.0f}/year")


def display_utility_comparison(comparison):
    """Display utility comparison results."""
    st.success("📊 Comparison Complete!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Current Monthly Cost",
            f"${comparison['current_monthly_cost']:.2f}"
        )
    
    with col2:
        st.metric(
            "Optimized Monthly Cost",
            f"${comparison['optimized_monthly_cost']:.2f}"
        )
    
    with col3:
        st.metric(
            "Monthly Savings",
            f"${comparison['monthly_savings']:.2f}",
            delta=f"-{comparison['savings_percentage']:.1f}%",
            delta_color="inverse"
        )
    
    st.metric(
        "Annual Savings",
        f"${comparison['annual_savings']:.2f}",
        delta=f"+${comparison['annual_savings']:.2f}/year"
    )
    
    # Breakdown
    st.markdown("#### 💡 Savings Breakdown")
    breakdown = comparison['breakdown']
    
    breakdown_data = {
        "Category": ["Electricity", "Natural Gas", "Water"],
        "Monthly Savings": [
            breakdown['electricity_savings'],
            breakdown['gas_savings'],
            breakdown['water_savings']
        ]
    }
    
    df = pd.DataFrame(breakdown_data)
    st.bar_chart(df.set_index("Category"))


def render_carbon_credit_calculator(calculator: FinancialCalculator):
    """Render carbon credit/tax calculator interface."""
    st.markdown("### 🌱 Carbon Credit/Tax Calculator")
    st.markdown("Estimate your carbon credit earnings or tax liability.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        annual_emissions = st.number_input(
            "Annual CO₂ Emissions (kg)",
            min_value=0.0,
            value=5000.0,
            step=100.0,
            help="Your total annual CO₂ emissions",
            key="annual_emissions"
        )
        
        annual_reductions = st.number_input(
            "Annual CO₂ Reductions (kg)",
            min_value=0.0,
            value=2000.0,
            step=100.0,
            help="CO₂ reductions from green initiatives",
            key="annual_reductions"
        )
    
    with col2:
        carbon_price = st.number_input(
            "Carbon Price ($/ton CO₂)",
            min_value=0.0,
            value=50.0,
            step=5.0,
            help="Price per metric ton of CO₂",
            key="carbon_price"
        )
        
        st.markdown("---")
        st.markdown("**Common Carbon Prices:**")
        st.markdown("- EU ETS: ~€80-100/ton")
        st.markdown("- California: ~$30/ton")
        st.markdown("- Canada: ~CAD$65/ton")
    
    if st.button("Calculate Carbon Position", key="calc_carbon"):
        credit = calculator.calculate_carbon_credit(
            annual_co2_reduction_kg=annual_reductions,
            annual_co2_emissions_kg=annual_emissions,
            carbon_price=carbon_price
        )
        
        st.success("🌱 Carbon Position Calculated!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Net Annual CO₂",
                f"{credit.annual_co2_kg:,.0f} kg",
                help="Emissions minus reductions"
            )
        
        with col2:
            st.metric(
                "Credit Value",
                f"${credit.annual_credit_value:,.2f}",
                help="Value of carbon credits earned"
            )
        
        with col3:
            st.metric(
                "Tax Liability",
                f"${credit.annual_tax_liability:,.2f}",
                help="Potential carbon tax owed"
            )
        
        # Net position
        if credit.net_position > 0:
            st.success(f"✅ **Net Position:** You could earn **${credit.net_position:,.2f}** in carbon credits!")
        elif credit.net_position < 0:
            st.warning(f"⚠️ **Net Position:** You would owe **${abs(credit.net_position):,.2f}** in carbon taxes.")
        else:
            st.info("⚖️ **Net Position:** Carbon neutral - emissions balanced by reductions!")
        
        # Tips
        st.markdown("---")
        st.markdown("#### 💡 Tips to Improve Your Carbon Position")
        
        if annual_emissions > annual_reductions:
            st.markdown("""
            - Install solar panels (saves ~4,000 kg CO₂/year)
            - Switch to electric vehicle (saves ~3,500 kg CO₂/year)
            - Improve home insulation (saves ~800 kg CO₂/year)
            - Adopt plant-based meals 2-3x/week (saves ~500 kg CO₂/year)
            """)


def render_receipt_scanner():
    """Render the Receipt Scanner interface."""
    st.subheader("🧾 Receipt & Product Scanner")
    st.markdown("Analyze your purchases to understand their environmental impact.")
    
    # Initialize scanner
    scanner = ReceiptScanner()
    
    # Initialize session state for receipt scanner
    if 'receipt_analysis' not in st.session_state:
        st.session_state.receipt_analysis = None
    
    # Create sub-tabs
    scan_tab1, scan_tab2, scan_tab3 = st.tabs([
        "📷 Upload Receipt",
        "✏️ Manual Entry",
        "📊 Analysis History"
    ])
    
    with scan_tab1:
        render_receipt_upload(scanner)
    
    with scan_tab2:
        render_manual_product_entry(scanner)
    
    with scan_tab3:
        render_analysis_history()


def analyze_receipt_image_with_llm(image_base64: str) -> Optional[str]:
    """
    Analyze a receipt image using LLM vision capabilities.
    
    Args:
        image_base64: Base64 encoded image
        
    Returns:
        Extracted text/products from the receipt or None
    """
    import os
    
    # Try Groq first (with vision model)
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if groq_api_key:
        try:
            from groq import Groq
            
            client = Groq(api_key=groq_api_key)
            
            # Use Llama 4 Scout vision model for image analysis
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Analyze this receipt image and extract all purchased items.
For each item, list it in this format (one per line):
PRODUCT_NAME    $PRICE

Only include actual products, not totals, taxes, subtotals, or store information.
If you can see quantities, include them like: 2x PRODUCT_NAME    $PRICE
Be precise with the product names and prices you can read."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.warning(f"Groq vision analysis failed: {str(e)}")
            return None
    else:
        st.warning("⚠️ GROQ_API_KEY not set. Please set it in your .env file to enable image analysis.")
        return None


def render_receipt_upload(scanner: ReceiptScanner):
    """Render receipt image upload interface."""
    st.markdown("### 📷 Upload Receipt Image")
    st.markdown("Upload a photo of your receipt for automatic product detection and environmental analysis.")
    
    uploaded_file = st.file_uploader(
        "Choose a receipt image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear photo of your receipt",
        key="receipt_upload"
    )
    
    if uploaded_file is not None:
        # Display the image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Receipt", use_column_width=True)
        
        with col2:
            st.markdown("**Receipt Details:**")
            
            store_name = st.text_input(
                "Store Name (optional)",
                placeholder="e.g., Walmart, Whole Foods",
                key="store_name"
            )
            
            st.info("""
            📝 **Tips for best results:**
            - Ensure the receipt is well-lit
            - Text should be readable
            - Avoid creases or folds
            """)
            
            # Analyze image button
            if st.button("🔍 Analyze Receipt Image", key="analyze_image"):
                st.session_state.analyze_receipt_image = True
                st.session_state.image_base64 = None
                try:
                    import base64
                    uploaded_file.seek(0)
                    image_bytes = uploaded_file.read()
                    st.session_state.image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    st.session_state.receipt_store_name = store_name or "Unknown Store"
                except Exception as e:
                    st.error(f"Error reading image: {str(e)}")
    
    # Process image analysis outside of columns to avoid nesting issues
    if st.session_state.get('analyze_receipt_image') and st.session_state.get('image_base64'):
        st.session_state.analyze_receipt_image = False
        
        with st.spinner("🤖 Analyzing receipt with AI..."):
            try:
                extracted_text = analyze_receipt_image_with_llm(st.session_state.image_base64)
                
                if extracted_text:
                    st.success("✅ Receipt analyzed successfully!")
                    st.markdown("**Detected Items:**")
                    st.text(extracted_text)
                    
                    # Parse and analyze
                    analysis = scanner.analyze_receipt(
                        text=extracted_text,
                        store_name=st.session_state.get('receipt_store_name', "Unknown Store")
                    )
                    st.session_state.receipt_analysis = analysis
                    display_receipt_analysis(analysis, scanner)
                else:
                    st.warning("Could not extract text from image. Please use the text input below or Manual Entry tab.")
            except Exception as e:
                st.error(f"Error analyzing image: {str(e)}")
                st.info("💡 Try pasting the receipt text below instead.")
    
    # Alternative: Paste text from receipt
    st.markdown("---")
    st.markdown("### 📋 Or Paste Receipt Text")
    
    receipt_text = st.text_area(
        "Paste receipt text here",
        placeholder="Apple          $1.99\nMilk 1 gallon  $4.50\nChicken breast $8.99\n...",
        height=150,
        key="receipt_text"
    )
    
    if st.button("Analyze Receipt Text", key="analyze_receipt"):
        if receipt_text.strip():
            with st.spinner("Analyzing receipt..."):
                analysis = scanner.analyze_receipt(
                    text=receipt_text,
                    store_name=store_name if 'store_name' in dir() else "Unknown Store"
                )
                st.session_state.receipt_analysis = analysis
                display_receipt_analysis(analysis, scanner)
        else:
            st.warning("Please enter receipt text to analyze.")


def render_manual_product_entry(scanner: ReceiptScanner):
    """Render manual product entry interface."""
    st.markdown("### ✏️ Manual Product Entry")
    st.markdown("Enter your purchased products to calculate their environmental impact.")
    
    # Initialize product list in session state
    if 'manual_products' not in st.session_state:
        st.session_state.manual_products = []
    
    # Add product form
    with st.form("add_product_form"):
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        
        with col1:
            product_name = st.text_input(
                "Product Name",
                placeholder="e.g., Organic Apples",
                key="product_name"
            )
        
        with col2:
            quantity = st.number_input(
                "Quantity",
                min_value=0.1,
                value=1.0,
                step=0.5,
                key="product_qty"
            )
        
        with col3:
            unit = st.selectbox(
                "Unit",
                ["unit", "kg", "lb", "liter"],
                key="product_unit"
            )
        
        with col4:
            price = st.number_input(
                "Price ($)",
                min_value=0.0,
                value=0.0,
                step=0.50,
                key="product_price"
            )
        
        submitted = st.form_submit_button("Add Product")
        
        if submitted and product_name:
            st.session_state.manual_products.append({
                "name": product_name,
                "quantity": quantity,
                "unit": unit,
                "price": price
            })
            st.success(f"Added: {product_name}")
    
    # Display current products
    if st.session_state.manual_products:
        st.markdown("#### 📝 Your Products")
        
        for idx, product in enumerate(st.session_state.manual_products):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.write(f"{idx + 1}. {product['name']} - {product['quantity']} {product['unit']} (${product['price']:.2f})")
            
            with col2:
                if st.button("❌", key=f"remove_{idx}"):
                    st.session_state.manual_products.pop(idx)
                    st.rerun()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Analyze Products", key="analyze_products"):
                st.session_state.trigger_manual_analysis = True
        
        with col2:
            if st.button("🗑️ Clear All", key="clear_products"):
                st.session_state.manual_products = []
                st.rerun()
        
        # Process analysis outside of columns
        if st.session_state.get('trigger_manual_analysis'):
            st.session_state.trigger_manual_analysis = False
            with st.spinner("Analyzing products..."):
                analysis = scanner.analyze_receipt(
                    products=st.session_state.manual_products,
                    store_name="Manual Entry"
                )
                st.session_state.receipt_analysis = analysis
                display_receipt_analysis(analysis, scanner)
    else:
        st.info("Add products using the form above.")


def display_receipt_analysis(analysis: ReceiptAnalysis, scanner: ReceiptScanner):
    """Display receipt analysis results."""
    st.success("✅ Analysis Complete!")
    
    # Summary metrics
    st.markdown("### 📊 Environmental Impact Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🌡️ Total CO₂",
            f"{analysis.total_co2_kg:.2f} kg",
            help="Carbon footprint of all products"
        )
    
    with col2:
        st.metric(
            "💧 Total Water",
            f"{analysis.total_water_liters:.0f} L",
            help="Embedded water in all products"
        )
    
    with col3:
        st.metric(
            "🗑️ Total Waste",
            f"{analysis.total_waste_kg:.3f} kg",
            help="Packaging and waste generated"
        )
    
    with col4:
        score_color = "🟢" if analysis.average_sustainability_score >= 70 else "🟡" if analysis.average_sustainability_score >= 50 else "🔴"
        st.metric(
            f"{score_color} Sustainability Score",
            f"{analysis.average_sustainability_score:.0f}/100",
            help="Average sustainability score"
        )
    
    # Comparison to average
    comparison = scanner.compare_to_average(analysis)
    
    st.markdown("### 📈 How You Compare")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**CO₂:** {comparison['co2']}")
    with col2:
        st.markdown(f"**Water:** {comparison['water']}")
    with col3:
        st.markdown(f"**Waste:** {comparison['waste']}")
    
    # Product details
    st.markdown("### 📦 Product Details")
    
    for idx, product in enumerate(analysis.products, 1):
        with st.expander(f"**{idx}. {product.name}** - Score: {product.sustainability_score:.0f}/100"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Category:** {product.category.value.replace('_', ' ').title()}")
                st.write(f"**Quantity:** {product.quantity} {product.unit}")
                st.write(f"**Price:** ${product.price:.2f}")
            
            with col2:
                st.write(f"**CO₂:** {product.co2_kg:.2f} kg")
                st.write(f"**Water:** {product.water_liters:.0f} L")
                st.write(f"**Waste:** {product.waste_kg:.3f} kg")
            
            with col3:
                if product.eco_alternatives:
                    st.write("**Eco Alternatives:**")
                    for alt in product.eco_alternatives:
                        st.write(f"• {alt}")
    
    # Category breakdown
    st.markdown("### 📊 Impact by Category")
    
    category_summary = scanner.get_category_summary(analysis)
    
    if category_summary:
        cat_data = {
            "Category": [],
            "CO₂ (kg)": [],
            "Water (L)": [],
            "Items": []
        }
        
        for cat, data in category_summary.items():
            cat_data["Category"].append(cat.replace("_", " ").title())
            cat_data["CO₂ (kg)"].append(round(data["total_co2_kg"], 2))
            cat_data["Water (L)"].append(round(data["total_water_liters"], 0))
            cat_data["Items"].append(data["count"])
        
        df = pd.DataFrame(cat_data)
        st.dataframe(df)
        
        st.bar_chart(df.set_index("Category")["CO₂ (kg)"])
    
    # Recommendations
    st.markdown("### 💡 Eco Recommendations")
    
    for rec in analysis.eco_recommendations:
        st.markdown(f"• {rec}")


def render_analysis_history():
    """Render analysis history interface."""
    st.markdown("### 📊 Analysis History")
    
    if st.session_state.receipt_analysis:
        analysis = st.session_state.receipt_analysis
        
        st.markdown(f"**Last Analysis:** {analysis.date}")
        st.markdown(f"**Store:** {analysis.store_name}")
        st.markdown(f"**Products:** {len(analysis.products)}")
        st.markdown(f"**Total CO₂:** {analysis.total_co2_kg:.2f} kg")
        st.markdown(f"**Sustainability Score:** {analysis.average_sustainability_score:.0f}/100")
        
        if st.button("View Full Analysis", key="view_full"):
            scanner = ReceiptScanner()
            display_receipt_analysis(analysis, scanner)
    else:
        st.info("No analysis history yet. Upload a receipt or enter products manually to get started.")


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("🌍 Environmental Impact AI Agent")
    st.markdown(
        "Get personalized recommendations to reduce your environmental footprint. "
        "Analyze CO₂ emissions, water usage, energy consumption, and waste generation. "
        "Ask questions about your activities or upload your data for comprehensive analysis."
    )
    
    # Initialize agent with error handling
    try:
        agent = initialize_agent()
    except Exception as e:
        st.error(f"❌ Failed to initialize agent: {str(e)}")
        st.info("💡 Try reinstalling dependencies: `pip install --upgrade sentence-transformers torch`")
        st.stop()
    
    # Check system health
    if agent:
        with st.sidebar:
            st.header("🔧 System Status")
            health = agent.check_system_health()
            
            if health["overall_healthy"]:
                st.success("✅ All systems operational")
            else:
                st.warning("⚠️ Some components unavailable")
            
            with st.expander("View Details"):
                st.write(f"LLM: {'✅' if health['llm_available'] else '❌'}")
                st.write(f"Vector Store: {'✅' if health['vector_store_ready'] else '❌'}")
                st.write(f"Reference Data: {'✅' if health['reference_data_loaded'] else '❌'}")
                
                if health.get('errors'):
                    st.error("Errors:")
                    for error in health['errors']:
                        st.write(f"- {error}")
            
            st.markdown("---")
            st.header("📊 Metrics We Track")
            st.markdown("""
            - 🌡️ **CO₂ Emissions** - Carbon footprint
            - 💧 **Water Usage** - Water consumption
            - ⚡ **Energy** - Electricity & fuel use
            - 🗑️ **Waste** - Waste generation
            - 🏭 **Pollution** - Air & water quality impact
            """)
            
            st.markdown("---")
            st.header("🎯 Quick Tips")
            st.markdown("""
            1. Be specific about quantities
            2. Mention time periods (daily, weekly)
            3. Include the activity type
            4. Ask for comparisons
            """)
    else:
        st.error("❌ Failed to initialize the agent. Please check the configuration.")
        st.stop()
    
    # Main content area
    st.markdown("---")
    
    # Create tabs for different interaction modes
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 Ask a Question", 
        "📊 Upload Dataset", 
        "📈 Environmental Dashboard",
        "💰 Financial Calculator",
        "🧾 Receipt Scanner"
    ])
    
    with tab1:
        render_query_interface(agent)
    
    with tab2:
        render_file_upload_interface(agent)
    
    with tab3:
        render_environmental_dashboard()
    
    with tab4:
        render_financial_calculator()
    
    with tab5:
        render_receipt_scanner()


def render_environmental_dashboard():
    """Render a comprehensive environmental impact dashboard."""
    st.subheader("📈 Environmental Impact Dashboard")
    st.markdown("Track and compare environmental impacts across different activities and categories.")
    
    # Create dashboard tabs
    dash_tab1, dash_tab2, dash_tab3, dash_tab4, dash_tab5 = st.tabs([
        "🔢 Impact Calculator",
        "📊 Activity Comparison",
        "🎯 Goal Tracker",
        "🌍 Footprint Analyzer",
        "💡 Eco Tips"
    ])
    
    with dash_tab1:
        render_impact_calculator()
    
    with dash_tab2:
        render_activity_comparison()
    
    with dash_tab3:
        render_goal_tracker()
    
    with dash_tab4:
        render_footprint_analyzer()
    
    with dash_tab5:
        render_eco_tips()


def render_impact_calculator():
    """Interactive environmental impact calculator."""
    st.markdown("### 🔢 Personal Impact Calculator")
    st.markdown("Enter your daily activities to calculate your environmental footprint.")
    
    # Transport section
    st.markdown("#### 🚗 Transportation")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        car_km = st.number_input("Car driving (km/day)", min_value=0.0, value=20.0, step=5.0, key="dash_car")
        car_type = st.selectbox("Car type", ["Petrol", "Diesel", "Hybrid", "Electric", "None"], key="dash_car_type")
    
    with col2:
        public_transit = st.number_input("Public transit (km/day)", min_value=0.0, value=0.0, step=5.0, key="dash_transit")
        flights_month = st.number_input("Flights per month", min_value=0, value=0, step=1, key="dash_flights")
    
    with col3:
        bike_walk = st.number_input("Bike/Walk (km/day)", min_value=0.0, value=2.0, step=1.0, key="dash_bike")
    
    # Home section
    st.markdown("#### 🏠 Home Energy")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        electricity_kwh = st.number_input("Electricity (kWh/day)", min_value=0.0, value=10.0, step=1.0, key="dash_elec")
        renewable_percent = st.slider("Renewable energy %", 0, 100, 0, key="dash_renewable")
    
    with col2:
        gas_usage = st.number_input("Natural gas (therms/day)", min_value=0.0, value=1.0, step=0.1, key="dash_gas")
        home_size = st.selectbox("Home size", ["Apartment", "Small house", "Medium house", "Large house"], key="dash_home")
    
    with col3:
        heating_type = st.selectbox("Heating type", ["Gas", "Electric", "Heat pump", "Oil", "Wood"], key="dash_heat")
    
    # Food section
    st.markdown("#### 🍽️ Food & Diet")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        diet_type = st.selectbox("Diet type", ["Heavy meat", "Average", "Low meat", "Vegetarian", "Vegan"], key="dash_diet")
    
    with col2:
        local_food = st.slider("Local food %", 0, 100, 30, key="dash_local")
    
    with col3:
        food_waste = st.selectbox("Food waste level", ["High", "Average", "Low", "Minimal"], key="dash_waste")
    
    # Water section
    st.markdown("#### 💧 Water Usage")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        shower_mins = st.number_input("Shower time (mins/day)", min_value=0, value=10, step=1, key="dash_shower")
    
    with col2:
        water_fixtures = st.selectbox("Water fixtures", ["Standard", "Low-flow", "Ultra low-flow"], key="dash_fixtures")
    
    with col3:
        garden_water = st.selectbox("Garden watering", ["None", "Minimal", "Moderate", "Heavy"], key="dash_garden")
    
    # Calculate impact
    if st.button("🔍 Calculate My Impact", key="calc_impact"):
        # CO2 calculations (kg/day)
        car_emissions = {"Petrol": 0.21, "Diesel": 0.18, "Hybrid": 0.10, "Electric": 0.05, "None": 0}
        co2_transport = car_km * car_emissions.get(car_type, 0.21) + public_transit * 0.05 + flights_month * 90 / 30
        
        electricity_factor = 0.42 * (1 - renewable_percent/100)
        co2_home = electricity_kwh * electricity_factor + gas_usage * 5.3
        
        diet_emissions = {"Heavy meat": 7.0, "Average": 4.5, "Low meat": 3.0, "Vegetarian": 2.0, "Vegan": 1.5}
        co2_food = diet_emissions.get(diet_type, 4.5) * (1 - local_food/200)
        
        total_co2 = co2_transport + co2_home + co2_food
        
        # Water calculations (liters/day)
        shower_water = shower_mins * {"Standard": 12, "Low-flow": 8, "Ultra low-flow": 5}.get(water_fixtures, 12)
        garden_factors = {"None": 0, "Minimal": 20, "Moderate": 50, "Heavy": 100}
        total_water = shower_water + 50 + garden_factors.get(garden_water, 20)  # +50 for other uses
        
        # Energy (kWh/day)
        total_energy = electricity_kwh + gas_usage * 29.3  # Convert therms to kWh equivalent
        
        # Display results
        st.success("📊 Your Daily Environmental Impact")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🌡️ CO₂ Emissions", f"{total_co2:.1f} kg/day", 
                     delta=f"{(total_co2 - 15) * -1:.1f} vs avg" if total_co2 < 15 else f"+{total_co2 - 15:.1f} vs avg",
                     delta_color="inverse" if total_co2 > 15 else "normal")
        
        with col2:
            st.metric("💧 Water Usage", f"{total_water:.0f} L/day",
                     delta=f"{(total_water - 150) * -1:.0f} vs avg" if total_water < 150 else f"+{total_water - 150:.0f} vs avg",
                     delta_color="inverse" if total_water > 150 else "normal")
        
        with col3:
            st.metric("⚡ Energy Use", f"{total_energy:.1f} kWh/day",
                     delta=f"{(total_energy - 25) * -1:.1f} vs avg" if total_energy < 25 else f"+{total_energy - 25:.1f} vs avg",
                     delta_color="inverse" if total_energy > 25 else "normal")
        
        with col4:
            annual_co2 = total_co2 * 365 / 1000
            st.metric("📅 Annual CO₂", f"{annual_co2:.1f} tons/year")
        
        # Impact breakdown chart
        st.markdown("#### 📈 Impact Breakdown")
        breakdown_data = {
            "Category": ["Transport", "Home Energy", "Food"],
            "CO₂ (kg/day)": [co2_transport, co2_home, co2_food]
        }
        df = pd.DataFrame(breakdown_data)
        st.bar_chart(df.set_index("Category"))
        
        # Equivalents
        st.markdown("#### 🌳 What This Means")
        trees_needed = int(total_co2 * 365 / 21)  # ~21 kg CO2/tree/year
        driving_equiv = int(total_co2 / 0.21)  # km in petrol car
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"🌲 You'd need **{trees_needed} trees** to offset your annual CO₂")
        with col2:
            st.info(f"🚗 Daily impact = driving **{driving_equiv} km** in a petrol car")
        with col3:
            showers = int(total_water / 60)  # 60L per 5-min shower
            st.info(f"🚿 Water usage = **{showers}** five-minute showers")


def render_activity_comparison():
    """Compare environmental impact of different activities."""
    st.markdown("### 📊 Activity Impact Comparison")
    st.markdown("Compare the environmental cost of everyday activities.")
    
    comparison_category = st.selectbox(
        "Select category to compare",
        ["🚗 Transportation", "🍽️ Food & Meals", "🏠 Household", "🛍️ Shopping", "🎮 Entertainment"],
        key="compare_category"
    )
    
    if comparison_category == "🚗 Transportation":
        st.markdown("#### CO₂ Emissions per 10 km traveled")
        transport_data = {
            "Mode": ["Petrol Car", "Diesel Car", "Hybrid Car", "Electric Car", "Motorcycle", "Bus", "Train", "E-bike", "Bicycle", "Walking"],
            "CO₂ (kg)": [2.1, 1.8, 1.0, 0.5, 1.2, 0.4, 0.1, 0.05, 0, 0],
            "Cost ($)": [3.0, 2.5, 1.5, 0.5, 1.0, 2.0, 1.5, 0.10, 0, 0],
            "Time (min)": [15, 15, 15, 15, 12, 25, 18, 25, 30, 120]
        }
        df = pd.DataFrame(transport_data)
        
        metric = st.radio("View by:", ["CO₂ (kg)", "Cost ($)", "Time (min)"], horizontal=True, key="transport_metric")
        st.bar_chart(df.set_index("Mode")[metric])
        st.dataframe(df)
        
    elif comparison_category == "🍽️ Food & Meals":
        st.markdown("#### Environmental Impact per Meal")
        food_data = {
            "Meal Type": ["Beef steak", "Pork chop", "Chicken breast", "Fish fillet", "Eggs & cheese", "Vegetarian", "Vegan"],
            "CO₂ (kg)": [6.5, 2.5, 1.5, 1.2, 1.0, 0.5, 0.3],
            "Water (L)": [2000, 800, 500, 400, 300, 200, 100],
            "Land (m²)": [25, 8, 5, 1, 4, 2, 1]
        }
        df = pd.DataFrame(food_data)
        
        metric = st.radio("View by:", ["CO₂ (kg)", "Water (L)", "Land (m²)"], horizontal=True, key="food_metric")
        st.bar_chart(df.set_index("Meal Type")[metric])
        st.dataframe(df)
        
    elif comparison_category == "🏠 Household":
        st.markdown("#### Daily Household Activities Impact")
        household_data = {
            "Activity": ["10-min shower", "Bath", "Dishwasher load", "Washing machine", "1 hour AC", "1 hour heating", "Cooking (gas)", "Cooking (electric)"],
            "CO₂ (kg)": [0.8, 1.5, 0.5, 0.6, 1.2, 2.0, 0.3, 0.4],
            "Water (L)": [80, 150, 15, 50, 0, 0, 0, 0],
            "Energy (kWh)": [2.0, 4.0, 1.5, 1.5, 3.0, 5.0, 0.8, 1.0]
        }
        df = pd.DataFrame(household_data)
        
        metric = st.radio("View by:", ["CO₂ (kg)", "Water (L)", "Energy (kWh)"], horizontal=True, key="house_metric")
        st.bar_chart(df.set_index("Activity")[metric])
        st.dataframe(df)
        
    elif comparison_category == "🛍️ Shopping":
        st.markdown("#### Shopping Choices Impact")
        shopping_data = {
            "Item": ["New smartphone", "Refurb. smartphone", "New jeans", "Second-hand jeans", "New cotton shirt", "Plastic bag", "Reusable bag"],
            "CO₂ (kg)": [70, 15, 25, 2, 8, 0.1, 1.0],
            "Water (L)": [12000, 0, 7500, 0, 2500, 50, 100],
            "Waste (g)": [200, 50, 100, 10, 50, 5, 0]
        }
        df = pd.DataFrame(shopping_data)
        
        metric = st.radio("View by:", ["CO₂ (kg)", "Water (L)", "Waste (g)"], horizontal=True, key="shop_metric")
        st.bar_chart(df.set_index("Item")[metric])
        st.dataframe(df)
        
    elif comparison_category == "🎮 Entertainment":
        st.markdown("#### Entertainment Activities (per hour)")
        entertainment_data = {
            "Activity": ["Video streaming (HD)", "Video streaming (4K)", "Gaming (console)", "Gaming (PC)", "Reading (e-reader)", "Reading (lamp)", "Outdoor sports"],
            "CO₂ (kg)": [0.036, 0.08, 0.15, 0.25, 0.003, 0.01, 0],
            "Energy (kWh)": [0.1, 0.2, 0.4, 0.6, 0.008, 0.03, 0]
        }
        df = pd.DataFrame(entertainment_data)
        
        metric = st.radio("View by:", ["CO₂ (kg)", "Energy (kWh)"], horizontal=True, key="ent_metric")
        st.bar_chart(df.set_index("Activity")[metric])
        st.dataframe(df)


def render_goal_tracker():
    """Track environmental sustainability goals."""
    st.markdown("### 🎯 Sustainability Goal Tracker")
    st.markdown("Set and track your environmental goals.")
    
    # Initialize session state for goals
    if 'env_goals' not in st.session_state:
        st.session_state.env_goals = {
            'co2_target': 10.0,
            'water_target': 120,
            'energy_target': 15.0,
            'waste_target': 0.5
        }
    
    st.markdown("#### 📝 Set Your Daily Targets")
    
    col1, col2 = st.columns(2)
    
    with col1:
        co2_target = st.number_input("🌡️ CO₂ Target (kg/day)", min_value=1.0, max_value=50.0, 
                                     value=st.session_state.env_goals['co2_target'], step=0.5, key="goal_co2")
        water_target = st.number_input("💧 Water Target (L/day)", min_value=50, max_value=500, 
                                       value=st.session_state.env_goals['water_target'], step=10, key="goal_water")
    
    with col2:
        energy_target = st.number_input("⚡ Energy Target (kWh/day)", min_value=5.0, max_value=100.0, 
                                        value=st.session_state.env_goals['energy_target'], step=1.0, key="goal_energy")
        waste_target = st.number_input("🗑️ Waste Target (kg/day)", min_value=0.1, max_value=5.0, 
                                       value=st.session_state.env_goals['waste_target'], step=0.1, key="goal_waste")
    
    if st.button("💾 Save Goals", key="save_goals"):
        st.session_state.env_goals = {
            'co2_target': co2_target,
            'water_target': water_target,
            'energy_target': energy_target,
            'waste_target': waste_target
        }
        st.success("✅ Goals saved successfully!")
    
    # Log today's activity
    st.markdown("#### 📊 Log Today's Activity")
    
    col1, col2 = st.columns(2)
    
    with col1:
        actual_co2 = st.number_input("Today's CO₂ (kg)", min_value=0.0, value=12.0, step=0.5, key="actual_co2")
        actual_water = st.number_input("Today's Water (L)", min_value=0, value=140, step=10, key="actual_water")
    
    with col2:
        actual_energy = st.number_input("Today's Energy (kWh)", min_value=0.0, value=18.0, step=1.0, key="actual_energy")
        actual_waste = st.number_input("Today's Waste (kg)", min_value=0.0, value=0.8, step=0.1, key="actual_waste")
    
    if st.button("📈 Check Progress", key="check_progress"):
        st.markdown("#### 📊 Today's Progress")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            co2_pct = (actual_co2 / co2_target) * 100
            color = "🟢" if co2_pct <= 100 else "🔴"
            st.metric(f"{color} CO₂", f"{actual_co2}/{co2_target} kg", 
                     delta=f"{co2_target - actual_co2:.1f} remaining" if actual_co2 < co2_target else f"+{actual_co2 - co2_target:.1f} over",
                     delta_color="normal" if actual_co2 < co2_target else "inverse")
            st.progress(min(co2_pct / 100, 1.0))
        
        with col2:
            water_pct = (actual_water / water_target) * 100
            color = "🟢" if water_pct <= 100 else "🔴"
            st.metric(f"{color} Water", f"{actual_water}/{water_target} L",
                     delta=f"{water_target - actual_water} remaining" if actual_water < water_target else f"+{actual_water - water_target} over",
                     delta_color="normal" if actual_water < water_target else "inverse")
            st.progress(min(water_pct / 100, 1.0))
        
        with col3:
            energy_pct = (actual_energy / energy_target) * 100
            color = "🟢" if energy_pct <= 100 else "🔴"
            st.metric(f"{color} Energy", f"{actual_energy}/{energy_target} kWh",
                     delta=f"{energy_target - actual_energy:.1f} remaining" if actual_energy < energy_target else f"+{actual_energy - energy_target:.1f} over",
                     delta_color="normal" if actual_energy < energy_target else "inverse")
            st.progress(min(energy_pct / 100, 1.0))
        
        with col4:
            waste_pct = (actual_waste / waste_target) * 100
            color = "🟢" if waste_pct <= 100 else "🔴"
            st.metric(f"{color} Waste", f"{actual_waste}/{waste_target} kg",
                     delta=f"{waste_target - actual_waste:.1f} remaining" if actual_waste < waste_target else f"+{actual_waste - waste_target:.1f} over",
                     delta_color="normal" if actual_waste < waste_target else "inverse")
            st.progress(min(waste_pct / 100, 1.0))
        
        # Overall score
        overall_score = 100 - ((co2_pct + water_pct + energy_pct + waste_pct) / 4 - 100)
        overall_score = max(0, min(100, overall_score))
        
        st.markdown("---")
        st.markdown(f"### 🏆 Today's Sustainability Score: **{overall_score:.0f}/100**")
        
        if overall_score >= 80:
            st.success("🌟 Excellent! You're making great sustainable choices!")
        elif overall_score >= 60:
            st.info("👍 Good job! A few small changes could make you even greener.")
        else:
            st.warning("💡 There's room for improvement. Check the Eco Tips tab for ideas!")


def render_footprint_analyzer():
    """Analyze and visualize carbon footprint."""
    st.markdown("### 🌍 Carbon Footprint Analyzer")
    st.markdown("Understand how your footprint compares to different benchmarks.")
    
    # Quick footprint estimate
    st.markdown("#### ⚡ Quick Annual Footprint Estimate")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        country = st.selectbox("Your country", 
                              ["USA", "UK", "Germany", "France", "Canada", "Australia", "Japan", "China", "India", "Brazil"],
                              key="fp_country")
        household_size = st.number_input("Household size", min_value=1, max_value=10, value=3, key="fp_household")
    
    with col2:
        housing = st.selectbox("Housing type", 
                              ["Apartment (small)", "Apartment (large)", "House (small)", "House (medium)", "House (large)"],
                              key="fp_housing")
        car_usage = st.selectbox("Car usage",
                                ["No car", "Occasional", "Regular", "Daily commute", "Heavy use"],
                                key="fp_car")
    
    with col3:
        diet = st.selectbox("Diet type",
                           ["Vegan", "Vegetarian", "Low meat", "Average", "High meat"],
                           key="fp_diet")
        flights_year = st.number_input("Flights per year", min_value=0, max_value=50, value=2, key="fp_flights")
    
    if st.button("🔍 Analyze Footprint", key="analyze_fp"):
        # Calculate footprint components
        country_base = {"USA": 16, "UK": 8, "Germany": 9, "France": 6, "Canada": 15, 
                       "Australia": 17, "Japan": 9, "China": 8, "India": 2, "Brazil": 3}
        
        housing_factor = {"Apartment (small)": 0.6, "Apartment (large)": 0.8, 
                         "House (small)": 1.0, "House (medium)": 1.3, "House (large)": 1.6}
        
        car_factor = {"No car": 0, "Occasional": 1.5, "Regular": 3, "Daily commute": 5, "Heavy use": 8}
        
        diet_factor = {"Vegan": 1.5, "Vegetarian": 2, "Low meat": 2.5, "Average": 3, "High meat": 4}
        
        flight_emissions = flights_year * 0.5  # tons per flight
        
        base = country_base.get(country, 10)
        housing_co2 = base * 0.3 * housing_factor.get(housing, 1.0) / household_size
        transport_co2 = car_factor.get(car_usage, 3) + flight_emissions
        food_co2 = diet_factor.get(diet, 3)
        other_co2 = base * 0.2
        
        total_co2 = housing_co2 + transport_co2 + food_co2 + other_co2
        
        st.success("📊 Your Estimated Annual Carbon Footprint")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("🌡️ Total Footprint", f"{total_co2:.1f} tons CO₂/year")
            
            world_avg = 4.5
            country_avg = country_base.get(country, 10)
            
            st.markdown("**Comparison:**")
            st.write(f"🌍 World average: {world_avg} tons")
            st.write(f"🏳️ {country} average: {country_avg} tons")
            
            if total_co2 < world_avg:
                st.success(f"✅ {((world_avg - total_co2) / world_avg * 100):.0f}% below world average!")
            else:
                st.warning(f"⚠️ {((total_co2 - world_avg) / world_avg * 100):.0f}% above world average")
        
        with col2:
            breakdown = {
                "Category": ["Housing", "Transport", "Food", "Other"],
                "CO₂ (tons)": [housing_co2, transport_co2, food_co2, other_co2]
            }
            df = pd.DataFrame(breakdown)
            st.bar_chart(df.set_index("Category"))
        
        # Trees to offset
        trees_needed = int(total_co2 * 1000 / 21)
        st.markdown(f"🌳 **{trees_needed} trees** would be needed to offset your annual emissions")
        
        # Reduction suggestions
        st.markdown("#### 💡 Top Ways to Reduce Your Footprint")
        
        suggestions = []
        if car_usage in ["Daily commute", "Heavy use"]:
            suggestions.append("🚗 Switch to EV or use public transit 2 days/week → Save ~2 tons/year")
        if diet in ["Average", "High meat"]:
            suggestions.append("🥗 Reduce meat to 2x/week → Save ~1 ton/year")
        if flights_year > 2:
            suggestions.append("✈️ Take one less flight per year → Save ~0.5 tons/year")
        if housing in ["House (medium)", "House (large)"]:
            suggestions.append("🏠 Install solar panels → Save ~3 tons/year")
        
        if not suggestions:
            suggestions.append("🌟 Great job! You're already making sustainable choices!")
        
        for suggestion in suggestions[:4]:
            st.write(suggestion)


def render_eco_tips():
    """Display eco-friendly tips and recommendations."""
    st.markdown("### 💡 Eco-Friendly Tips & Quick Wins")
    
    tip_category = st.selectbox(
        "Select category",
        ["🏠 Home & Energy", "🚗 Transportation", "🍽️ Food & Diet", "🛍️ Shopping & Consumption", "💧 Water Conservation", "🗑️ Waste Reduction"],
        key="tip_category"
    )
    
    if tip_category == "🏠 Home & Energy":
        st.markdown("#### Top Energy-Saving Tips")
        tips = [
            ("💡 Switch to LED bulbs", "Save 75% on lighting costs, ~$100/year", "Easy"),
            ("🌡️ Install a smart thermostat", "Save 10-15% on heating/cooling, ~$200/year", "Easy"),
            ("🔌 Unplug devices when not in use", "Save 5-10% on electricity, ~$50/year", "Easy"),
            ("🏠 Add weatherstripping to doors/windows", "Save 10-20% on heating, ~$150/year", "Medium"),
            ("☀️ Install solar panels", "Save 50-100% on electricity, ~$1,500/year", "Advanced"),
            ("❄️ Upgrade to Energy Star appliances", "Save 10-50% per appliance", "Medium"),
            ("🌬️ Use ceiling fans instead of AC", "Save up to 40% on cooling costs", "Easy"),
            ("🧺 Wash clothes in cold water", "Save 90% of washing machine energy", "Easy")
        ]
        
    elif tip_category == "🚗 Transportation":
        tips = [
            ("🚶 Walk or bike for trips under 3 km", "Save ~$500/year + health benefits", "Easy"),
            ("🚌 Use public transit", "Save ~$3,000/year vs driving", "Medium"),
            ("🚗 Carpool to work", "Save ~$1,500/year on fuel", "Easy"),
            ("⚡ Switch to an electric vehicle", "Save ~$1,500/year on fuel + maintenance", "Advanced"),
            ("🏠 Work from home when possible", "Save ~$2,000/year + 3 tons CO₂", "Easy"),
            ("🚲 Get an e-bike for commuting", "Save ~$2,000/year vs car", "Medium"),
            ("🛞 Keep tires properly inflated", "Improve fuel efficiency by 3%", "Easy"),
            ("📍 Combine errands into one trip", "Reduce fuel use by 20%+", "Easy")
        ]
        
    elif tip_category == "🍽️ Food & Diet":
        tips = [
            ("🥗 Have one meatless day per week", "Save 0.5 tons CO₂/year", "Easy"),
            ("🛒 Buy local and seasonal produce", "Reduce food miles by 90%", "Easy"),
            ("🍱 Meal prep to reduce food waste", "Save $1,500/year on food", "Medium"),
            ("🌱 Start a small herb garden", "Save $200/year + zero transport emissions", "Easy"),
            ("🥡 Bring reusable containers for leftovers", "Reduce packaging waste by 50%", "Easy"),
            ("🧊 Organize your fridge properly", "Reduce food waste by 20%", "Easy"),
            ("☕ Use a reusable coffee cup", "Save 365 disposable cups/year", "Easy"),
            ("🍎 Choose imperfect produce", "Reduce food waste + save 30%", "Easy")
        ]
        
    elif tip_category == "🛍️ Shopping & Consumption":
        tips = [
            ("🛍️ Bring reusable shopping bags", "Save 500+ plastic bags/year", "Easy"),
            ("👕 Buy second-hand clothing", "Save 90% of clothing's carbon footprint", "Easy"),
            ("📱 Buy refurbished electronics", "Save 80% of manufacturing emissions", "Medium"),
            ("📦 Consolidate online orders", "Reduce packaging waste by 50%", "Easy"),
            ("🔧 Repair instead of replace", "Extend product life, save money", "Medium"),
            ("📖 Borrow or rent rarely-used items", "Reduce consumption significantly", "Easy"),
            ("🏷️ Choose products with minimal packaging", "Reduce waste by 30%", "Easy"),
            ("♻️ Buy products with recycled content", "Support circular economy", "Easy")
        ]
        
    elif tip_category == "💧 Water Conservation":
        tips = [
            ("🚿 Take shorter showers (5 mins)", "Save 40+ liters per shower", "Easy"),
            ("🚽 Install dual-flush toilet", "Save 15,000 liters/year", "Medium"),
            ("🚰 Fix leaky faucets", "Save 10,000+ liters/year", "Easy"),
            ("🌧️ Install rain barrel for garden", "Save 5,000+ liters/year", "Medium"),
            ("🧹 Sweep instead of hosing driveways", "Save 300+ liters each time", "Easy"),
            ("🧽 Only run full loads (dishwasher/washer)", "Save 3,000+ liters/year", "Easy"),
            ("🌱 Use drought-resistant plants", "Reduce garden water by 50%", "Medium"),
            ("💧 Install low-flow showerheads", "Save 10,000+ liters/year", "Easy")
        ]
        
    else:  # Waste Reduction
        tips = [
            ("♻️ Learn what can be recycled locally", "Increase recycling rate by 50%", "Easy"),
            ("🍌 Compost food scraps", "Reduce waste by 30% + create fertilizer", "Medium"),
            ("📧 Go paperless for bills", "Save 40+ pounds of paper/year", "Easy"),
            ("🥤 Use reusable water bottle", "Save 150+ plastic bottles/year", "Easy"),
            ("🍽️ Use cloth napkins", "Save 2,000+ paper napkins/year", "Easy"),
            ("📦 Refuse unnecessary receipts", "Reduce paper waste significantly", "Easy"),
            ("🔋 Recycle batteries and e-waste properly", "Prevent toxic chemicals in landfills", "Easy"),
            ("🛒 Buy in bulk to reduce packaging", "Reduce packaging waste by 40%", "Medium")
        ]
    
    # Display tips with difficulty badges
    for tip, impact, difficulty in tips:
        difficulty_color = {"Easy": "🟢", "Medium": "🟡", "Advanced": "🔴"}
        with st.expander(f"{tip}"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**Impact:** {impact}")
            with col2:
                st.write(f"{difficulty_color[difficulty]} {difficulty}")
    
    # Weekly challenge
    st.markdown("---")
    st.markdown("### 🏆 Weekly Eco Challenge")
    
    import random
    challenges = [
        "🚶 Walk or bike for all trips under 2 km this week",
        "🥗 Try 3 meatless dinners this week",
        "🔌 Unplug all devices when not in use",
        "🚿 Take 5-minute showers only",
        "🛍️ Refuse all single-use plastics",
        "🌱 Start composting your food scraps",
        "💡 Use only natural lighting during the day",
        "📦 Make zero online orders this week"
    ]
    
    if 'weekly_challenge' not in st.session_state:
        st.session_state.weekly_challenge = random.choice(challenges)
    
    st.info(f"**This Week's Challenge:** {st.session_state.weekly_challenge}")
    
    if st.button("🔄 Get New Challenge"):
        st.session_state.weekly_challenge = random.choice(challenges)
        st.rerun()


if __name__ == "__main__":
    main()
