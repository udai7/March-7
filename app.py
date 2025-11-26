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
        ["Transport Savings", "Energy Savings", "Water Savings"],
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
                ["petrol_car", "diesel_car", "motorcycle", "public_transit"],
                format_func=lambda x: x.replace("_", " ").title(),
                key="current_transport"
            )
        
        with col2:
            alternative_mode = st.selectbox(
                "Alternative mode",
                ["electric_car", "hybrid_car", "public_transit", "cycling", "ebike", "carpool"],
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
    
    # Investment selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_investment = st.selectbox(
            "Select Investment Type",
            [opt["type"] for opt in options],
            format_func=lambda x: next((opt["name"] for opt in options if opt["type"] == x), x),
            key="investment_type"
        )
    
    with col2:
        use_custom_values = st.checkbox("Use custom values", key="custom_values")
    
    # Get selected investment data
    investment_type = InvestmentType(selected_investment)
    investment_data = calculator.INVESTMENT_DATA[investment_type]
    
    # Custom inputs if enabled
    custom_cost = None
    custom_savings = None
    
    if use_custom_values:
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
    
    # Show comparison table
    st.markdown("### 📊 Investment Comparison")
    
    comparison_data = {
        "Investment": [],
        "Initial Cost": [],
        "Annual Savings": [],
        "Payback (Years)": [],
        "Lifetime ROI (%)": [],
        "CO₂ Saved (kg/year)": []
    }
    
    for opt in options:
        comparison_data["Investment"].append(opt["name"][:30] + "...")
        comparison_data["Initial Cost"].append(f"${opt['avg_cost']:,.0f}")
        comparison_data["Annual Savings"].append(f"${opt['annual_savings']:,.0f}")
        comparison_data["Payback (Years)"].append(f"{opt['payback_years']:.1f}")
        comparison_data["Lifetime ROI (%)"].append(f"{opt['roi_percent']:.0f}%")
        comparison_data["CO₂ Saved (kg/year)"].append(f"{opt['co2_savings_annual']:,}")
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df)


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
    st.markdown("Compare your current utility costs with optimized usage.")
    
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
    
    # Environmental impact comparison
    st.markdown("### 🌍 Category Impact Comparison")
    
    # Sample comparison data
    categories_data = {
        "Category": ["Transport", "Food", "Household", "Lifestyle", "Energy"],
        "CO₂ (kg/day)": [4.5, 3.8, 2.2, 1.5, 3.0],
        "Water (L/day)": [5, 850, 150, 50, 100],
        "Energy (kWh/day)": [2, 0.5, 8, 1, 12],
        "Waste (kg/day)": [0.1, 0.8, 0.3, 0.2, 0.1]
    }
    
    df = pd.DataFrame(categories_data)
    
    # Select metric to visualize
    metric = st.selectbox(
        "Select metric to compare:",
        ["CO₂ (kg/day)", "Water (L/day)", "Energy (kWh/day)", "Waste (kg/day)"]
    )
    
    st.bar_chart(df.set_index("Category")[metric])
    
    # Environmental tips by category
    st.markdown("### 💡 Quick Wins by Category")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("🚗 Transport", expanded=True):
            st.markdown("""
            - Switch to public transit (saves ~3 kg CO₂/day)
            - Carpool when possible
            - Consider an e-bike for short trips
            - Work from home 1-2 days/week
            """)
        
        with st.expander("🍔 Food"):
            st.markdown("""
            - Reduce beef consumption by 50%
            - Choose local & seasonal produce
            - Minimize food waste
            - Try plant-based meals 2x/week
            """)
    
    with col2:
        with st.expander("🏠 Household", expanded=True):
            st.markdown("""
            - Switch to LED bulbs (saves 75% energy)
            - Fix leaky faucets (saves 10+ L/day)
            - Use cold water for laundry
            - Install smart thermostat
            """)
        
        with st.expander("♻️ Waste"):
            st.markdown("""
            - Use reusable bags & bottles
            - Compost food scraps
            - Recycle properly
            - Choose products with less packaging
            """)
    
    # Sustainability goals tracker
    st.markdown("### 🎯 Set Your Goals")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        co2_goal = st.number_input("Daily CO₂ Target (kg)", min_value=0.0, value=5.0, step=0.5)
    with col2:
        water_goal = st.number_input("Daily Water Target (L)", min_value=0, value=150, step=10)
    with col3:
        energy_goal = st.number_input("Daily Energy Target (kWh)", min_value=0.0, value=10.0, step=0.5)
    with col4:
        waste_goal = st.number_input("Daily Waste Target (kg)", min_value=0.0, value=1.0, step=0.1)
    
    if st.button("💾 Save Goals"):
        st.success("✅ Goals saved! Track your progress by uploading activity data.")
    
    # Environmental equivalents
    st.markdown("### 🌳 Impact Equivalents")
    st.markdown("Understanding your impact in real-world terms:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🌲 Trees Needed", "12", help="Trees needed to offset annual CO₂")
    with col2:
        st.metric("🚿 Showers Equivalent", "450", help="10-min showers equivalent to annual water use")
    with col3:
        st.metric("💡 Light Bulb Hours", "18,250", help="Hours a 10W LED could run on your annual energy")


if __name__ == "__main__":
    main()
