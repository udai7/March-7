"""
CO₂ Reduction AI Agent - Streamlit Application

Main entry point for the Streamlit web interface that helps users
identify high CO₂ emission activities and generate actionable recommendations.
"""

import streamlit as st
import pandas as pd
from typing import Optional, List
import config
from components.agent import CO2ReductionAgent
from components.llm_client import LLMClient
from components.vector_store import VectorStore
from components.reference_data import ReferenceDataManager
from models.data_models import AgentResponse, DatasetAnalysis


# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache(allow_output_mutation=True)
def initialize_agent() -> Optional[CO2ReductionAgent]:
    """
    Initialize and cache the CO2 Reduction Agent with all components.
    
    Returns:
        Initialized CO2ReductionAgent or None if initialization fails
    """
    try:
        # Initialize LLM client
        llm_client = LLMClient(
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
        
    except Exception as e:
        st.error(f"Failed to initialize agent: {str(e)}")
        return None


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
    st.subheader("Ask About CO₂ Reduction")
    st.markdown("Enter your question about reducing carbon emissions from your daily activities.")
    
    # Example queries in an expander
    with st.expander("💡 Example Questions"):
        st.markdown("""
        - I drive 20 km daily using a petrol car. How can I reduce emissions?
        - What's the CO₂ impact of eating beef vs chicken?
        - Compare emissions from driving vs taking the bus
        - How can I reduce my household energy consumption?
        - What are the best ways to reduce my carbon footprint?
        - I use air conditioning 8 hours daily. What are alternatives?
        - Compare flying vs train travel for long distances
        - How much CO₂ does online shopping produce?
        """)
    
    # Query input with form for better UX
    with st.form(key="query_form", clear_on_submit=False):
        user_query = st.text_area(
            "Your Question:",
            value=st.session_state.last_query,
            height=100,
            placeholder="e.g., I drive 30 km daily in a petrol car. How can I reduce my emissions?",
            help="Describe your activity or ask a general question about CO₂ reduction"
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
    Display query results with recommendations and visualizations.
    
    Args:
        response: AgentResponse object with recommendations
    """
    st.success("✅ Analysis Complete!")
    
    # Display summary
    st.markdown("### 📝 Summary")
    st.info(response.summary)
    
    # Display current emission and potential savings
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
            "Annual Savings",
            f"{response.annual_savings_kg:.1f} kg CO₂",
            delta=f"-{response.annual_savings_kg:.1f} kg/year",
            delta_color="inverse",
            help="Projected annual CO₂ savings"
        )
    
    # Display recommendations
    if response.recommendations:
        st.markdown("### 💡 Recommendations")
        
        for idx, rec in enumerate(response.recommendations, 1):
            with st.expander(f"**{idx}. {rec.action}**", expanded=(idx == 1)):
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
        
        # Emission comparison chart
        if len(response.recommendations) > 0:
            st.markdown("### 📊 Emission Comparison")
            
            # Prepare data for chart
            chart_data = {
                "Option": ["Current"],
                "CO₂ Emission (kg/day)": [response.current_emission]
            }
            
            for idx, rec in enumerate(response.recommendations[:5], 1):
                alternative_emission = response.current_emission - rec.emission_reduction_kg
                chart_data["Option"].append(f"Alt {idx}")
                chart_data["CO₂ Emission (kg/day)"].append(alternative_emission)
            
            chart_df = pd.DataFrame(chart_data)
            st.bar_chart(chart_df.set_index("Option"))
    else:
        st.warning("No specific recommendations available for this query.")


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


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Initialize agent
    agent = initialize_agent()
    
    # Header
    st.title("🌱 CO₂ Reduction AI Agent")
    st.markdown(
        "Get personalized recommendations to reduce your carbon footprint. "
        "Ask questions about your activities or upload your data for analysis."
    )
    
    # Check system health
    if agent:
        with st.sidebar:
            st.header("System Status")
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
    else:
        st.error("❌ Failed to initialize the agent. Please check the configuration.")
        st.stop()
    
    # Main content area
    st.markdown("---")
    
    # Create tabs for different interaction modes
    tab1, tab2 = st.tabs(["💬 Ask a Question", "📊 Upload Dataset"])
    
    with tab1:
        render_query_interface(agent)
    
    with tab2:
        render_file_upload_interface(agent)


if __name__ == "__main__":
    main()
