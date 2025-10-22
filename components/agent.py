"""
Main CO2 Reduction Agent orchestrating the RAG workflow.

This module implements the core agent that processes queries, analyzes datasets,
and generates recommendations by coordinating all components.
"""

import pandas as pd
from typing import List, Optional, Dict, Any
from models.data_models import (
    Activity, AgentResponse, DatasetAnalysis, Recommendation, Category
)
from components.query_processor import QueryProcessor, QueryIntent
from components.dataset_analyzer import DatasetAnalyzer
from components.recommendation_generator import RecommendationGenerator
from components.llm_client import LLMClient
from components.vector_store import VectorStore, Document
from components.reference_data import ReferenceDataManager
from components.prompt_templates import PromptTemplates
from components.response_parser import ResponseParser, parse_llm_response


class CO2ReductionAgent:
    """
    Main agent class that orchestrates the CO2 reduction recommendation system.
    
    Combines query processing, vector retrieval, LLM generation, and
    recommendation ranking to provide comprehensive CO2 reduction advice.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        vector_store: VectorStore,
        reference_data_manager: ReferenceDataManager
    ):
        """
        Initialize the CO2 Reduction Agent.
        
        Args:
            llm_client: LLM client for text generation
            vector_store: Vector store for knowledge retrieval
            reference_data_manager: Reference data manager for activity lookup
        """
        self.llm = llm_client
        self.vector_store = vector_store
        self.reference_manager = reference_data_manager
        
        # Initialize sub-components
        self.query_processor = QueryProcessor()
        self.dataset_analyzer = DatasetAnalyzer()
        self.recommendation_generator = RecommendationGenerator(reference_data_manager)
        self.prompt_templates = PromptTemplates()
        self.response_parser = ResponseParser()
        
        # Ensure reference data is loaded
        if self.reference_manager.data is None:
            self.reference_manager.load_reference_data()
    
    def process_query(self, query: str) -> AgentResponse:
        """
        Process a text query end-to-end and generate recommendations.
        
        Args:
            query: User's natural language query
            
        Returns:
            AgentResponse with recommendations and analysis
            
        Raises:
            ValueError: If query is empty or invalid
            RuntimeError: If LLM generation fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Step 1: Process the query to extract intent and activities
        query_info = self.query_processor.process_query(query)
        intent = query_info['intent']
        activities = query_info['activities']
        parameters = query_info['parameters']
        
        # Step 2: Handle based on intent
        if intent == QueryIntent.SINGLE_ACTIVITY:
            return self._handle_single_activity_query(query, activities, parameters)
        
        elif intent == QueryIntent.COMPARISON:
            return self._handle_comparison_query(query, activities)
        
        elif intent == QueryIntent.GENERAL_ADVICE:
            return self._handle_general_advice_query(query)
        
        else:
            # Unknown intent - treat as general advice
            return self._handle_general_advice_query(query)
    
    def analyze_dataset(self, df: pd.DataFrame) -> DatasetAnalysis:
        """
        Process uploaded CSV/Excel files and generate analysis.
        
        Args:
            df: DataFrame with activity data
            
        Returns:
            DatasetAnalysis with complete analysis and recommendations
            
        Raises:
            ValueError: If dataset validation fails
        """
        # Step 1: Analyze the dataset
        analysis = self.dataset_analyzer.analyze_dataset(df)
        
        # Step 2: Generate recommendations for top emitters
        recommendations = self.recommendation_generator.generate_recommendations_for_multiple(
            analysis.top_emitters,
            max_per_activity=2
        )
        
        # Step 3: Enhance recommendations with LLM insights
        if recommendations:
            # Get top 3 recommendations
            top_recommendations = recommendations[:3]
            
            # Optionally enhance with LLM-generated insights
            try:
                enhanced_recs = self._enhance_recommendations_with_llm(
                    analysis.top_emitters,
                    top_recommendations
                )
                if enhanced_recs:
                    recommendations = enhanced_recs + recommendations[3:]
            except Exception:
                # If LLM enhancement fails, use original recommendations
                pass
        
        # Update analysis with recommendations
        analysis.recommendations = recommendations
        
        return analysis
    
    def generate_recommendations(
        self,
        activity: Activity,
        use_llm: bool = True
    ) -> List[Recommendation]:
        """
        Generate recommendations combining retrieval and generation.
        
        Args:
            activity: Activity to generate recommendations for
            use_llm: Whether to use LLM for enhanced recommendations
            
        Returns:
            List of Recommendation objects
        """
        # Step 1: Generate base recommendations from reference data
        base_recommendations = self.recommendation_generator.generate_recommendations_for_activity(
            activity,
            max_recommendations=5
        )
        
        if not use_llm or not base_recommendations:
            return base_recommendations
        
        # Step 2: Retrieve relevant knowledge from vector store
        search_query = f"reduce {activity.category.value.lower()} emissions {activity.name}"
        retrieved_docs = self.vector_store.search(search_query, k=3)
        
        # Step 3: Use LLM to enhance recommendations
        try:
            alternatives = [rec.action for rec in base_recommendations]
            prompt = self.prompt_templates.recommendation_prompt(
                activity=activity.name,
                emission=activity.emission_kg_per_day,
                category=activity.category.value,
                alternatives=alternatives
            )
            
            # Add retrieved context
            if retrieved_docs:
                context_text = "\n\n".join([doc.content for doc in retrieved_docs])
                prompt = f"{prompt}\n\nAdditional Context:\n{context_text}"
            
            llm_response = self.llm.generate(prompt, max_tokens=800)
            
            # Parse LLM response
            llm_recommendations = parse_llm_response(llm_response)
            
            # Merge with base recommendations (prefer LLM if available)
            if llm_recommendations:
                return llm_recommendations
            
        except Exception:
            # If LLM fails, return base recommendations
            pass
        
        return base_recommendations
    
    def _handle_single_activity_query(
        self,
        query: str,
        activities: List[str],
        parameters: Dict[str, Any]
    ) -> AgentResponse:
        """
        Handle queries about a single activity.
        
        Args:
            query: Original query
            activities: Extracted activity names
            parameters: Extracted parameters
            
        Returns:
            AgentResponse with recommendations
        """
        # Find the activity in reference data
        activity_name = activities[0] if activities else "unknown activity"
        
        # Search for similar activities in reference data
        similar_activities = self.reference_manager.search_similar_activities(
            activity_name,
            n=1
        )
        
        if not similar_activities:
            # If no match found, return general advice
            return self._handle_general_advice_query(query)
        
        current_activity = similar_activities[0]
        
        # Adjust emission based on parameters if provided
        if 'distance_km' in parameters:
            # Rough adjustment for distance (this is simplified)
            distance = parameters['distance_km']
            # Assume reference is for average distance, scale accordingly
            current_activity.emission_kg_per_day *= (distance / 10.0)  # Assume 10km baseline
        
        # Generate recommendations
        recommendations = self.generate_recommendations(current_activity, use_llm=True)
        
        # Calculate total potential reduction
        total_reduction = sum(rec.emission_reduction_kg for rec in recommendations)
        annual_savings = total_reduction * 365
        
        # Generate summary
        summary = self._generate_summary(
            current_activity,
            recommendations,
            query_type="single_activity"
        )
        
        return AgentResponse(
            current_emission=current_activity.emission_kg_per_day,
            recommendations=recommendations[:5],  # Top 5 recommendations
            total_potential_reduction=total_reduction,
            annual_savings_kg=annual_savings,
            summary=summary
        )
    
    def _handle_comparison_query(
        self,
        query: str,
        activities: List[str]
    ) -> AgentResponse:
        """
        Handle queries comparing multiple activities.
        
        Args:
            query: Original query
            activities: Extracted activity names
            
        Returns:
            AgentResponse with comparison
        """
        # Find activities in reference data
        found_activities = []
        for activity_name in activities[:3]:  # Limit to 3 activities
            similar = self.reference_manager.search_similar_activities(activity_name, n=1)
            if similar:
                found_activities.append(similar[0])
        
        if len(found_activities) < 2:
            # Not enough activities for comparison
            return self._handle_general_advice_query(query)
        
        # Sort by emission (highest first)
        found_activities.sort(key=lambda x: x.emission_kg_per_day, reverse=True)
        
        current_activity = found_activities[0]
        alternatives = found_activities[1:]
        
        # Generate comparison using LLM
        try:
            prompt = self.prompt_templates.comparison_prompt(
                current=current_activity,
                alternatives=alternatives
            )
            
            llm_response = self.llm.generate(prompt, max_tokens=600)
            
            # Create recommendations from alternatives
            recommendations = self.recommendation_generator.rank_recommendations(
                current_activity,
                alternatives
            )
            
            total_reduction = sum(rec.emission_reduction_kg for rec in recommendations)
            annual_savings = total_reduction * 365
            
            return AgentResponse(
                current_emission=current_activity.emission_kg_per_day,
                recommendations=recommendations,
                total_potential_reduction=total_reduction,
                annual_savings_kg=annual_savings,
                summary=llm_response
            )
            
        except Exception:
            # Fallback to basic comparison
            recommendations = self.recommendation_generator.rank_recommendations(
                current_activity,
                alternatives
            )
            
            total_reduction = sum(rec.emission_reduction_kg for rec in recommendations)
            annual_savings = total_reduction * 365
            
            summary = f"Comparing {current_activity.name} with alternatives. "
            summary += f"Potential reduction: {total_reduction:.2f} kg CO2/day."
            
            return AgentResponse(
                current_emission=current_activity.emission_kg_per_day,
                recommendations=recommendations,
                total_potential_reduction=total_reduction,
                annual_savings_kg=annual_savings,
                summary=summary
            )
    
    def _handle_general_advice_query(self, query: str) -> AgentResponse:
        """
        Handle general advice queries.
        
        Args:
            query: Original query
            
        Returns:
            AgentResponse with general advice
        """
        # Retrieve relevant knowledge from vector store
        retrieved_docs = self.vector_store.search(query, k=5)
        
        context = [doc.content for doc in retrieved_docs] if retrieved_docs else []
        
        # Generate response using LLM with context
        try:
            if context:
                llm_response = self.llm.generate_with_context(
                    query=query,
                    context=context,
                    max_tokens=600
                )
            else:
                prompt = self.prompt_templates.general_advice_prompt(query, [])
                llm_response = self.llm.generate(prompt, max_tokens=600)
            
            # Try to extract any recommendations from the response
            recommendations = parse_llm_response(llm_response)
            
            # If no structured recommendations, create generic ones
            if not recommendations:
                recommendations = self._create_generic_recommendations()
            
            total_reduction = sum(rec.emission_reduction_kg for rec in recommendations)
            annual_savings = total_reduction * 365
            
            return AgentResponse(
                current_emission=0.0,  # Unknown for general advice
                recommendations=recommendations[:5],
                total_potential_reduction=total_reduction,
                annual_savings_kg=annual_savings,
                summary=llm_response
            )
            
        except Exception as e:
            # Fallback response
            recommendations = self._create_generic_recommendations()
            total_reduction = sum(rec.emission_reduction_kg for rec in recommendations)
            annual_savings = total_reduction * 365
            
            return AgentResponse(
                current_emission=0.0,
                recommendations=recommendations,
                total_potential_reduction=total_reduction,
                annual_savings_kg=annual_savings,
                summary="Here are some general recommendations for reducing your carbon footprint."
            )
    
    def _enhance_recommendations_with_llm(
        self,
        activities: List[Activity],
        base_recommendations: List[Recommendation]
    ) -> Optional[List[Recommendation]]:
        """
        Enhance recommendations using LLM analysis.
        
        Args:
            activities: List of activities to analyze
            base_recommendations: Base recommendations to enhance
            
        Returns:
            Enhanced recommendations or None if enhancement fails
        """
        try:
            prompt = self.prompt_templates.analysis_prompt(activities)
            llm_response = self.llm.generate(prompt, max_tokens=800)
            
            # Parse enhanced recommendations
            enhanced_recs = parse_llm_response(llm_response)
            
            if enhanced_recs:
                return enhanced_recs
            
        except Exception:
            pass
        
        return None
    
    def _generate_summary(
        self,
        activity: Activity,
        recommendations: List[Recommendation],
        query_type: str = "single_activity"
    ) -> str:
        """
        Generate a summary of the analysis.
        
        Args:
            activity: Current activity
            recommendations: Generated recommendations
            query_type: Type of query
            
        Returns:
            Summary text
        """
        if not recommendations:
            return f"Currently: {activity.name} produces {activity.emission_kg_per_day:.2f} kg CO2/day. No specific recommendations available."
        
        best_rec = recommendations[0]
        
        summary = f"Your current activity '{activity.name}' produces {activity.emission_kg_per_day:.2f} kg CO2/day "
        summary += f"({activity.emission_kg_per_day * 365:.1f} kg/year). "
        summary += f"Our top recommendation: {best_rec.action}. "
        summary += f"This could reduce emissions by {best_rec.emission_reduction_kg:.2f} kg/day "
        summary += f"({best_rec.reduction_percentage:.1f}%), "
        summary += f"saving {best_rec.emission_reduction_kg * 365:.1f} kg CO2 annually."
        
        return summary
    
    def _create_generic_recommendations(self) -> List[Recommendation]:
        """
        Create generic recommendations when specific ones aren't available.
        
        Returns:
            List of generic Recommendation objects
        """
        generic_recs = [
            Recommendation(
                action="Use public transportation or carpool instead of driving alone",
                emission_reduction_kg=2.5,
                reduction_percentage=30.0,
                implementation_difficulty="Easy",
                timeframe="Immediate",
                additional_benefits=["Cost savings", "Reduced traffic stress"]
            ),
            Recommendation(
                action="Reduce meat consumption and eat more plant-based meals",
                emission_reduction_kg=1.8,
                reduction_percentage=25.0,
                implementation_difficulty="Easy",
                timeframe="Immediate",
                additional_benefits=["Health benefits", "Lower food costs"]
            ),
            Recommendation(
                action="Switch to energy-efficient LED lighting",
                emission_reduction_kg=0.5,
                reduction_percentage=15.0,
                implementation_difficulty="Easy",
                timeframe="Short-term",
                additional_benefits=["Lower energy bills", "Longer lifespan"]
            )
        ]
        
        return generic_recs
    
    def check_system_health(self) -> Dict[str, Any]:
        """
        Check the health status of all system components.
        
        Returns:
            Dictionary with health status of each component
        """
        health = {
            "llm_available": False,
            "vector_store_ready": False,
            "reference_data_loaded": False,
            "errors": []
        }
        
        # Check LLM
        try:
            health["llm_available"] = self.llm.check_availability()
        except Exception as e:
            health["errors"].append(f"LLM check failed: {str(e)}")
        
        # Check vector store
        try:
            stats = self.vector_store.get_collection_stats()
            health["vector_store_ready"] = stats["document_count"] > 0
            health["vector_store_docs"] = stats["document_count"]
        except Exception as e:
            health["errors"].append(f"Vector store check failed: {str(e)}")
        
        # Check reference data
        try:
            health["reference_data_loaded"] = self.reference_manager.data is not None
            if health["reference_data_loaded"]:
                health["reference_data_count"] = len(self.reference_manager.data)
        except Exception as e:
            health["errors"].append(f"Reference data check failed: {str(e)}")
        
        health["overall_healthy"] = (
            health["llm_available"] and
            health["vector_store_ready"] and
            health["reference_data_loaded"]
        )
        
        return health
