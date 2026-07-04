"""
Main Environmental Impact Agent orchestrating the RAG workflow.

This module implements the core agent that processes queries, analyzes datasets,
and generates recommendations by coordinating all components for comprehensive
environmental impact analysis (CO2, water, energy, waste, and more).
"""

import pandas as pd
from typing import List, Optional, Dict, Any
import config
from models.data_models import (
    Activity, AgentResponse, DatasetAnalysis, Recommendation, Category, EnvironmentalMetrics
)
from components.query_processor import QueryProcessor, QueryIntent
from components.dataset_analyzer import DatasetAnalyzer
from components.recommendation_generator import RecommendationGenerator
from components.llm_client import LLMClient
from components.vector_store import VectorStore, Document
from components.reference_data import ReferenceDataManager
from components.prompt_templates import PromptTemplates
from components.response_parser import ResponseParser, parse_llm_response
from components.environmental_scorer import EnvironmentalScorer


class CO2ReductionAgent:
    """
    Main agent class that orchestrates the environmental impact recommendation system.
    
    Combines query processing, vector retrieval, LLM generation, and
    recommendation ranking to provide comprehensive environmental impact advice
    covering CO2, water, energy, waste, and pollution metrics.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        vector_store: VectorStore,
        reference_data_manager: ReferenceDataManager
    ):
        """
        Initialize the Environmental Impact Agent.
        
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
        self.environmental_scorer = EnvironmentalScorer()
        
        # Ensure reference data is loaded
        if self.reference_manager.data is None:
            self.reference_manager.load_reference_data()
    
    def process_query(self, query: str) -> AgentResponse:
        """
        Process a text query end-to-end and generate recommendations.
        
        Args:
            query: User's natural language query
            
        Returns:
            AgentResponse with recommendations and comprehensive environmental analysis
            
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
        Process uploaded CSV/Excel files and generate analysis with LLM fallback.
        
        Args:
            df: DataFrame with activity data
            
        Returns:
            DatasetAnalysis with complete analysis and recommendations
            
        Raises:
            ValueError: If dataset validation fails
        """
        # Step 1: Analyze the dataset
        analysis = self.dataset_analyzer.analyze_dataset(df)
        
        # Step 2: Check if we have unknown/unusual categories that need LLM help
        has_unknown_categories = any(
            activity.category == Category.LIFESTYLE and 
            any(keyword in activity.name.lower() for keyword in ['energy', 'electric', 'power', 'charge'])
            for activity in analysis.top_emitters
        )
        
        # Step 3: Generate recommendations
        if has_unknown_categories or len(analysis.top_emitters) == 0:
            # Use LLM-based recommendations for unknown categories
            recommendations = self._generate_llm_recommendations_for_dataset(analysis)
        else:
            # Use reference data recommendations
            recommendations = self.recommendation_generator.generate_recommendations_for_multiple(
                analysis.top_emitters,
                max_per_activity=2
            )
        
        # Step 4: Enhance recommendations with LLM insights if we have good base recommendations
        if recommendations and not has_unknown_categories:
            try:
                enhanced_recs = self._enhance_recommendations_with_llm(
                    analysis.top_emitters,
                    recommendations[:3]
                )
                if enhanced_recs:
                    recommendations = enhanced_recs + recommendations[3:]
            except Exception:
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
        Handle queries about a single activity using RAG approach.
        
        Args:
            query: Original query
            activities: Extracted activity names
            parameters: Extracted parameters
            
        Returns:
            AgentResponse with recommendations
        """
        # Try multiple search strategies to find the activity
        similar_activities = []
        query_lower = query.lower()
        
        # Strategy 1: Try key transport/household/food keywords from query FIRST (most reliable)
        search_terms = []
        
        # Extract key terms based on query content
        if any(word in query_lower for word in ['drive', 'driving', 'car', 'vehicle']):
            search_terms.append('driving car')
        if any(word in query_lower for word in ['bus']):
            search_terms.append('bus')
        if any(word in query_lower for word in ['train', 'metro', 'subway']):
            search_terms.append('train')
        if any(word in query_lower for word in ['bike', 'bicycle', 'cycling']):
            search_terms.append('cycling')
        if any(word in query_lower for word in ['walk', 'walking']):
            search_terms.append('walking')
        if any(word in query_lower for word in ['flight', 'fly', 'airplane', 'plane']):
            search_terms.append('flying')
        if any(word in query_lower for word in ['electric', 'electricity', 'power']):
            search_terms.append('electricity')
        if any(word in query_lower for word in ['meat', 'beef', 'chicken', 'pork']):
            search_terms.append('meat')
        if any(word in query_lower for word in ['heating', 'heat']):
            search_terms.append('heating')
        if any(word in query_lower for word in ['shower', 'bath', 'water']):
            search_terms.append('water')
        
        for term in search_terms:
            similar = self.reference_manager.search_similar_activities(
                term,
                n=3,
                cutoff=0.3
            )
            if similar:
                similar_activities = similar
                break
        
        # Strategy 2: Try extracted activities (only if Strategy 1 didn't work)
        if not similar_activities and activities:
            for activity_name in activities:
                similar = self.reference_manager.search_similar_activities(
                    activity_name,
                    n=3,
                    cutoff=0.4
                )
                if similar:
                    similar_activities = similar
                    break
        
        # Strategy 3: If still no match, use the full query
        if not similar_activities:
            similar_activities = self.reference_manager.search_similar_activities(
                query,
                n=3,
                cutoff=0.3
            )
        
        # If still no match found, treat as general advice with RAG
        if not similar_activities:
            return self._handle_general_advice_query(query)
        
        current_activity = similar_activities[0]
        
        # Adjust emission based on parameters if provided
        if 'distance_km' in parameters:
            # Rough adjustment for distance (this is simplified)
            distance = parameters['distance_km']
            # Assume reference is for average distance, scale accordingly
            current_activity.emission_kg_per_day *= (distance / 10.0)  # Assume 10km baseline
        
        # Retrieve relevant context from vector store
        retrieved_docs = self.vector_store.search(query, k=3)
        context = [doc.content for doc in retrieved_docs] if retrieved_docs else []
        
        # Generate recommendations using LLM with context (RAG approach)
        try:
            # Build prompt with activity info and context
            prompt = f"""You are a CO₂ reduction advisor. A user is asking about: "{query}"

Current Activity: {current_activity.name}
Current CO₂ Emission: {current_activity.emission_kg_per_day} kg/day
Category: {current_activity.category.value}

Based on the following sustainability knowledge, provide 3-5 specific, actionable recommendations to reduce their carbon footprint:

"""
            if context:
                prompt += "\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(context)])
            
            prompt += "\n\nProvide recommendations in this format:\n1. [Action] - [Brief explanation]\n2. [Action] - [Brief explanation]\n..."
            
            llm_response = self.llm.generate(prompt, max_tokens=500, temperature=0.4)
            
            # Parse recommendations
            recommendations = self._parse_llm_recommendations(llm_response)
            
            # If parsing failed, fall back to reference data recommendations
            if not recommendations:
                recommendations = self.generate_recommendations(current_activity, use_llm=False)
            
            total_reduction = sum(rec.emission_reduction_kg for rec in recommendations[:5])
            annual_savings = total_reduction * 365
            
            # Use LLM response as summary
            summary = llm_response
            
        except Exception as e:
            # Fall back to non-LLM approach
            print(f"LLM generation failed: {str(e)}")
            recommendations = self.generate_recommendations(current_activity, use_llm=False)
            total_reduction = sum(rec.emission_reduction_kg for rec in recommendations[:5])
            annual_savings = total_reduction * 365
            summary = self._generate_summary(current_activity, recommendations[:5], query_type="single_activity")
        
        return AgentResponse(
            current_emission=current_activity.emission_kg_per_day,
            recommendations=recommendations[:5],
            total_potential_reduction=total_reduction,
            annual_savings_kg=annual_savings,
            summary=summary,
            # Extended environmental metrics
            current_water_usage=self._get_activity_water_usage(current_activity),
            current_energy_usage=self._get_activity_energy_usage(current_activity),
            current_waste_generation=self._get_activity_waste_generation(current_activity),
            total_water_savings=self._calculate_water_savings(recommendations[:5]),
            total_energy_savings=self._calculate_energy_savings(recommendations[:5]),
            total_waste_reduction=self._calculate_waste_reduction(recommendations[:5]),
            environmental_score=self._calculate_environmental_score(current_activity)
        )
    
    def _get_activity_water_usage(self, activity: Activity) -> float:
        """Get water usage for an activity from reference data."""
        metrics = self.reference_manager.get_activity_environmental_metrics(activity.name)
        return metrics.water_liters_per_day if metrics else 0.0
    
    def _get_activity_energy_usage(self, activity: Activity) -> float:
        """Get energy usage for an activity from reference data."""
        metrics = self.reference_manager.get_activity_environmental_metrics(activity.name)
        return metrics.energy_kwh_per_day if metrics else 0.0
    
    def _get_activity_waste_generation(self, activity: Activity) -> float:
        """Get waste generation for an activity from reference data."""
        metrics = self.reference_manager.get_activity_environmental_metrics(activity.name)
        return metrics.waste_kg_per_day if metrics else 0.0
    
    def _calculate_water_savings(self, recommendations: List[Recommendation]) -> float:
        """Calculate total water savings from recommendations."""
        return sum(rec.water_reduction_liters for rec in recommendations)
    
    def _calculate_energy_savings(self, recommendations: List[Recommendation]) -> float:
        """Calculate total energy savings from recommendations."""
        return sum(rec.energy_reduction_kwh for rec in recommendations)
    
    def _calculate_waste_reduction(self, recommendations: List[Recommendation]) -> float:
        """Calculate total waste reduction from recommendations."""
        return sum(rec.waste_reduction_kg for rec in recommendations)
    
    def _calculate_environmental_score(self, activity: Activity) -> float:
        """Calculate environmental score for an activity."""
        metrics = self.reference_manager.get_activity_environmental_metrics(activity.name)
        if metrics:
            return self.environmental_scorer.calculate_environmental_score(
                co2=metrics.co2_kg_per_day,
                water=metrics.water_liters_per_day,
                energy=metrics.energy_kwh_per_day,
                waste=metrics.waste_kg_per_day,
                pollution_index=metrics.pollution_index
            )
        return 0.0
    
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
        Handle general advice queries using RAG (Retrieval-Augmented Generation).
        
        Args:
            query: Original query
            
        Returns:
            AgentResponse with LLM-generated advice based on retrieved context
        """
        # Step 1: Retrieve relevant knowledge from vector store
        retrieved_docs = self.vector_store.search(query, k=5)
        
        # Step 2: Always use LLM with retrieved context (RAG approach)
        context = [doc.content for doc in retrieved_docs] if retrieved_docs else []
        
        try:
            # Use LLM to generate response with context
            llm_response = self.llm.generate_with_context(
                query=query,
                context=context,
                max_tokens=500,
                temperature=0.4
            )
            
            # Parse recommendations from LLM response
            recommendations = self._parse_llm_recommendations(llm_response)
            
            # If no recommendations parsed, create some based on context
            if not recommendations:
                if context:
                    recommendations = self._create_recommendations_from_context(context, query)
                else:
                    recommendations = self._create_generic_recommendations()
            
            total_reduction = sum(rec.emission_reduction_kg for rec in recommendations)
            annual_savings = total_reduction * 365
            
            return AgentResponse(
                current_emission=0.0,
                recommendations=recommendations[:5],
                total_potential_reduction=total_reduction,
                annual_savings_kg=annual_savings,
                summary=llm_response
            )
            
        except Exception as e:
            # If LLM fails, fall back to context-based recommendations
            print(f"LLM generation failed: {str(e)}")
            
            if context:
                recommendations = self._create_recommendations_from_context(context, query)
                summary = "Based on your question about sustainability, here are relevant recommendations from our knowledge base."
            else:
                recommendations = self._create_generic_recommendations()
                summary = "Here are some general recommendations for reducing your carbon footprint."
            
            total_reduction = sum(rec.emission_reduction_kg for rec in recommendations)
            annual_savings = total_reduction * 365
            
            return AgentResponse(
                current_emission=0.0,
                recommendations=recommendations[:5],
                total_potential_reduction=total_reduction,
                annual_savings_kg=annual_savings,
                summary=summary
            )
    
    def _filter_relevant_docs(self, docs: List[Document], query: str) -> List[Document]:
        """
        Filter documents by relevance score to avoid irrelevant answers.
        
        Args:
            docs: Retrieved documents with similarity scores
            query: Original query
            
        Returns:
            List of relevant documents above threshold
        """
        import config
        
        if not docs:
            return []
        
        # Filter by similarity threshold
        relevant = [doc for doc in docs if hasattr(doc, 'score') and doc.score >= config.RELEVANCE_THRESHOLD]
        
        # If no docs meet threshold, check if any are close
        if not relevant and docs:
            # Use top doc if it's reasonably close (relaxed threshold)
            if hasattr(docs[0], 'score') and docs[0].score >= 0.3:
                relevant = [docs[0]]
        
        return relevant
    
    def _handle_out_of_scope_with_llm(self, query: str) -> AgentResponse:
        """
        Handle queries outside knowledge base scope using LLM as fallback.
        
        Args:
            query: Original query
            
        Returns:
            AgentResponse with LLM-generated advice or out-of-scope message
        """
        try:
            # Use LLM to generate response for out-of-scope queries
            prompt = f"""You are a CO₂ reduction and sustainability advisor. A user asked: "{query}"

Provide helpful, actionable advice to reduce carbon emissions related to their question. 
If the question is about sustainability or environmental impact, give specific recommendations.
If it's completely unrelated to sustainability, politely explain your scope and suggest related topics.

Format your response as:
1. Brief answer to their question (2-3 sentences)
2. 2-3 specific actionable recommendations with estimated CO₂ impact if possible

Keep it concise and practical."""
            
            llm_response = self.llm.generate(prompt, max_tokens=400)
            
            # Parse the LLM response to extract recommendations
            recommendations = self._parse_llm_recommendations(llm_response)
            
            # If we got recommendations, use them
            if recommendations:
                total_reduction = sum(rec.emission_reduction_kg for rec in recommendations)
                annual_savings = total_reduction * 365
                
                return AgentResponse(
                    current_emission=0.0,
                    recommendations=recommendations[:5],
                    total_potential_reduction=total_reduction,
                    annual_savings_kg=annual_savings,
                    summary=llm_response
                )
            else:
                # Use the LLM response as summary with generic recommendations
                generic_recs = self._create_generic_recommendations()
                total_reduction = sum(rec.emission_reduction_kg for rec in generic_recs)
                annual_savings = total_reduction * 365
                
                return AgentResponse(
                    current_emission=0.0,
                    recommendations=generic_recs[:3],
                    total_potential_reduction=total_reduction,
                    annual_savings_kg=annual_savings,
                    summary=llm_response
                )
                
        except Exception as e:
            # If LLM fails, return the original out-of-scope message
            summary = (
                "I couldn't find relevant information in my sustainability knowledge base to answer your question. "
                "I specialize in CO₂ reduction and carbon footprint advice. "
                "Please ask about transportation, energy use, food choices, or other sustainability topics."
            )
            
            recommendations = self._create_generic_recommendations()
            total_reduction = sum(rec.emission_reduction_kg for rec in recommendations)
            annual_savings = total_reduction * 365
            
            return AgentResponse(
                current_emission=0.0,
                recommendations=recommendations[:3],
                total_potential_reduction=total_reduction,
                annual_savings_kg=annual_savings,
                summary=summary
            )
    
    def _generate_llm_recommendations_for_dataset(
        self,
        analysis: DatasetAnalysis
    ) -> List[Recommendation]:
        """
        Generate recommendations using LLM for datasets with unknown categories.
        
        Args:
            analysis: DatasetAnalysis object
            
        Returns:
            List of LLM-generated recommendations
        """
        try:
            # Build context about the dataset
            activities_text = "\n".join([
                f"- {act.name}: {act.emission_kg_per_day:.2f} kg CO2/day ({act.category.value})"
                for act in analysis.top_emitters[:5]
            ])
            
            category_text = "\n".join([
                f"- {cat}: {emission:.2f} kg CO2/day"
                for cat, emission in analysis.category_breakdown.items()
            ])
            
            # Retrieve relevant context from vector store
            search_query = f"reduce emissions {' '.join([act.name for act in analysis.top_emitters[:3]])}"
            retrieved_docs = self.vector_store.search(search_query, k=3)
            context = [doc.content for doc in retrieved_docs] if retrieved_docs else []
            
            # Build prompt
            prompt = f"""You are a CO₂ reduction advisor analyzing a user's carbon footprint dataset.

Dataset Summary:
- Total Daily Emission: {analysis.total_daily_emission:.2f} kg CO2/day
- Total Annual Emission: {analysis.total_annual_emission:.2f} kg CO2/year

Top Emitting Activities:
{activities_text}

Category Breakdown:
{category_text}

Based on this data and the following sustainability knowledge, provide 5 specific, actionable recommendations to reduce their carbon footprint:

"""
            if context:
                prompt += "\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(context)])
            
            prompt += "\n\nProvide recommendations in this format:\n1. [Action] - [Brief explanation with estimated CO2 reduction]\n2. [Action] - [Brief explanation]\n..."
            
            llm_response = self.llm.generate(prompt, max_tokens=600, temperature=0.4)
            
            # Parse recommendations
            recommendations = self._parse_llm_recommendations(llm_response)
            
            if not recommendations:
                # Fallback to generic recommendations
                recommendations = self._create_generic_recommendations()
            
            return recommendations
            
        except Exception as e:
            print(f"LLM recommendation generation failed: {str(e)}")
            # Fallback to generic recommendations
            return self._create_generic_recommendations()
    
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
    
    def _create_recommendations_from_context(
        self,
        context: List[str],
        query: str
    ) -> List[Recommendation]:
        """
        Create recommendations from retrieved context documents.
        
        Args:
            context: List of retrieved context strings
            query: Original user query
            
        Returns:
            List of Recommendation objects based on context
        """
        recommendations = []
        
        for ctx in context[:5]:  # Use top 5 context items
            # Parse the context to extract recommendation info
            # Context format from sustainability_tips.txt is typically:
            # "Title: [action]\nCategory: [category]\nDescription: [details]\nEmission Reduction: [amount]"
            
            lines = ctx.split('\n')
            action = ""
            emission_reduction = 1.0  # Default
            difficulty = "Medium"
            benefits = []
            
            for line in lines:
                line = line.strip()
                if line.startswith("Title:"):
                    action = line.replace("Title:", "").strip()
                elif line.startswith("Emission Reduction:"):
                    # Try to extract number
                    import re
                    numbers = re.findall(r'\d+\.?\d*', line)
                    if numbers:
                        emission_reduction = float(numbers[0])
                elif line.startswith("Difficulty:"):
                    difficulty = line.replace("Difficulty:", "").strip()
                elif line.startswith("Additional Benefits:"):
                    benefits_text = line.replace("Additional Benefits:", "").strip()
                    benefits = [b.strip() for b in benefits_text.split(',')]
            
            if action:
                recommendations.append(Recommendation(
                    action=action,
                    emission_reduction_kg=emission_reduction,
                    reduction_percentage=20.0,  # Estimate
                    implementation_difficulty=difficulty,
                    timeframe="Short-term",
                    additional_benefits=benefits if benefits else ["Environmental impact"]
                ))
        
        # If we couldn't parse any recommendations, return generic ones
        if not recommendations:
            return self._create_generic_recommendations()
        
        return recommendations
    
    def _parse_llm_recommendations(self, llm_response: str) -> List[Recommendation]:
        """
        Parse LLM response to extract recommendations.
        
        Args:
            llm_response: Raw LLM response text
            
        Returns:
            List of Recommendation objects parsed from response
        """
        recommendations = []
        
        # Try to extract numbered recommendations
        import re
        lines = llm_response.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for numbered items or bullet points
            if re.match(r'^[\d\-\*\•]+[\.\)]\s+', line):
                # Extract the recommendation text
                action = re.sub(r'^[\d\-\*\•]+[\.\)]\s+', '', line).strip()
                
                # Try to extract emission reduction if mentioned
                emission_reduction = 0.5  # Default
                numbers = re.findall(r'(\d+(?:\.\d+)?)\s*(?:kg|kilogram)', action.lower())
                if numbers:
                    emission_reduction = float(numbers[0])
                
                if action and len(action) > 10:  # Valid recommendation
                    recommendations.append(Recommendation(
                        action=action,
                        emission_reduction_kg=emission_reduction,
                        reduction_percentage=15.0,
                        implementation_difficulty="Medium",
                        timeframe="Short-term",
                        additional_benefits=["Environmental impact"]
                    ))
        
        return recommendations[:5]  # Return top 5
    
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
