"""
Unit tests for recommendation validation and ranking.

Tests emission calculations, recommendation ranking, and validation logic.
"""

import pytest
from components.response_validator import ResponseValidator
from components.recommendation_ranker import RecommendationRanker, RankingWeights
from components.context_manager import UserContext, ContextAwareRecommender, Budget, Lifestyle, Timeframe


class TestResponseValidator:
    """Test suite for ResponseValidator."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.validator = ResponseValidator()
    
    def test_emission_sanity_checks(self):
        """Test basic emission value validation."""
        # Negative emissions should fail
        is_valid, msg = self.validator.validate_emission_value("test activity", -1.0)
        assert not is_valid
        assert "Negative" in msg
        
        # Extremely high emissions should fail
        is_valid, msg = self.validator.validate_emission_value("test activity", 150.0)
        assert not is_valid
        assert "high" in msg.lower()
        
        # Reasonable emissions should pass
        is_valid, msg = self.validator.validate_emission_value("test activity", 5.0)
        assert is_valid
    
    def test_reduction_calculation(self):
        """Test reduction percentage calculation validation."""
        # Correct calculation should pass
        is_valid, msg = self.validator.validate_reduction_calculation(
            current_emission=10.0,
            alternative_emission=3.0,
            claimed_reduction_pct=70.0,
            tolerance=5.0
        )
        assert is_valid
        
        # Incorrect calculation should fail
        is_valid, msg = self.validator.validate_reduction_calculation(
            current_emission=10.0,
            alternative_emission=3.0,
            claimed_reduction_pct=50.0,  # Should be 70%
            tolerance=5.0
        )
        assert not is_valid
        assert "error" in msg.lower()
    
    def test_annual_savings_calculation(self):
        """Test annual savings calculation validation."""
        # Correct calculation: 1 kg/day * 365 = 365 kg/year
        is_valid, msg = self.validator.validate_annual_savings(
            daily_reduction=1.0,
            claimed_annual=365.0,
            tolerance=10.0
        )
        assert is_valid
        
        # Incorrect calculation
        is_valid, msg = self.validator.validate_annual_savings(
            daily_reduction=1.0,
            claimed_annual=200.0,  # Should be 365
            tolerance=10.0
        )
        assert not is_valid
    
    def test_recommendation_validation(self):
        """Test complete recommendation validation."""
        # Valid recommendation
        valid_rec = {
            'activity': 'Driving petrol car 20km',
            'current_emission': 4.6,
            'alternative_name': 'Electric car',
            'alternative_emission': 1.2,
            'reduction_percentage': 74.0,
            'annual_savings_kg': 1241.0
        }
        
        is_valid, issues = self.validator.validate_recommendation(valid_rec)
        assert is_valid
        assert len(issues) == 0
        
        # Invalid recommendation (missing fields)
        invalid_rec = {
            'activity': 'Driving',
            'current_emission': 4.6
        }
        
        is_valid, issues = self.validator.validate_recommendation(invalid_rec)
        assert not is_valid
        assert len(issues) > 0
    
    def test_auto_correct_calculations(self):
        """Test automatic correction of calculation errors."""
        rec_with_error = {
            'current_emission': 10.0,
            'alternative_emission': 3.0,
            'reduction_percentage': 50.0,  # Wrong: should be 70%
            'annual_savings_kg': 2000.0     # Wrong: should be 2555
        }
        
        corrected = self.validator.auto_correct_calculations(rec_with_error)
        
        assert corrected['reduction_percentage'] == 70.0
        assert corrected['annual_savings_kg'] == 2555.0
        assert corrected.get('_corrected') is True


class TestRecommendationRanker:
    """Test suite for RecommendationRanker."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.ranker = RecommendationRanker()
    
    def test_emission_score_calculation(self):
        """Test emission reduction scoring."""
        # High reduction should score well
        rec_high = {'reduction_percentage': 80, 'annual_savings_kg': 1000}
        score = self.ranker._calculate_emission_score(rec_high)
        assert score >= 0.8
        
        # Low reduction should score lower
        rec_low = {'reduction_percentage': 20, 'annual_savings_kg': 100}
        score = self.ranker._calculate_emission_score(rec_low)
        assert score <= 0.3
    
    def test_cost_effectiveness_score(self):
        """Test cost-effectiveness scoring."""
        # Low cost should score well
        rec_low_cost = {
            'cost_category': 'low',
            'annual_savings_kg': 500,
            'co_benefits': 'Cost savings of $500/year'
        }
        score = self.ranker._calculate_cost_effectiveness_score(rec_low_cost)
        assert score >= 0.7
        
        # High cost should score lower
        rec_high_cost = {
            'cost_category': 'high',
            'annual_savings_kg': 100
        }
        score = self.ranker._calculate_cost_effectiveness_score(rec_high_cost)
        assert score <= 0.5
    
    def test_ease_score_calculation(self):
        """Test ease of implementation scoring."""
        # Easy should score 1.0
        rec_easy = {'difficulty': 'Easy', 'prerequisites': 'None'}
        score = self.ranker._calculate_ease_score(rec_easy)
        assert score == 1.0
        
        # Hard should score lower
        rec_hard = {'difficulty': 'Hard', 'prerequisites': 'Requires contractor approval'}
        score = self.ranker._calculate_ease_score(rec_hard)
        assert score <= 0.3
    
    def test_recommendation_ranking(self):
        """Test full recommendation ranking."""
        recommendations = [
            {
                'name': 'Low impact, easy',
                'reduction_percentage': 20,
                'annual_savings_kg': 100,
                'cost_category': 'low',
                'difficulty': 'Easy',
                'timeframe': 'Immediate',
                'co_benefits': 'Health benefits'
            },
            {
                'name': 'High impact, hard',
                'reduction_percentage': 80,
                'annual_savings_kg': 1500,
                'cost_category': 'high',
                'difficulty': 'Hard',
                'timeframe': 'Long-term',
                'co_benefits': 'Cost savings, health benefits'
            },
            {
                'name': 'Medium impact, medium',
                'reduction_percentage': 50,
                'annual_savings_kg': 500,
                'cost_category': 'medium',
                'difficulty': 'Medium',
                'timeframe': 'Short-term',
                'co_benefits': 'Convenience, health'
            }
        ]
        
        ranked = self.ranker.rank_recommendations(recommendations)
        
        # Should have scores assigned
        assert all('ranking_score' in rec for rec in ranked)
        
        # Should be sorted by score
        scores = [rec['ranking_score'] for rec in ranked]
        assert scores == sorted(scores, reverse=True)
    
    def test_custom_weights(self):
        """Test ranking with custom weights."""
        # Prioritize emission reduction heavily
        custom_weights = RankingWeights(
            emission_reduction=0.70,
            cost_effectiveness=0.10,
            ease_of_implementation=0.10,
            time_to_impact=0.05,
            co_benefits=0.05
        )
        
        ranker = RecommendationRanker(weights=custom_weights)
        
        # High emission reduction should rank top
        recs = [
            {'reduction_percentage': 20, 'annual_savings_kg': 100, 'cost_category': 'low', 
             'difficulty': 'Easy', 'timeframe': 'Immediate', 'co_benefits': 'Many'},
            {'reduction_percentage': 90, 'annual_savings_kg': 2000, 'cost_category': 'high',
             'difficulty': 'Hard', 'timeframe': 'Long-term', 'co_benefits': 'Few'}
        ]
        
        ranked = ranker.rank_recommendations(recs)
        assert ranked[0]['reduction_percentage'] == 90


class TestContextAwareRecommender:
    """Test suite for ContextAwareRecommender."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.recommender = ContextAwareRecommender()
    
    def test_budget_filtering(self):
        """Test budget-based filtering."""
        recommendations = [
            {'name': 'Cheap option', 'cost_category': 'low', 'reduction_percentage': 50},
            {'name': 'Expensive option', 'cost_category': 'high', 'reduction_percentage': 80}
        ]
        
        # Low budget user should not see expensive options
        context = UserContext(budget=Budget.LOW)
        filtered = self.recommender.filter_recommendations(recommendations, context)
        
        assert len(filtered) == 1
        assert filtered[0]['name'] == 'Cheap option'
    
    def test_lifestyle_filtering(self):
        """Test lifestyle-based filtering."""
        recommendations = [
            {'name': 'Use public transport', 'prerequisites': 'Requires bus routes'},
            {'name': 'Install solar', 'prerequisites': 'Requires large property'}
        ]
        
        # Rural user should not see public transport
        context = UserContext(lifestyle=Lifestyle.RURAL)
        filtered = self.recommender.filter_recommendations(recommendations, context)
        
        assert len(filtered) == 1
        assert 'solar' in filtered[0]['name'].lower()
    
    def test_timeframe_filtering(self):
        """Test timeframe-based filtering."""
        recommendations = [
            {'name': 'Immediate action', 'timeframe': 'Immediate', 'reduction_percentage': 30},
            {'name': 'Long-term project', 'timeframe': 'Long-term', 'reduction_percentage': 90}
        ]
        
        # Immediate timeframe user should prefer quick actions
        context = UserContext(timeframe=Timeframe.IMMEDIATE)
        filtered = self.recommender.filter_recommendations(recommendations, context)
        
        assert len(filtered) == 1
        assert filtered[0]['name'] == 'Immediate action'
    
    def test_context_extraction(self):
        """Test extracting context from query."""
        query = "I need a cheap solution now for my city apartment"
        context = self.recommender.extract_context_from_query(query)
        
        assert context.budget == Budget.LOW
        assert context.timeframe == Timeframe.IMMEDIATE
        assert context.lifestyle == Lifestyle.URBAN
    
    def test_context_score_calculation(self):
        """Test context relevance scoring."""
        rec = {
            'name': 'Low-cost immediate action',
            'cost_category': 'low',
            'timeframe': 'Immediate',
            'co_benefits': 'Cost savings'
        }
        
        context = UserContext(budget=Budget.LOW, timeframe=Timeframe.IMMEDIATE)
        score = self.recommender._calculate_context_score(rec, context)
        
        # Should score high for matching context
        assert score >= 0.8


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_validated_ranking_workflow(self):
        """Test complete workflow: validate, then rank."""
        validator = ResponseValidator()
        ranker = RecommendationRanker()
        
        recommendations = [
            {
                'activity': 'Driving petrol car',
                'current_emission': 4.6,
                'alternative_name': 'Electric car',
                'alternative_emission': 1.2,
                'reduction_percentage': 74.0,
                'annual_savings_kg': 1241.0,
                'cost_category': 'high',
                'difficulty': 'Hard',
                'timeframe': 'Long-term',
                'co_benefits': 'Lower fuel costs'
            },
            {
                'activity': 'Driving petrol car',
                'current_emission': 4.6,
                'alternative_name': 'Public transport',
                'alternative_emission': 1.2,
                'reduction_percentage': 74.0,
                'annual_savings_kg': 1241.0,
                'cost_category': 'low',
                'difficulty': 'Easy',
                'timeframe': 'Immediate',
                'co_benefits': 'Cost savings, productive time'
            }
        ]
        
        # Validate all recommendations
        valid, invalid = validator.validate_batch(recommendations)
        assert len(valid) == 2
        assert len(invalid) == 0
        
        # Rank valid recommendations
        ranked = ranker.rank_recommendations(valid)
        
        # Public transport should rank higher (easier, cheaper, immediate)
        assert 'Public transport' in ranked[0]['alternative_name']
    
    def test_context_aware_validated_ranking(self):
        """Test full workflow with context filtering."""
        validator = ResponseValidator()
        ranker = RecommendationRanker()
        context_recommender = ContextAwareRecommender()
        
        recommendations = [
            {
                'activity': 'Driving',
                'current_emission': 4.6,
                'alternative_name': 'Electric car',
                'alternative_emission': 1.2,
                'reduction_percentage': 74.0,
                'annual_savings_kg': 1241.0,
                'cost_category': 'high',
                'difficulty': 'Hard',
                'timeframe': 'Long-term',
                'co_benefits': 'Lower fuel costs'
            },
            {
                'activity': 'Driving',
                'current_emission': 4.6,
                'alternative_name': 'Cycling',
                'alternative_emission': 0.0,
                'reduction_percentage': 100.0,
                'annual_savings_kg': 1679.0,
                'cost_category': 'low',
                'difficulty': 'Medium',
                'timeframe': 'Immediate',
                'co_benefits': 'Health, zero cost'
            }
        ]
        
        # 1. Validate
        valid, _ = validator.validate_batch(recommendations)
        
        # 2. Filter by context (low budget, immediate timeframe)
        context = UserContext(budget=Budget.LOW, timeframe=Timeframe.IMMEDIATE)
        filtered = context_recommender.filter_recommendations(valid, context)
        
        # 3. Rank filtered recommendations
        ranked = ranker.rank_recommendations(filtered)
        
        # Should only have cycling (low cost, immediate)
        assert len(ranked) == 1
        assert 'Cycling' in ranked[0]['alternative_name']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
