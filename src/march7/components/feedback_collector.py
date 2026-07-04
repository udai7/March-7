"""
User feedback collection and storage system.

Collects user feedback on recommendation quality to improve the system
and identify problem areas.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Feedback:
    """
    User feedback on a recommendation.
    
    Attributes:
        timestamp: When feedback was submitted
        query: Original user query
        response: System response that was rated
        rating: 1-5 star rating (1=poor, 5=excellent)
        is_helpful: Boolean helpful/not helpful
        is_accurate: Whether recommendation was accurate
        feedback_text: Optional text feedback
        category: Category of query (transport, food, etc.)
        user_id: Optional anonymous user identifier
    """
    timestamp: str
    query: str
    response: str
    rating: Optional[int] = None
    is_helpful: Optional[bool] = None
    is_accurate: Optional[bool] = None
    feedback_text: Optional[str] = None
    category: Optional[str] = None
    user_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert feedback to dictionary."""
        return asdict(self)


class FeedbackCollector:
    """
    Collects and stores user feedback on recommendations.
    
    Saves feedback to JSON file for analysis and system improvement.
    """
    
    def __init__(self, feedback_file: str = "data/feedback.jsonl"):
        """
        Initialize feedback collector.
        
        Args:
            feedback_file: Path to feedback storage file (JSONL format)
        """
        self.feedback_file = Path(feedback_file)
        
        # Create directory if it doesn't exist
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create file if it doesn't exist
        if not self.feedback_file.exists():
            self.feedback_file.touch()
    
    def save_feedback(
        self,
        query: str,
        response: str,
        rating: Optional[int] = None,
        is_helpful: Optional[bool] = None,
        is_accurate: Optional[bool] = None,
        feedback_text: Optional[str] = None,
        category: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Save user feedback.
        
        Args:
            query: Original user query
            response: System response
            rating: 1-5 star rating
            is_helpful: Boolean helpful/not helpful
            is_accurate: Whether recommendation was accurate
            feedback_text: Optional text feedback
            category: Category of query
            user_id: Optional user identifier
            
        Returns:
            True if feedback saved successfully
        """
        feedback = Feedback(
            timestamp=datetime.now().isoformat(),
            query=query,
            response=response,
            rating=rating,
            is_helpful=is_helpful,
            is_accurate=is_accurate,
            feedback_text=feedback_text,
            category=category,
            user_id=user_id
        )
        
        try:
            # Append feedback to JSONL file
            with open(self.feedback_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(feedback.to_dict()) + '\n')
            return True
        except Exception as e:
            print(f"Error saving feedback: {e}")
            return False
    
    def log_inaccuracy_report(
        self,
        query: str,
        response: str,
        issue_description: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Log an inaccuracy report.
        
        Args:
            query: Original query
            response: Inaccurate response
            issue_description: Description of the issue
            user_id: Optional user identifier
            
        Returns:
            True if logged successfully
        """
        return self.save_feedback(
            query=query,
            response=response,
            rating=1,
            is_accurate=False,
            feedback_text=issue_description or "User reported inaccuracy",
            user_id=user_id
        )
    
    def load_all_feedback(self) -> List[Dict[str, Any]]:
        """
        Load all feedback from file.
        
        Returns:
            List of feedback dictionaries
        """
        feedback_list = []
        
        try:
            if self.feedback_file.exists():
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            feedback_list.append(json.loads(line))
        except Exception as e:
            print(f"Error loading feedback: {e}")
        
        return feedback_list
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get feedback statistics.
        
        Returns:
            Dictionary with feedback statistics
        """
        all_feedback = self.load_all_feedback()
        
        if not all_feedback:
            return {
                "total_count": 0,
                "average_rating": None,
                "helpful_percentage": None,
                "accurate_percentage": None,
                "categories": {}
            }
        
        # Calculate statistics
        ratings = [f['rating'] for f in all_feedback if f.get('rating')]
        helpful_count = sum(1 for f in all_feedback if f.get('is_helpful'))
        accurate_count = sum(1 for f in all_feedback if f.get('is_accurate'))
        
        # Category breakdown
        categories = {}
        for f in all_feedback:
            cat = f.get('category', 'Unknown')
            if cat not in categories:
                categories[cat] = {'count': 0, 'ratings': []}
            categories[cat]['count'] += 1
            if f.get('rating'):
                categories[cat]['ratings'].append(f['rating'])
        
        # Calculate category averages
        for cat in categories:
            if categories[cat]['ratings']:
                categories[cat]['average_rating'] = sum(categories[cat]['ratings']) / len(categories[cat]['ratings'])
            else:
                categories[cat]['average_rating'] = None
        
        return {
            "total_count": len(all_feedback),
            "average_rating": sum(ratings) / len(ratings) if ratings else None,
            "helpful_percentage": (helpful_count / len(all_feedback) * 100) if all_feedback else None,
            "accurate_percentage": (accurate_count / len(all_feedback) * 100) if all_feedback else None,
            "categories": categories
        }
    
    def get_low_rated_queries(self, threshold: int = 2) -> List[Dict[str, Any]]:
        """
        Get queries with low ratings for analysis.
        
        Args:
            threshold: Maximum rating to include (default 2)
            
        Returns:
            List of low-rated feedback items
        """
        all_feedback = self.load_all_feedback()
        
        low_rated = [
            f for f in all_feedback
            if f.get('rating') and f['rating'] <= threshold
        ]
        
        return low_rated
    
    def get_inaccuracy_reports(self) -> List[Dict[str, Any]]:
        """
        Get all inaccuracy reports.
        
        Returns:
            List of feedback items marked as inaccurate
        """
        all_feedback = self.load_all_feedback()
        
        inaccurate = [
            f for f in all_feedback
            if f.get('is_accurate') is False
        ]
        
        return inaccurate
    
    def export_feedback(self, output_file: str = "feedback_export.json") -> bool:
        """
        Export all feedback to a JSON file.
        
        Args:
            output_file: Output file path
            
        Returns:
            True if exported successfully
        """
        all_feedback = self.load_all_feedback()
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "feedback": all_feedback,
                    "statistics": self.get_statistics(),
                    "exported_at": datetime.now().isoformat()
                }, f, indent=2)
            return True
        except Exception as e:
            print(f"Error exporting feedback: {e}")
            return False
