"""
Main implementation of the Debiased AI Agent.
This module contains the core agent class that integrates
the bias detection and mitigation components.
"""

import logging
from typing import Dict, List, Optional, Union, Any

from .bias_detector.sentiment_analyzer import SentimentAnalyzer
from .bias_detector.bias_metrics import BiasMetrics
from .debiasing.reframing import ResponseReframer
from .models.llm_interface import LLMInterface
from .utils.logging_utils import setup_logger
from .utils.text_processing import preprocess_text

logger = setup_logger(__name__)

class DeBiasedAgent:
    """
    An AI agent that uses sentiment analysis to detect and mitigate
    biases in conversational responses.
    """
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        bias_categories: Optional[List[str]] = None,
        sensitivity: float = 0.5,
        model_config: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the debiased agent.
        
        Args:
            model: The name of the LLM to use
            bias_categories: List of bias categories to monitor
            sensitivity: Threshold for bias detection (0.0 to 1.0)
            model_config: Additional configuration for the model
            api_key: API key for the LLM service
        """
        self.model_config = model_config or {}
        self.bias_categories = bias_categories or ["gender", "race", "age", "religion", "socioeconomic"]
        self.sensitivity = max(0.0, min(1.0, sensitivity))  # Clamp between 0 and 1
        
        # Initialize components
        self.llm = LLMInterface(model=model, api_key=api_key, **self.model_config)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.bias_metrics = BiasMetrics(categories=self.bias_categories)
        self.reframer = ResponseReframer(sensitivity=self.sensitivity)
        
        logger.info(f"Initialized DeBiasedAgent with model: {model}")
        logger.info(f"Monitoring bias categories: {self.bias_categories}")
    
    def generate_response(
        self, 
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        max_attempts: int = 3
    ) -> Dict[str, Any]:
        """
        Generate a debiased response to user input.
        
        Args:
            user_input: The user's query or message
            context: Additional context for response generation
            max_attempts: Maximum number of debiasing attempts
        
        Returns:
            Dictionary containing the debiased response and bias metrics
        """
        context = context or {}
        processed_input = preprocess_text(user_input)
        
        # Generate initial response from the LLM
        initial_response = self.llm.generate(
            prompt=processed_input,
            context=context
        )
        
        # Analyze the response for potential biases
        sentiment_scores = self.sentiment_analyzer.analyze(initial_response)
        bias_scores = self.bias_metrics.calculate_bias(
            text=initial_response,
            sentiment=sentiment_scores
        )
        
        # Check if debiasing is needed
        needs_debiasing = any(
            score > self.sensitivity for category, score in bias_scores.items()
            if category in self.bias_categories
        )
        
        if not needs_debiasing:
            logger.info("No significant bias detected, returning original response")
            return {
                "response": initial_response,
                "bias_scores": bias_scores,
                "was_debiased": False,
                "debiasing_attempts": 0
            }
        
        # Apply debiasing through iterative reframing
        current_response = initial_response
        current_scores = bias_scores
        attempts = 0
        
        while needs_debiasing and attempts < max_attempts:
            attempts += 1
            logger.info(f"Debiasing attempt {attempts}/{max_attempts}")
            
            # Get problematic categories
            problematic_categories = [
                category for category, score in current_scores.items()
                if score > self.sensitivity and category in self.bias_categories
            ]
            
            # Reframe the response
            reframed_response = self.reframer.reframe(
                response=current_response,
                bias_categories=problematic_categories,
                bias_scores=current_scores,
                user_input=user_input,
                context=context
            )
            
            # Re-analyze the reframed response
            sentiment_scores = self.sentiment_analyzer.analyze(reframed_response)
            new_bias_scores = self.bias_metrics.calculate_bias(
                text=reframed_response,
                sentiment=sentiment_scores
            )
            
            # Check if we've made progress
            improved = all(
                new_bias_scores.get(cat, 0) <= current_scores.get(cat, 0)
                for cat in problematic_categories
            )
            
            if improved:
                current_response = reframed_response
                current_scores = new_bias_scores
            
            # Check if we've reached acceptable levels
            needs_debiasing = any(
                score > self.sensitivity for category, score in current_scores.items()
                if category in self.bias_categories
            )
            
            if not needs_debiasing or not improved:
                break
        
        return {
            "response": current_response,
            "initial_response": initial_response,
            "bias_scores": {
                "initial": bias_scores,
                "final": current_scores
            },
            "was_debiased": True,
            "debiasing_attempts": attempts,
            "problematic_categories": problematic_categories if 'problematic_categories' in locals() else []
        }
    
    def evaluate_bias(self, text: str) -> Dict[str, float]:
        """
        Analyze a text for potential biases without generating a response.
        
        Args:
            text: The text to analyze
        
        Returns:
            Dictionary of bias scores by category
        """
        sentiment_scores = self.sentiment_analyzer.analyze(text)
        bias_scores = self.bias_metrics.calculate_bias(text, sentiment_scores)
        return bias_scores
    
    def update_sensitivity(self, sensitivity: float) -> None:
        """Update the sensitivity threshold for bias detection."""
        self.sensitivity = max(0.0, min(1.0, sensitivity))
        self.reframer.sensitivity = self.sensitivity
        logger.info(f"Updated sensitivity threshold to {self.sensitivity}")
    
    def add_bias_category(self, category: str) -> None:
        """Add a new bias category to monitor."""
        if category not in self.bias_categories:
            self.bias_categories.append(category)
            self.bias_metrics.update_categories(self.bias_categories)
            logger.info(f"Added new bias category: {category}")
    
    def remove_bias_category(self, category: str) -> None:
        """Remove a bias category from monitoring."""
        if category in self.bias_categories:
            self.bias_categories.remove(category)
            self.bias_metrics.update_categories(self.bias_categories)
            logger.info(f"Removed bias category: {category}")
