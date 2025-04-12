"""
Bias metrics component for quantifying different types of bias in text.
This module implements various metrics to measure the degree of different
bias types in textual content.
"""

from typing import Dict, List, Optional, Any
import re
import math

class BiasMetrics:
    """
    Calculates bias metrics for text based on sentiment analysis results
    and additional bias-specific measures.
    """
    
    def __init__(self, categories: Optional[List[str]] = None):
        """
        Initialize the bias metrics calculator.
        
        Args:
            categories: List of bias categories to monitor
        """
        self.categories = categories or ["gender", "race", "age", "religion", "socioeconomic"]
        
        # Load bias-specific keyword dictionaries
        self.bias_keywords = self._load_bias_keywords()
        
        # Load language model association pairs
        self.association_pairs = self._load_association_pairs()
    
    def _load_bias_keywords(self) -> Dict[str, Dict[str, float]]:
        """
        Load keyword dictionaries for different bias categories.
        In a production system, this would load from files.
        
        Returns:
            Dictionary mapping categories to keyword dictionaries
        """
        # This is a minimal example - a real implementation would load
        # from comprehensive files with more entries and nuanced scoring
        return {
            "gender": {
                "emotional": 0.7, "rational": 0.7, "bossy": 0.8, "assertive": 0.7,
                "nurse": 0.6, "doctor": 0.6, "caring": 0.5, "ambitious": 0.5,
                "sensitive": 0.6, "strong": 0.6, "leader": 0.7, "helper": 0.7,
                "he": 0.3, "she": 0.3, "man": 0.4, "woman": 0.4, "men": 0.4, "women": 0.4
            },
            "race": {
                "articulate": 0.7, "intelligent": 0.6, "dangerous": 0.8, "criminal": 0.9,
                "hardworking": 0.6, "lazy": 0.8, "exotic": 0.7, "normal": 0.5,
                "aggressive": 0.7, "violent": 0.8, "skilled": 0.5, "athletic": 0.6,
                "ethnic": 0.4, "cultural": 0.4, "minority": 0.5, "diverse": 0.5
            },
            "age": {
                "experienced": 0.6, "outdated": 0.7, "energetic": 0.6, "tired": 0.7,
                "wise": 0.6, "forgetful": 0.7, "tech-savvy": 0.7, "traditional": 0.6,
                "innovative": 0.6, "old-fashioned": 0.7, "mature": 0.5, "youthful": 0.5,
                "senior": 0.4, "young": 0.4, "elderly": 0.5, "middle-aged": 0.4
            },
            "religion": {
                "devout": 0.6, "fanatic": 0.8, "spiritual": 0.5, "fundamentalist": 0.8,
                "traditional": 0.5, "extremist": 0.9, "peaceful": 0.6, "radical": 0.8,
                "moderate": 0.5, "conservative": 0.6, "liberal": 0.6, "orthodox": 0.7,
                "believer": 0.4, "atheist": 0.4, "religious": 0.5, "secular": 0.5
            },
            "socioeconomic": {
                "educated": 0.6, "uneducated": 0.8, "professional": 0.6, "unskilled": 0.7,
                "wealthy": 0.7, "poor": 0.7, "privileged": 0.7, "disadvantaged": 0.7,
                "hard-working": 0.6, "lazy": 0.8, "deserving": 0.7, "entitled": 0.7,
                "elite": 0.6, "working-class": 0.6, "upper-class": 0.6, "lower-class": 0.6
            }
        }
    
    def _load_association_pairs(self) -> Dict[str, List[tuple]]:
        """
        Load word association pairs for bias detection.
        These pairs represent stereotypical or biased associations.
        
        Returns:
            Dictionary mapping categories to lists of word pairs
        """
        return {
            "gender": [
                ("women", "emotional"), ("men", "rational"),
                ("women", "caring"), ("men", "strong"),
                ("women", "nurturing"), ("men", "ambitious"),
                ("girls", "gentle"), ("boys", "tough")
            ],
            "race": [
                ("asian", "smart"), ("black", "athletic"),
                ("white", "privileged"), ("latino", "hardworking"),
                ("middle eastern", "religious"), ("indian", "technical")
            ],
            "age": [
                ("young", "inexperienced"), ("old", "wise"),
                ("millennial", "entitled"), ("boomer", "outdated"),
                ("teenager", "irresponsible"), ("senior", "slow")
            ],
            "religion": [
                ("muslim", "radical"), ("christian", "traditional"),
                ("jewish", "frugal"), ("atheist", "immoral"),
                ("hindu", "spiritual"), ("buddhist", "peaceful")
            ],
            "socioeconomic": [
                ("rich", "greedy"), ("poor", "lazy"),
                ("educated", "elite"), ("uneducated", "ignorant"),
                ("wealthy", "selfish"), ("low-income", "unmotivated")
            ]
        }
    
    def calculate_bias(self, text: str, sentiment: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate bias metrics for the given text.
        
        Args:
            text: The text to analyze
            sentiment: Sentiment analysis results from SentimentAnalyzer
            
        Returns:
            Dictionary mapping bias categories to bias scores (0.0 to 1.0)
        """
        # Initialize results with zeros
        bias_scores = {category: 0.0 for category in self.categories}
        
        # Calculate keyword bias
        keyword_bias = self._calculate_keyword_bias(text)
        
        # Calculate association bias
        association_bias = self._calculate_association_bias(text)
        
        # Calculate sentiment-based bias
        sentiment_bias = self._calculate_sentiment_bias(sentiment)
        
        # Calculate pattern-based bias
        pattern_bias = self._calculate_pattern_bias(sentiment.get("detected_patterns", {}))
        
        # Combine the different bias metrics
        for category in self.categories:
            # Weighted combination of different bias signals
            # Weights can be adjusted based on performance
            bias_scores[category] = (
                0.3 * keyword_bias.get(category, 0.0) +
                0.3 * association_bias.get(category, 0.0) +
                0.2 * sentiment_bias.get(category, 0.0) +
                0.2 * pattern_bias.get(category, 0.0)
            )
            
            # Ensure the score is in the range [0, 1]
            bias_scores[category] = max(0.0, min(1.0, bias_scores[category]))
        
        return bias_scores
    
    def _calculate_keyword_bias(self, text: str) -> Dict[str, float]:
        """
        Calculate bias based on presence of category-specific keywords.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary mapping categories to keyword bias scores
        """
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        
        bias_scores = {}
        
        for category, keywords in self.bias_keywords.items():
            category_score = 0.0
            matched_keywords = 0
            
            for word in words:
                if word in keywords:
                    category_score += keywords[word]
                    matched_keywords += 1
            
            # Calculate normalized score
            if matched_keywords > 0:
                bias_scores[category] = category_score / matched_keywords
            else:
                bias_scores[category] = 0.0
        
        return bias_scores
    
    def _calculate_association_bias(self, text: str) -> Dict[str, float]:
        """
        Calculate bias based on stereotypical word associations.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary mapping categories to association bias scores
        """
        text = text.lower()
        bias_scores = {}
        
        for category, pairs in self.association_pairs.items():
            category_score = 0.0
            matched_pairs = 0
            
            for word1, word2 in pairs:
                # Check for both words within a reasonable distance
                if word1 in text and word2 in text:
                    # Simple proximity check - a real implementation would be more sophisticated
                    if self._words_in_proximity(text, word1, word2, window_size=10):
                        category_score += 1.0
                        matched_pairs += 1
            
            # Calculate normalized score
            if matched_pairs > 0:
                # Adjust by number of total possible pairs to prevent bias from large numbers
                normalized_score = category_score / len(pairs)
                bias_scores[category] = normalized_score
            else:
                bias_scores[category] = 0.0
        
        return bias_scores
    
    def _words_in_proximity(self, text: str, word1: str, word2: str, window_size: int = 10) -> bool:
        """
        Check if two words appear within a certain window of each other in the text.
        
        Args:
            text: The text to check
            word1: First word to look for
            word2: Second word to look for
            window_size: Maximum number of words between the two target words
            
        Returns:
            Boolean indicating whether the words are in proximity
        """
        words = re.findall(r'\b\w+\b', text)
        
        for i, word in enumerate(words):
            if word == word1:
                # Look for word2 within the window
                start = max(0, i - window_size)
                end = min(len(words), i + window_size + 1)
                
                if word2 in words[start:end]:
                    return True
            
            elif word == word2:
                # Look for word1 within the window
                start = max(0, i - window_size)
                end = min(len(words), i + window_size + 1)
                
                if word1 in words[start:end]:
                    return True
        
        return False
    
    def _calculate_sentiment_bias(self, sentiment: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate bias based on sentiment analysis results.
        
        Args:
            sentiment: Sentiment analysis results
            
        Returns:
            Dictionary mapping categories to sentiment-based bias scores
        """
        bias_scores = {}
        
        # Extract category sentiments if available
        category_sentiments = sentiment.get("category_sentiments", {})
        
        for category in self.categories:
            if category in category_sentiments:
                # Convert sentiment value to a bias score
                # The further from neutral (0), the higher the bias
                sentiment_value = category_sentiments[category]
                bias_scores[category] = min(1.0, abs(sentiment_value) * 2.0)
            else:
                bias_scores[category] = 0.0
                
        # Analyze sentence-level biases
        sentence_analysis = sentiment.get("sentence_analysis", [])
        if sentence_analysis:
            for category in self.categories:
                category_sentences = [
                    s for s in sentence_analysis 
                    if category in s.get("bias_categories", [])
                ]
                
                if category_sentences:
                    # Average the sentiment values
                    avg_sentiment = sum(abs(s.get("sentiment", 0.0)) for s in category_sentences) / len(category_sentences)
                    
                    # Combine with existing score
                    if category in bias_scores:
                        bias_scores[category] = max(bias_scores[category], min(1.0, avg_sentiment * 2.0))
                    else:
                        bias_scores[category] = min(1.0, avg_sentiment * 2.0)
        
        return bias_scores
    
    def _calculate_pattern_bias(self, detected_patterns: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Calculate bias based on detected bias patterns.
        
        Args:
            detected_patterns: Dictionary of detected bias patterns by category
            
        Returns:
            Dictionary mapping categories to pattern-based bias scores
        """
        bias_scores = {}
        
        for category in self.categories:
            if category in detected_patterns and detected_patterns[category]:
                # Base score on number of patterns detected, with diminishing returns
                num_patterns = len(detected_patterns[category])
                bias_scores[category] = min(1.0, 0.4 + (1 - math.exp(-0.5 * num_patterns)))
            else:
                bias_scores[category] = 0.0
        
        return bias_scores
    
    def update_categories(self, categories: List[str]) -> None:
        """
        Update the list of bias categories to monitor.
        
        Args:
            categories: New list of categories
        """
        self.categories = categories
