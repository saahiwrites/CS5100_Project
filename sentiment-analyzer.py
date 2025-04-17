"""
Sentiment analysis component for detecting biases in text.
This module implements advanced sentiment analysis techniques 
specifically tuned for detecting subtle biases in conversational text.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import defaultdict

# For production code, you would use libraries like:
# import nltk
# from nltk.sentiment import SentimentIntensityAnalyzer
# from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

class SentimentAnalyzer:
    """
    Analyzes text for sentiment patterns that may indicate bias.
    Uses a combination of lexicon-based analysis and contextual patterns.
    """
    
    def __init__(self, custom_lexicon: Optional[Dict[str, float]] = None):
        """
        Initialize the sentiment analyzer.
        
        Args:
            custom_lexicon: Optional custom lexicon of words and their sentiment scores
        """
        self.lexicon = self._load_default_lexicon()
        if custom_lexicon:
            self.lexicon.update(custom_lexicon)
            
        # Load bias-specific patterns
        self.bias_patterns = self._load_bias_patterns()
        
        # For production, initialize more sophisticated models here:
        # self.transformer_analyzer = pipeline(
        #     "sentiment-analysis", 
        #     model="distilbert-base-uncased-finetuned-sst-2-english"
        # )
    
    def _load_default_lexicon(self) -> Dict[str, float]:
        """
        Load the default sentiment lexicon.
        In a production system, this would load from a file.
        
        Returns:
            Dictionary mapping words to sentiment scores
        """
        # This is a minimal example - a real implementation would load
        # from a comprehensive lexicon file with thousands of entries
        return {
            # Gender-related terms
            "man": 0.0, "woman": 0.0, "male": 0.0, "female": 0.0,
            "emotional": -0.2, "rational": 0.2, "nurturing": 0.1, "assertive": 0.1,
            "bossy": -0.4, "ambitious": 0.3,
            
            # Race-related terms
            "articulate": 0.1, "threatening": -0.5, "hardworking": 0.3,
            "exotic": -0.2, "dangerous": -0.5, "intelligent": 0.4,
            
            # Age-related terms
            "experienced": 0.3, "energetic": 0.2, "slow": -0.3, "outdated": -0.4,
            "wise": 0.4, "naive": -0.3, "tech-savvy": 0.2,
            
            # Religion-related terms
            "devout": 0.1, "radical": -0.4, "traditional": 0.0, "extremist": -0.6,
            
            # Socioeconomic terms
            "professional": 0.3, "uneducated": -0.4, "poor": -0.2, "wealthy": 0.2,
            "lazy": -0.5, "hardworking": 0.4, "welfare": -0.1
        }
    
    def _load_bias_patterns(self) -> Dict[str, List[re.Pattern]]:
        """
        Load regex patterns for detecting specific biases.
        
        Returns:
            Dictionary mapping bias categories to lists of compiled regex patterns
        """
        patterns = {
            "gender": [
                re.compile(r"women (tend to|usually|often|typically) (be|are) (\w+)", re.IGNORECASE),
                re.compile(r"men (tend to|usually|often|typically) (be|are) (\w+)", re.IGNORECASE),
                re.compile(r"(all|most) (men|women|males|females) are", re.IGNORECASE)
            ],
            "race": [
                re.compile(r"people from (\w+) (are|tend to be) (\w+)", re.IGNORECASE),
                re.compile(r"(\w+) people are (known for|characterized by)", re.IGNORECASE)
            ],
            "age": [
                re.compile(r"(older|younger) people (can't|cannot|don't|do not|aren't able to)", re.IGNORECASE),
                re.compile(r"(millennials|boomers|gen z|seniors) (always|all|tend to)", re.IGNORECASE)
            ],
            "religion": [
                re.compile(r"(muslims|christians|jews|hindus|buddhists) are", re.IGNORECASE),
                re.compile(r"people who (believe in|practice|follow) (\w+) (tend to|are)", re.IGNORECASE)
            ],
            "socioeconomic": [
                re.compile(r"(poor|rich|wealthy|low-income) people (are|have|lack)", re.IGNORECASE),
                re.compile(r"people (without|with) (education|degrees|college) (can't|cannot|don't)", re.IGNORECASE)
            ]
        }
        return patterns
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for sentiment patterns that may indicate bias.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary containing sentiment scores and detected bias patterns
        """
        # Normalize text
        normalized_text = text.lower()
        words = re.findall(r'\b\w+\b', normalized_text)
        
        # Calculate lexicon-based sentiment
        sentiment_scores = self._calculate_lexicon_sentiment(words)
        
        # Detect bias patterns
        bias_matches = self._detect_bias_patterns(normalized_text)
        
        # Perform sentence-level analysis
        sentence_sentiments = self._analyze_sentences(text)
        
        # For production, add transformer-based analysis:
        # transformer_results = self._get_transformer_sentiment(text)
        
        return {
            "overall_sentiment": sentiment_scores["overall"],
            "category_sentiments": sentiment_scores["categories"],
            "detected_patterns": bias_matches,
            "sentence_analysis": sentence_sentiments,
            # "transformer_sentiment": transformer_results
        }
    
    def _calculate_lexicon_sentiment(self, words: List[str]) -> Dict[str, Any]:
        """
        Calculate sentiment scores based on lexicon matching.
        
        Args:
            words: List of words from the text
            
        Returns:
            Dictionary with overall sentiment and category-specific sentiments
        """
        word_sentiments = [self.lexicon.get(word, 0) for word in words if word in self.lexicon]
        
        # If no sentiment words found, return neutral
        if not word_sentiments:
            return {
                "overall": 0.0,
                "categories": {
                    "gender": 0.0,
                    "race": 0.0,
                    "age": 0.0,
                    "religion": 0.0,
                    "socioeconomic": 0.0
                }
            }
        
        # Calculate overall sentiment
        overall = sum(word_sentiments) / len(word_sentiments)
        
        # Categorize words by bias category
        category_matches = {
            "gender": ["man", "woman", "male", "female", "emotional", "rational", "nurturing", "assertive", 
                      "bossy", "ambitious"],
            "race": ["articulate", "threatening", "exotic", "dangerous", "intelligent"],
            "age": ["experienced", "energetic", "slow", "outdated", "wise", "naive", "tech-savvy"],
            "religion": ["devout", "radical", "traditional", "extremist"],
            "socioeconomic": ["professional", "uneducated", "poor", "wealthy", "lazy", "hardworking", "welfare"]
        }
        
        # Calculate sentiment by category
        category_sentiments = {}
        for category, category_words in category_matches.items():
            matched_sentiments = [self.lexicon.get(word, 0) for word in words 
                                 if word in category_words]
            
            if matched_sentiments:
                category_sentiments[category] = sum(matched_sentiments) / len(matched_sentiments)
            else:
                category_sentiments[category] = 0.0
        
        return {
            "overall": overall,
            "categories": category_sentiments
        }
    
    def _detect_bias_patterns(self, text: str) -> Dict[str, List[str]]:
        """
        Detect bias patterns in text using regex patterns.
        
        Args:
            text: The normalized text to analyze
            
        Returns:
            Dictionary mapping bias categories to lists of matched text
        """
        matches = defaultdict(list)
        
        for category, patterns in self.bias_patterns.items():
            for pattern in patterns:
                found_matches = pattern.findall(text)
                if found_matches:
                    for match in found_matches:
                        if isinstance(match, tuple):
                            matches[category].append(" ".join(match))
                        else:
                            matches[category].append(match)
        
        return dict(matches)
    
    def _analyze_sentences(self, text: str) -> List[Dict[str, Any]]:
        """
        Perform sentiment analysis on individual sentences.
        
        Args:
            text: The text to analyze
            
        Returns:
            List of dictionaries with sentiment info for each sentence
        """
        # Simple sentence splitter - a production system would use nltk or similar
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        results = []
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            words = re.findall(r'\b\w+\b', sentence.lower())
            word_sentiments = [self.lexicon.get(word, 0) for word in words if word in self.lexicon]
            
            if word_sentiments:
                sentiment = sum(word_sentiments) / len(word_sentiments)
            else:
                sentiment = 0.0
                
            # Check for bias patterns
            sentence_bias = {}
            for category, patterns in self.bias_patterns.items():
                for pattern in patterns:
                    if pattern.search(sentence.lower()):
                        if category not in sentence_bias:
                            sentence_bias[category] = True
            
            results.append({
                "text": sentence,
                "sentiment": sentiment,
                "bias_categories": list(sentence_bias.keys())
            })
            
        return results
    
    # def _get_transformer_sentiment(self, text: str) -> Dict[str, float]:
    #     """
    #     Get sentiment analysis from transformer model.
    #     """
    #     chunks = self._split_into_chunks(text)
    #     results = []
    #     
    #     for chunk in chunks:
    #         prediction = self.transformer_analyzer(chunk)
    #         results.append(prediction)
    #     
    #     # Aggregate results
    #     positive = [r[0]["score"] if r[0]["label"] == "POSITIVE" else r[1]["score"] 
    #                 for r in results if len(r) > 1]
    #     negative = [r[0]["score"] if r[0]["label"] == "NEGATIVE" else r[1]["score"] 
    #                 for r in results if len(r) > 1]
    #     
    #     if positive and negative:
    #         avg_positive = sum(positive) / len(positive)
    #         avg_negative = sum(negative) / len(negative)
    #         sentiment = avg_positive - avg_negative
    #     else:
    #         sentiment = 0.0
    #     
    #     return {
    #         "positive": avg_positive if positive else 0.0,
    #         "negative": avg_negative if negative else 0.0,
    #         "sentiment": sentiment
    #     }
