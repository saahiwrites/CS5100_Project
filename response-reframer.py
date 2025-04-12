"""
Response reframing component to mitigate biases in generated text.
This module implements techniques to reframe biased responses while
preserving the core information and intent of the original response.
"""

from typing import Dict, List, Optional, Any
import re

from ..models.llm_interface import LLMInterface

class ResponseReframer:
    """
    Reframes responses to mitigate detected biases.
    Uses a combination of rule-based techniques and LLM-based reframing.
    """
    
    def __init__(self, sensitivity: float = 0.5, model: Optional[str] = None):
        """
        Initialize the response reframer.
        
        Args:
            sensitivity: Threshold for bias detection (0.0 to 1.0)
            model: Optional model name for LLM-based reframing
        """
        self.sensitivity = sensitivity
        self.model_name = model or "gpt-3.5-turbo"
        
        # Initialize LLM for reframing if needed
        self.llm = LLMInterface(model=self.model_name)
        
        # Load reframing rules
        self.reframing_rules = self._load_reframing_rules()
        
        # Load bias detection patterns
        self.bias_patterns = self._load_bias_patterns()
    
    def _load_reframing_rules(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Load rules for reframing biased language.
        
        Returns:
            Dictionary mapping categories to lists of reframing rules
        """
        return {
            "gender": [
                {"pattern": r"all women are (\w+)", "replacement": r"some women may be \1"},
                {"pattern": r"all men are (\w+)", "replacement": r"some men may be \1"},
                {"pattern": r"women (tend to|usually|often|typically) (\w+)", 
                 "replacement": r"some women may \2"},
                {"pattern": r"men (tend to|usually|often|typically) (\w+)", 
                 "replacement": r"some men may \2"},
                {"pattern": r"(he|she) is (better at|worse at|more suited for) (\w+)", 
                 "replacement": r"individuals vary in their aptitude for \3"}
            ],
            "race": [
                {"pattern": r"(\w+) people are (known for|characterized by) (\w+)", 
                 "replacement": r"some \1 individuals may \2 \3"},
                {"pattern": r"people from (\w+) (are|tend to be) (\w+)", 
                 "replacement": r"there is diversity among people from \1"},
                {"pattern": r"(all|most) (\w+) are (\w+)", 
                 "replacement": r"there is significant diversity within \2 communities"}
            ],
            "age": [
                {"pattern": r"(older|younger) people (can't|cannot|don't|do not|aren't able to) (\w+)", 
                 "replacement": r"some \1 individuals may face challenges with \3"},
                {"pattern": r"(millennials|boomers|gen z|seniors) (always|all|tend to) (\w+)", 
                 "replacement": r"some \1 may \3, though there is significant individual variation"},
                {"pattern": r"people over (\d+) (can't|cannot|don't) (\w+)", 
                 "replacement": r"some individuals over \1 may find \3 challenging, though experiences vary widely"}
            ],
            "religion": [
                {"pattern": r"(muslims|christians|jews|hindus|buddhists) are (\w+)", 
                 "replacement": r"some people who practice \1 may be \2, though there is much diversity"},
                {"pattern": r"people who (believe in|practice|follow) (\w+) (tend to|are) (\w+)", 
                 "replacement": r"some people who \1 \2 may be \4, though experiences vary widely"}
            ],
            "socioeconomic": [
                {"pattern": r"(poor|rich|wealthy|low-income) people (are|have|lack) (\w+)", 
                 "replacement": r"some people with \1 backgrounds may \2 \3, though individual circumstances vary"},
                {"pattern": r"people (without|with) (education|degrees|college) (can't|cannot|don't) (\w+)", 
                 "replacement": r"formal education is one of many factors that can influence \4"}
            ]
        }
    
    def _load_bias_patterns(self) -> Dict[str, List[re.Pattern]]:
        """
        Load patterns for detecting biased language.
        
        Returns:
            Dictionary mapping categories to lists of regex patterns
        """
        patterns = {}
        
        for category, rules in self.reframing_rules.items():
            patterns[category] = [re.compile(rule["pattern"], re.IGNORECASE) for rule in rules]
        
        return patterns
    
    def reframe(
        self, 
        response: str, 
        bias_categories: List[str],
        bias_scores: Dict[str, float],
        user_input: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Reframe a response to mitigate detected biases.
        
        Args:
            response: The original response to reframe
            bias_categories: Categories of bias detected
            bias_scores: Scores indicating bias severity by category
            user_input: Optional original user query for context
            context: Optional additional context
            
        Returns:
            Reframed response with reduced bias
        """
        # Apply rule-based reframing first
        reframed = self._apply_reframing_rules(response, bias_categories)
        
        # If bias score still high or rule-based reframing didn't change much,
        # use LLM-based reframing
        if self._should_use_llm_reframing(response, reframed, bias_scores):
            reframed = self._llm_reframing(response, bias_categories, user_input, context)
        
        return reframed
    
    def _apply_reframing_rules(self, text: str, bias_categories: List[str]) -> str:
        """
        Apply rule-based reframing to the text.
        
        Args:
            text: The text to reframe
            bias_categories: Categories of bias to target
            
        Returns:
            Reframed text
        """
        reframed = text
        
        for category in bias_categories:
            if category in self.reframing_rules:
                for rule in self.reframing_rules[category]:
                    pattern = re.compile(rule["pattern"], re.IGNORECASE)
                    reframed = pattern.sub(rule["replacement"], reframed)
        
        return reframed
    
    def _should_use_llm_reframing(
        self, 
        original: str, 
        reframed: str, 
        bias_scores: Dict[str, float]
    ) -> bool:
        """
        Determine if LLM-based reframing should be used.
        
        Args:
            original: Original response
            reframed: Rule-based reframed response
            bias_scores: Bias scores by category
            
        Returns:
            Boolean indicating whether to use LLM reframing
        """
        # Check if any bias score is still high
        high_bias = any(score > self.sensitivity for category, score in bias_scores.items())
        
        # Check if rule-based reframing made minimal changes
        minimal_change = self._similarity_ratio(original, reframed) > 0.9
        
        return high_bias or minimal_change
    
    def _similarity_ratio(self, a: str, b: str) -> float:
        """
        Calculate the similarity ratio between two strings.
        
        Args:
            a: First string
            b: Second string
            
        Returns:
            Similarity ratio between 0.0 and 1.0
        """
        # This is a simplified similarity measure
        # A production system would use a more sophisticated approach
        # like Levenshtein distance or cosine similarity
        
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
            
        # Simple character-based similarity
        shorter = min(len(a), len(b))
        longer = max(len(a), len(b))
        
        if longer == 0:
            return 1.0
            
        matching = sum(1 for i in range(shorter) if a[i] == b[i])
        return matching / longer
    
    def _llm_reframing(
        self, 
        response: str, 
        bias_categories: List[str],
        user_input: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Use an LLM to reframe the response for reduced bias.
        
        Args:
            response: Original response to reframe
            bias_categories: Categories of bias to target
            user_input: Optional original user query
            context: Optional additional context
            
        Returns:
            LLM-reframed response
        """
        # Construct prompt for reframing
        categories_str = ", ".join(bias_categories)
        
        prompt = f"""
Reframe the following text to reduce bias related to {categories_str} while preserving 
the core information, intent, and helpfulness of the response. Maintain factual accuracy
and a balanced perspective. Do not add disclaimers or explanations about bias - just
rewrite the content itself to be more balanced and inclusive.

Original response:
"{response}"

Reframed response:
"""
        
        # Add context if available
        if user_input:
            prompt = f"Original question: {user_input}\n\n{prompt}"
            
        # Generate reframed response
        reframed = self.llm.generate(prompt, context or {})
        
        # Clean up the response
        reframed = reframed.strip()
        if reframed.startswith('"') and reframed.endswith('"'):
            reframed = reframed[1:-1]
            
        return reframed
    
    def get_reframing_suggestions(self, text: str, bias_categories: List[str]) -> Dict[str, List[Dict[str, str]]]:
        """
        Get suggestions for reframing without applying them.
        
        Args:
            text: The text to analyze
            bias_categories: Categories of bias to consider
            
        Returns:
            Dictionary mapping categories to lists of suggested replacements
        """
        suggestions = {}
        
        for category in bias_categories:
            if category in self.bias_patterns:
                category_suggestions = []
                
                for pattern in self.bias_patterns[category]:
                    matches = pattern.finditer(text)
                    for match in matches:
                        matched_text = match.group(0)
                        
                        # Find the corresponding replacement rule
                        for rule in self.reframing_rules.get(category, []):
                            if re.compile(rule["pattern"], re.IGNORECASE).search(matched_text):
                                replacement = re.sub(
                                    re.compile(rule["pattern"], re.IGNORECASE),
                                    rule["replacement"],
                                    matched_text
                                )
                                
                                category_suggestions.append({
                                    "original": matched_text,
                                    "suggested": replacement,
                                    "start": match.start(),
                                    "end": match.end()
                                })
                                break
                
                if category_suggestions:
                    suggestions[category] = category_suggestions
        
        return suggestions
