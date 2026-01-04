"""Text preprocessing utilities for financial sentiment analysis."""

import re
from typing import List, Optional, Union

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class TextPreprocessor:
    """Text preprocessing pipeline for financial sentiment analysis.
    
    This class provides comprehensive text cleaning and preprocessing
    specifically tailored for financial news and social media content.
    """
    
    def __init__(self, language: str = "english"):
        """Initialize the text preprocessor.
        
        Args:
            language: Language for stopwords and tokenization
        """
        self.language = language
        self.lemmatizer = WordNetLemmatizer()
        
        # Download required NLTK data
        try:
            self.stop_words = set(stopwords.words(language))
        except LookupError:
            nltk.download("stopwords")
            self.stop_words = set(stopwords.words(language))
        
        try:
            word_tokenize("test")
        except LookupError:
            nltk.download("punkt")
        
        try:
            self.lemmatizer.lemmatize("test")
        except LookupError:
            nltk.download("wordnet")
        
        # Financial-specific stopwords to keep
        self.financial_terms = {
            "bull", "bear", "rally", "crash", "surge", "plunge", "volatility",
            "earnings", "revenue", "profit", "loss", "growth", "decline",
            "upgrade", "downgrade", "outperform", "underperform", "buy", "sell",
            "hold", "target", "price", "stock", "shares", "market", "trading"
        }
        
        # Remove financial terms from stopwords
        self.stop_words = self.stop_words - self.financial_terms
    
    def clean_text(self, text: str, remove_urls: bool = True, 
                   remove_mentions: bool = True, remove_hashtags: bool = False) -> str:
        """Clean text by removing unwanted elements.
        
        Args:
            text: Input text to clean
            remove_urls: Whether to remove URLs
            remove_mentions: Whether to remove @mentions
            remove_hashtags: Whether to remove #hashtags
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        if remove_urls:
            text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", 
                         "", text)
        
        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)
        
        # Remove phone numbers
        text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "", text)
        
        # Remove mentions
        if remove_mentions:
            text = re.sub(r"@\w+", "", text)
        
        # Remove hashtags (optional)
        if remove_hashtags:
            text = re.sub(r"#\w+", "", text)
        
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        if not isinstance(text, str):
            return []
        
        tokens = word_tokenize(text)
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from token list.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of tokens with stopwords removed
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens to their base forms.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text: str, steps: Optional[List[str]] = None) -> str:
        """Apply full preprocessing pipeline.
        
        Args:
            text: Input text to preprocess
            steps: List of preprocessing steps to apply
            
        Returns:
            Preprocessed text
        """
        if steps is None:
            steps = ["clean", "tokenize", "remove_stopwords", "lemmatize", "join"]
        
        processed = text
        
        if "clean" in steps:
            processed = self.clean_text(processed)
        
        if "tokenize" in steps:
            processed = self.tokenize(processed)
        
        if "remove_stopwords" in steps and isinstance(processed, list):
            processed = self.remove_stopwords(processed)
        
        if "lemmatize" in steps and isinstance(processed, list):
            processed = self.lemmatize(processed)
        
        if "join" in steps and isinstance(processed, list):
            processed = " ".join(processed)
        
        return processed
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str, 
                           output_column: Optional[str] = None) -> pd.DataFrame:
        """Preprocess text column in DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing text
            output_column: Name of output column (defaults to f"{text_column}_processed")
            
        Returns:
            DataFrame with preprocessed text column
        """
        if output_column is None:
            output_column = f"{text_column}_processed"
        
        df = df.copy()
        df[output_column] = df[text_column].apply(self.preprocess)
        
        return df
    
    def extract_financial_entities(self, text: str) -> dict:
        """Extract financial entities and patterns from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing extracted financial information
        """
        entities = {
            "has_earnings_keywords": False,
            "has_merger_keywords": False,
            "has_regulatory_keywords": False,
            "has_price_target": False,
            "has_percentage": False,
            "has_dollar_amount": False
        }
        
        text_lower = text.lower()
        
        # Earnings keywords
        earnings_keywords = ["earnings", "revenue", "profit", "loss", "quarterly", "annual"]
        entities["has_earnings_keywords"] = any(keyword in text_lower for keyword in earnings_keywords)
        
        # Merger keywords
        merger_keywords = ["merger", "acquisition", "takeover", "buyout", "deal"]
        entities["has_merger_keywords"] = any(keyword in text_lower for keyword in merger_keywords)
        
        # Regulatory keywords
        regulatory_keywords = ["sec", "fda", "regulation", "compliance", "investigation", "fine"]
        entities["has_regulatory_keywords"] = any(keyword in text_lower for keyword in regulatory_keywords)
        
        # Price target patterns
        price_target_pattern = r"\$(\d+(?:\.\d{2})?)"
        entities["has_price_target"] = bool(re.search(price_target_pattern, text))
        
        # Percentage patterns
        percentage_pattern = r"(\d+(?:\.\d+)?%)"
        entities["has_percentage"] = bool(re.search(percentage_pattern, text))
        
        # Dollar amount patterns
        dollar_pattern = r"\$(\d+(?:,\d{3})*(?:\.\d{2})?)"
        entities["has_dollar_amount"] = bool(re.search(dollar_pattern, text))
        
        return entities
