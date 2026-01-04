"""Tests for market sentiment analysis models."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.models import FinBERTSentimentModel, BaselineSentimentModel
from src.data import SyntheticDataGenerator, TextPreprocessor
from src.utils.evaluation import Evaluator


class TestBaselineModels:
    """Test baseline sentiment analysis models."""
    
    def test_textblob_model(self):
        """Test TextBlob baseline model."""
        model = BaselineSentimentModel("textblob")
        
        # Test positive sentiment
        positive_text = "Apple stock surges after strong earnings"
        prediction = model.predict(positive_text)
        assert prediction in ["positive", "neutral", "negative"]
        
        # Test negative sentiment
        negative_text = "Market crashes amid economic uncertainty"
        prediction = model.predict(negative_text)
        assert prediction in ["positive", "neutral", "negative"]
        
        # Test batch prediction
        texts = [positive_text, negative_text]
        predictions = model.predict(texts)
        assert len(predictions) == 2
        assert all(pred in ["positive", "neutral", "negative"] for pred in predictions)
    
    def test_vader_model(self):
        """Test VADER baseline model."""
        model = BaselineSentimentModel("vader")
        
        # Test prediction
        text = "This is amazing news for investors!"
        prediction = model.predict(text)
        assert prediction in ["positive", "neutral", "negative"]
        
        # Test probabilities
        probabilities = model.predict_proba(text)
        assert len(probabilities) == 3
        assert abs(sum(probabilities) - 1.0) < 1e-6
    
    def test_ml_model_training(self):
        """Test machine learning model training."""
        model = BaselineSentimentModel("logistic")
        
        # Create sample training data
        texts = [
            "Great earnings report",
            "Terrible market performance",
            "Neutral market conditions",
            "Strong growth prospects",
            "Declining revenue"
        ]
        labels = ["positive", "negative", "neutral", "positive", "negative"]
        
        # Train model
        model.fit(texts, labels)
        
        # Test prediction
        prediction = model.predict("Excellent quarterly results")
        assert prediction in ["positive", "neutral", "negative"]


class TestSyntheticDataGenerator:
    """Test synthetic data generation."""
    
    def test_text_data_generation(self):
        """Test synthetic text data generation."""
        config = Mock()
        config.num_samples = 100
        config.symbols = ["AAPL", "MSFT"]
        config.text_generation = Mock()
        config.text_generation.sentiment_distribution = Mock()
        config.text_generation.sentiment_distribution.positive = 0.4
        config.text_generation.sentiment_distribution.neutral = 0.3
        config.text_generation.sentiment_distribution.negative = 0.3
        config.text_generation.keywords = {
            "positive": ["surge", "growth"],
            "negative": ["decline", "crash"],
            "neutral": ["report", "analysis"]
        }
        
        generator = SyntheticDataGenerator(config)
        text_data = generator.generate_text_data(50)
        
        assert len(text_data) == 50
        assert "date" in text_data.columns
        assert "symbol" in text_data.columns
        assert "headline" in text_data.columns
        assert "sentiment_label" in text_data.columns
        
        # Check sentiment distribution
        sentiment_counts = text_data["sentiment_label"].value_counts()
        assert all(sentiment in ["positive", "neutral", "negative"] for sentiment in sentiment_counts.index)
    
    def test_market_data_generation(self):
        """Test synthetic market data generation."""
        config = Mock()
        config.symbols = ["AAPL", "MSFT"]
        config.start_date = "2020-01-01"
        config.end_date = "2020-01-31"
        config.market_simulation = Mock()
        config.market_simulation.base_price = 100.0
        config.market_simulation.drift = 0.05
        config.market_simulation.volatility = 0.2
        config.market_simulation.jump_probability = 0.1
        config.market_simulation.jump_size = 0.05
        
        generator = SyntheticDataGenerator(config)
        market_data = generator.generate_market_data(["AAPL"], "2020-01-01", "2020-01-31")
        
        assert len(market_data) > 0
        assert "date" in market_data.columns
        assert "symbol" in market_data.columns
        assert "open" in market_data.columns
        assert "high" in market_data.columns
        assert "low" in market_data.columns
        assert "close" in market_data.columns
        assert "volume" in market_data.columns


class TestTextPreprocessor:
    """Test text preprocessing utilities."""
    
    def test_text_cleaning(self):
        """Test text cleaning functionality."""
        preprocessor = TextPreprocessor()
        
        # Test URL removal
        text_with_url = "Check out this link: https://example.com for more info"
        cleaned = preprocessor.clean_text(text_with_url, remove_urls=True)
        assert "https://example.com" not in cleaned
        
        # Test mention removal
        text_with_mention = "Great analysis by @financialexpert"
        cleaned = preprocessor.clean_text(text_with_mention, remove_mentions=True)
        assert "@financialexpert" not in cleaned
        
        # Test case conversion
        text_mixed_case = "APPLE Stock Surges!"
        cleaned = preprocessor.clean_text(text_mixed_case)
        assert cleaned == "apple stock surges!"
    
    def test_financial_entity_extraction(self):
        """Test financial entity extraction."""
        preprocessor = TextPreprocessor()
        
        text = "Apple reports $100M revenue growth in Q4 earnings"
        entities = preprocessor.extract_financial_entities(text)
        
        assert isinstance(entities, dict)
        assert "has_earnings_keywords" in entities
        assert "has_dollar_amount" in entities
        assert entities["has_earnings_keywords"] == True
        assert entities["has_dollar_amount"] == True
    
    def test_preprocessing_pipeline(self):
        """Test full preprocessing pipeline."""
        preprocessor = TextPreprocessor()
        
        text = "APPLE Stock Surges! Check https://example.com for details @trader"
        processed = preprocessor.preprocess(text)
        
        assert isinstance(processed, str)
        assert "apple" in processed.lower()
        assert "https://example.com" not in processed
        assert "@trader" not in processed


class TestEvaluator:
    """Test evaluation utilities."""
    
    def test_classification_evaluation(self):
        """Test classification evaluation metrics."""
        evaluator = Evaluator()
        
        y_true = ["positive", "negative", "neutral", "positive", "negative"]
        y_pred = ["positive", "negative", "neutral", "neutral", "negative"]
        
        metrics = evaluator.evaluate_classification(y_true, y_pred)
        
        assert "accuracy" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert "f1_macro" in metrics
        assert "confusion_matrix" in metrics
        
        assert 0 <= metrics["accuracy"] <= 1
        assert isinstance(metrics["confusion_matrix"], list)
    
    def test_financial_performance_evaluation(self):
        """Test financial performance evaluation."""
        evaluator = Evaluator()
        
        # Create sample market data
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "symbol": ["AAPL"] * 100,
            "close": 100 + np.cumsum(np.random.randn(100) * 0.01),
            "sentiment_pred": np.random.choice(["positive", "negative", "neutral"], 100)
        })
        
        predictions = df["sentiment_pred"].tolist()
        metrics = evaluator.evaluate_financial_performance(df, predictions)
        
        assert isinstance(metrics, dict)
        if metrics:  # If we have valid data
            assert "total_return" in metrics
            assert "volatility" in metrics


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    @patch('src.models.finbert_model.AutoTokenizer')
    @patch('src.models.finbert_model.AutoModelForSequenceClassification')
    def test_finbert_model_initialization(self, mock_model, mock_tokenizer):
        """Test FinBERT model initialization."""
        # Mock the transformers components
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        
        config = Mock()
        config.model = Mock()
        config.model.model_name = "ProsusAI/finbert"
        config.model.num_labels = 3
        config.model.max_length = 512
        config.model.batch_size = 16
        config.model.learning_rate = 2e-5
        config.model.num_epochs = 3
        config.model.warmup_steps = 100
        config.model.weight_decay = 0.01
        config.model.early_stopping_patience = 3
        config.data = Mock()
        config.data.train_split = 0.7
        config.data.val_split = 0.15
        config.data.test_split = 0.15
        
        model = FinBERTSentimentModel(config)
        
        assert model.tokenizer is not None
        assert model.model is not None
        assert model.label_map == {"negative": 0, "neutral": 1, "positive": 2}
    
    def test_end_to_end_pipeline(self):
        """Test end-to-end pipeline with synthetic data."""
        # This is a simplified test that doesn't require actual model training
        config = Mock()
        config.num_samples = 50
        config.symbols = ["AAPL"]
        config.text_generation = Mock()
        config.text_generation.sentiment_distribution = Mock()
        config.text_generation.sentiment_distribution.positive = 0.4
        config.text_generation.sentiment_distribution.neutral = 0.3
        config.text_generation.sentiment_distribution.negative = 0.3
        config.text_generation.keywords = {
            "positive": ["surge", "growth"],
            "negative": ["decline", "crash"],
            "neutral": ["report", "analysis"]
        }
        config.market_simulation = Mock()
        config.market_simulation.base_price = 100.0
        config.market_simulation.drift = 0.05
        config.market_simulation.volatility = 0.2
        config.market_simulation.jump_probability = 0.1
        config.market_simulation.jump_size = 0.05
        config.start_date = "2020-01-01"
        config.end_date = "2020-01-31"
        
        # Generate data
        generator = SyntheticDataGenerator(config)
        text_data, market_data = generator.generate_combined_dataset()
        
        # Preprocess text
        preprocessor = TextPreprocessor()
        text_data = preprocessor.preprocess_dataframe(text_data, "headline")
        
        # Test baseline model
        model = BaselineSentimentModel("textblob")
        predictions = model.predict(text_data["headline"].tolist())
        
        # Evaluate
        evaluator = Evaluator()
        metrics = evaluator.evaluate_classification(
            text_data["sentiment_label"].tolist(),
            predictions
        )
        
        assert metrics["accuracy"] >= 0
        assert metrics["accuracy"] <= 1


if __name__ == "__main__":
    pytest.main([__file__])
