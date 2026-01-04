"""Synthetic data generation for market sentiment analysis."""

import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig


class SyntheticDataGenerator:
    """Generates synthetic financial news and market data for testing and demonstration.
    
    This class creates realistic synthetic data that mimics real financial news
    and market behavior patterns for research and educational purposes.
    """
    
    def __init__(self, config: DictConfig):
        """Initialize the synthetic data generator.
        
        Args:
            config: Configuration object containing generation parameters
        """
        self.config = config
        self.text_config = config.text_generation
        self.market_config = config.market_simulation
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
    
    def generate_text_data(self, num_samples: int) -> pd.DataFrame:
        """Generate synthetic financial news text data.
        
        Args:
            num_samples: Number of text samples to generate
            
        Returns:
            DataFrame containing synthetic text data with sentiment labels
        """
        data = []
        
        for i in range(num_samples):
            # Generate random date within range
            start_date = datetime.strptime("2020-01-01", "%Y-%m-%d")
            end_date = datetime.strptime("2023-12-31", "%Y-%m-%d")
            random_date = start_date + timedelta(
                days=random.randint(0, (end_date - start_date).days)
            )
            
            # Generate random symbol
            symbol = random.choice(self.config.symbols)
            
            # Generate sentiment label based on distribution
            sentiment_label = self._sample_sentiment()
            
            # Generate headline based on sentiment
            headline = self._generate_headline(symbol, sentiment_label)
            
            data.append({
                "date": random_date,
                "symbol": symbol,
                "headline": headline,
                "sentiment_label": sentiment_label,
                "text_length": len(headline),
                "word_count": len(headline.split())
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values("date").reset_index(drop=True)
        
        return df
    
    def generate_market_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Generate synthetic market data (OHLCV).
        
        Args:
            symbols: List of stock symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame containing synthetic market data
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        market_data = []
        
        for symbol in symbols:
            # Generate price series using geometric Brownian motion
            prices = self._generate_price_series(
                len(date_range),
                self.market_config.base_price,
                self.market_config.drift,
                self.market_config.volatility
            )
            
            # Generate volume data
            volumes = self._generate_volume_series(len(date_range))
            
            for i, date in enumerate(date_range):
                # Add some randomness to OHLC
                open_price = prices[i] * (1 + np.random.normal(0, 0.01))
                high_price = max(open_price, prices[i]) * (1 + abs(np.random.normal(0, 0.005)))
                low_price = min(open_price, prices[i]) * (1 - abs(np.random.normal(0, 0.005)))
                close_price = prices[i]
                
                market_data.append({
                    "date": date,
                    "symbol": symbol,
                    "open": round(open_price, 2),
                    "high": round(high_price, 2),
                    "low": round(low_price, 2),
                    "close": round(close_price, 2),
                    "volume": volumes[i]
                })
        
        df = pd.DataFrame(market_data)
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        
        return df
    
    def generate_combined_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate combined text and market dataset.
        
        Returns:
            Tuple of (text_data, market_data) DataFrames
        """
        # Generate text data
        text_data = self.generate_text_data(self.config.num_samples)
        
        # Generate market data
        market_data = self.generate_market_data(
            self.config.symbols,
            self.config.start_date,
            self.config.end_date
        )
        
        return text_data, market_data
    
    def _sample_sentiment(self) -> str:
        """Sample sentiment label based on configured distribution.
        
        Returns:
            Sentiment label ("positive", "neutral", "negative")
        """
        dist = self.text_config.sentiment_distribution
        rand = random.random()
        
        if rand < dist.positive:
            return "positive"
        elif rand < dist.positive + dist.neutral:
            return "neutral"
        else:
            return "negative"
    
    def _generate_headline(self, symbol: str, sentiment: str) -> str:
        """Generate a realistic financial headline.
        
        Args:
            symbol: Stock symbol
            sentiment: Sentiment label
            
        Returns:
            Generated headline text
        """
        keywords = self.text_config.keywords[sentiment]
        
        # Base templates for different sentiment
        templates = {
            "positive": [
                f"{symbol} stock {random.choice(keywords)} as investors show confidence",
                f"{symbol} {random.choice(keywords)} in latest trading session",
                f"Analysts {random.choice(keywords)} {symbol} outlook",
                f"{symbol} {random.choice(keywords)} following strong earnings report",
                f"Market {random.choice(keywords)} {symbol} shares"
            ],
            "negative": [
                f"{symbol} stock {random.choice(keywords)} amid market concerns",
                f"{symbol} {random.choice(keywords)} in latest trading session",
                f"Analysts {random.choice(keywords)} {symbol} outlook",
                f"{symbol} {random.choice(keywords)} following disappointing results",
                f"Market {random.choice(keywords)} {symbol} shares"
            ],
            "neutral": [
                f"{symbol} {random.choice(keywords)} scheduled for next week",
                f"{symbol} {random.choice(keywords)} shows mixed signals",
                f"Analysts {random.choice(keywords)} {symbol} performance",
                f"{symbol} {random.choice(keywords)} released quarterly results",
                f"Market {random.choice(keywords)} {symbol} trading activity"
            ]
        }
        
        template = random.choice(templates[sentiment])
        
        # Add some random financial terms
        financial_terms = [
            "trading volume", "market cap", "P/E ratio", "revenue growth",
            "quarterly earnings", "dividend yield", "beta coefficient",
            "analyst consensus", "price target", "market sentiment"
        ]
        
        if random.random() < 0.3:  # 30% chance to add financial term
            term = random.choice(financial_terms)
            template += f" with focus on {term}"
        
        return template
    
    def _generate_price_series(self, length: int, base_price: float, 
                              drift: float, volatility: float) -> np.ndarray:
        """Generate price series using geometric Brownian motion.
        
        Args:
            length: Length of the series
            base_price: Starting price
            drift: Annual drift rate
            volatility: Annual volatility
            
        Returns:
            Array of prices
        """
        dt = 1/252  # Daily timestep
        prices = np.zeros(length)
        prices[0] = base_price
        
        for i in range(1, length):
            # Geometric Brownian motion
            shock = np.random.normal(0, np.sqrt(dt))
            prices[i] = prices[i-1] * np.exp((drift - 0.5 * volatility**2) * dt + 
                                           volatility * shock)
            
            # Add occasional jumps
            if random.random() < self.market_config.jump_probability:
                jump = np.random.normal(0, self.market_config.jump_size)
                prices[i] *= (1 + jump)
        
        return prices
    
    def _generate_volume_series(self, length: int) -> np.ndarray:
        """Generate realistic volume series.
        
        Args:
            length: Length of the series
            
        Returns:
            Array of volumes
        """
        # Base volume with trend and seasonality
        base_volume = 1000000
        trend = np.linspace(0, 0.2, length)  # 20% increase over period
        seasonality = 0.1 * np.sin(2 * np.pi * np.arange(length) / 252)  # Annual cycle
        noise = np.random.normal(0, 0.3, length)
        
        volumes = base_volume * (1 + trend + seasonality + noise)
        volumes = np.maximum(volumes, base_volume * 0.1)  # Minimum volume
        
        return volumes.astype(int)
