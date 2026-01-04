"""Market data loading and processing utilities."""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf
from omegaconf import DictConfig


class MarketDataLoader:
    """Loads and processes market data for sentiment analysis.
    
    This class handles downloading, cleaning, and processing of market data
    from various sources including Yahoo Finance and synthetic data.
    """
    
    def __init__(self, config: DictConfig):
        """Initialize the market data loader.
        
        Args:
            config: Configuration object containing data parameters
        """
        self.config = config
        self.symbols = config.symbols
        self.start_date = config.start_date
        self.end_date = config.end_date
    
    def load_yahoo_data(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """Load market data from Yahoo Finance.
        
        Args:
            symbols: List of symbols to load (defaults to config symbols)
            
        Returns:
            DataFrame containing OHLCV data
        """
        if symbols is None:
            symbols = self.symbols
        
        try:
            # Download data for all symbols
            data = yf.download(
                symbols,
                start=self.start_date,
                end=self.end_date,
                group_by="ticker",
                auto_adjust=True,
                prepost=True,
                threads=True
            )
            
            # Process multi-level columns
            if len(symbols) == 1:
                # Single symbol case
                df = data.reset_index()
                df["symbol"] = symbols[0]
                df = df.rename(columns={"Date": "date"})
            else:
                # Multiple symbols case
                df_list = []
                for symbol in symbols:
                    if symbol in data.columns.get_level_values(0):
                        symbol_data = data[symbol].reset_index()
                        symbol_data["symbol"] = symbol
                        symbol_data = symbol_data.rename(columns={"Date": "date"})
                        df_list.append(symbol_data)
                
                df = pd.concat(df_list, ignore_index=True)
            
            # Clean column names
            df.columns = df.columns.str.lower()
            
            # Add technical indicators
            df = self._add_technical_indicators(df)
            
            return df
            
        except Exception as e:
            print(f"Error loading Yahoo Finance data: {e}")
            return pd.DataFrame()
    
    def load_synthetic_data(self) -> pd.DataFrame:
        """Load synthetic market data.
        
        Returns:
            DataFrame containing synthetic OHLCV data
        """
        from .synthetic_data import SyntheticDataGenerator
        
        generator = SyntheticDataGenerator(self.config)
        market_data = generator.generate_market_data(
            self.symbols, self.start_date, self.end_date
        )
        
        # Add technical indicators
        market_data = self._add_technical_indicators(market_data)
        
        return market_data
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to market data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional technical indicators
        """
        df = df.copy()
        
        # Sort by symbol and date
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        
        # Calculate returns
        df["price_change_1d"] = df.groupby("symbol")["close"].pct_change()
        df["price_change_7d"] = df.groupby("symbol")["close"].pct_change(periods=7)
        
        # Calculate volume ratio (current vs 20-day average)
        df["volume_20d_avg"] = df.groupby("symbol")["volume"].rolling(20).mean().reset_index(0, drop=True)
        df["volume_ratio"] = df["volume"] / df["volume_20d_avg"]
        
        # Calculate volatility (20-day rolling standard deviation of returns)
        df["volatility_7d"] = df.groupby("symbol")["price_change_1d"].rolling(7).std().reset_index(0, drop=True)
        df["volatility_20d"] = df.groupby("symbol")["price_change_1d"].rolling(20).std().reset_index(0, drop=True)
        
        # Calculate RSI (14-day)
        df["rsi_14d"] = self._calculate_rsi(df["close"], 14)
        
        # Calculate moving averages
        df["ma_5d"] = df.groupby("symbol")["close"].rolling(5).mean().reset_index(0, drop=True)
        df["ma_20d"] = df.groupby("symbol")["close"].rolling(20).mean().reset_index(0, drop=True)
        df["ma_50d"] = df.groupby("symbol")["close"].rolling(50).mean().reset_index(0, drop=True)
        
        # Price relative to moving averages
        df["price_vs_ma5"] = df["close"] / df["ma_5d"] - 1
        df["price_vs_ma20"] = df["close"] / df["ma_20d"] - 1
        df["price_vs_ma50"] = df["close"] / df["ma_50d"] - 1
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Series of prices
            window: RSI calculation window
            
        Returns:
            Series of RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def create_labels(self, df: pd.DataFrame, horizon: int = 1, 
                     threshold: float = 0.02) -> pd.DataFrame:
        """Create sentiment labels based on future price movements.
        
        Args:
            df: DataFrame with market data
            horizon: Number of days ahead to look for price movement
            threshold: Minimum price change threshold for labeling
            
        Returns:
            DataFrame with sentiment labels
        """
        df = df.copy()
        
        # Calculate future returns
        df["future_return"] = df.groupby("symbol")["close"].shift(-horizon) / df["close"] - 1
        
        # Create sentiment labels based on returns
        def label_sentiment(return_val):
            if pd.isna(return_val):
                return None
            elif return_val > threshold:
                return "positive"
            elif return_val < -threshold:
                return "negative"
            else:
                return "neutral"
        
        df["sentiment_label"] = df["future_return"].apply(label_sentiment)
        
        return df
    
    def split_data(self, df: pd.DataFrame, 
                  train_split: float = 0.7, 
                  val_split: float = 0.15, 
                  test_split: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/validation/test sets using time-based splits.
        
        Args:
            df: DataFrame to split
            train_split: Fraction for training set
            val_split: Fraction for validation set
            test_split: Fraction for test set
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Sort by date to ensure proper time-based splitting
        df = df.sort_values("date").reset_index(drop=True)
        
        # Calculate split indices
        n_samples = len(df)
        train_end = int(n_samples * train_split)
        val_end = int(n_samples * (train_split + val_split))
        
        # Split data
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        return train_df, val_df, test_df
