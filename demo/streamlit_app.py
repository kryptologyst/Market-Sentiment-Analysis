"""Streamlit demo for market sentiment analysis."""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

from src.data import SyntheticDataGenerator, TextPreprocessor
from src.models import FinBERTSentimentModel, BaselineSentimentModel
from src.utils.core import load_config, set_seed


# Page configuration
st.set_page_config(
    page_title="Market Sentiment Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ IMPORTANT DISCLAIMER</h4>
    <p><strong>This is a research and educational demonstration only.</strong></p>
    <ul>
        <li>This tool is NOT providing investment advice</li>
        <li>Results may be inaccurate and should not be used for trading decisions</li>
        <li>Backtests are hypothetical and do not guarantee future performance</li>
        <li>Always consult with qualified financial professionals before making investment decisions</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">📈 Market Sentiment Analysis Demo</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model",
    ["FinBERT", "TextBlob", "VADER", "Logistic Regression", "Naive Bayes"],
    help="Choose the sentiment analysis model to use"
)

# Data source selection
data_source = st.sidebar.selectbox(
    "Data Source",
    ["Synthetic Data", "Real Market Data"],
    help="Choose between synthetic data or real market data"
)

# Symbol selection for real data
if data_source == "Real Market Data":
    symbols = st.sidebar.multiselect(
        "Select Stock Symbols",
        ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META", "NFLX"],
        default=["AAPL"],
        help="Select one or more stock symbols to analyze"
    )
else:
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]

# Date range for real data
if data_source == "Real Market Data":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
    with col2:
        end_date = st.date_input("End Date", value=pd.to_datetime("2023-12-31"))

# Initialize models (cached)
@st.cache_resource
def load_models():
    """Load and cache models."""
    try:
        config = load_config("configs/config.yaml")
        set_seed(config.seed)
        
        models = {
            "FinBERT": FinBERTSentimentModel(config),
            "TextBlob": BaselineSentimentModel("textblob"),
            "VADER": BaselineSentimentModel("vader"),
            "Logistic Regression": BaselineSentimentModel("logistic"),
            "Naive Bayes": BaselineSentimentModel("naive_bayes")
        }
        
        return models, config
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Load models
models, config = load_models()

if models is None:
    st.error("Failed to load models. Please check the configuration.")
    st.stop()

# Generate or load data
@st.cache_data
def generate_data(data_source: str, symbols: List[str], start_date=None, end_date=None):
    """Generate or load data based on source."""
    if data_source == "Synthetic Data":
        # Generate synthetic data
        data_generator = SyntheticDataGenerator(config.data)
        text_data, market_data = data_generator.generate_combined_dataset()
        
        # Merge data
        merged_data = pd.merge(text_data, market_data, on=["date", "symbol"], how="inner")
        
        # Add labels
        from src.data import MarketDataLoader
        market_loader = MarketDataLoader(config.data)
        merged_data = market_loader.create_labels(merged_data)
        merged_data = merged_data.dropna(subset=["sentiment_label"])
        
        return merged_data
    
    else:
        # Load real market data
        try:
            # Download market data
            market_data = yf.download(
                symbols,
                start=start_date,
                end=end_date,
                group_by="ticker"
            )
            
            # Process market data
            processed_data = []
            for symbol in symbols:
                if len(symbols) == 1:
                    symbol_data = market_data.reset_index()
                else:
                    symbol_data = market_data[symbol].reset_index()
                
                symbol_data["symbol"] = symbol
                symbol_data.columns = symbol_data.columns.str.lower()
                processed_data.append(symbol_data)
            
            market_df = pd.concat(processed_data, ignore_index=True)
            
            # Generate synthetic text data for demonstration
            data_generator = SyntheticDataGenerator(config.data)
            text_data, _ = data_generator.generate_combined_dataset()
            
            # Merge with market data
            merged_data = pd.merge(text_data, market_df, on=["date", "symbol"], how="inner")
            
            # Add labels
            from src.data import MarketDataLoader
            market_loader = MarketDataLoader(config.data)
            merged_data = market_loader.create_labels(merged_data)
            merged_data = merged_data.dropna(subset=["sentiment_label"])
            
            return merged_data
            
        except Exception as e:
            st.error(f"Error loading real market data: {e}")
            return pd.DataFrame()

# Generate data
with st.spinner("Generating data..."):
    data = generate_data(data_source, symbols, start_date if data_source == "Real Market Data" else None, 
                        end_date if data_source == "Real Market Data" else None)

if data.empty:
    st.error("No data available. Please check your configuration.")
    st.stop()

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🤖 Sentiment Analysis", "📈 Trading Strategy", "📋 Model Comparison"])

with tab1:
    st.header("Data Overview")
    
    # Data statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(data))
    
    with col2:
        st.metric("Unique Symbols", data["symbol"].nunique())
    
    with col3:
        st.metric("Date Range", f"{data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}")
    
    with col4:
        st.metric("Avg Text Length", f"{data['headline'].str.len().mean():.1f} chars")
    
    # Sentiment distribution
    st.subheader("Sentiment Distribution")
    sentiment_counts = data["sentiment_label"].value_counts()
    
    fig_sentiment = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Label Distribution",
        color_discrete_map={"positive": "#2E8B57", "neutral": "#FFD700", "negative": "#DC143C"}
    )
    st.plotly_chart(fig_sentiment, use_container_width=True)
    
    # Sample data
    st.subheader("Sample Data")
    st.dataframe(data[["date", "symbol", "headline", "sentiment_label", "close"]].head(10))

with tab2:
    st.header("Sentiment Analysis")
    
    # Text input for analysis
    st.subheader("Analyze Custom Text")
    custom_text = st.text_area(
        "Enter financial news text to analyze:",
        value="Apple stock surges after strong quarterly earnings beat expectations",
        height=100
    )
    
    if st.button("Analyze Sentiment"):
        if custom_text:
            # Get predictions from selected model
            model = models[model_type]
            
            try:
                prediction = model.predict(custom_text)
                probabilities = model.predict_proba(custom_text)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Predicted Sentiment", prediction.title())
                
                with col2:
                    # Create probability bar chart
                    prob_df = pd.DataFrame({
                        "Sentiment": ["Negative", "Neutral", "Positive"],
                        "Probability": probabilities
                    })
                    
                    fig_prob = px.bar(
                        prob_df,
                        x="Sentiment",
                        y="Probability",
                        title="Sentiment Probabilities",
                        color="Sentiment",
                        color_discrete_map={"Negative": "#DC143C", "Neutral": "#FFD700", "Positive": "#2E8B57"}
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error analyzing text: {e}")
    
    # Batch analysis
    st.subheader("Batch Analysis")
    
    # Select sample size
    sample_size = st.slider("Number of samples to analyze", 10, min(100, len(data)), 50)
    
    if st.button("Run Batch Analysis"):
        with st.spinner("Analyzing samples..."):
            # Sample data
            sample_data = data.sample(n=sample_size, random_state=42)
            
            # Get predictions
            model = models[model_type]
            predictions = model.predict(sample_data["headline"].tolist())
            
            # Calculate accuracy
            accuracy = (sample_data["sentiment_label"] == predictions).mean()
            
            # Display results
            st.metric("Accuracy", f"{accuracy:.2%}")
            
            # Confusion matrix
            from sklearn.metrics import confusion_matrix
            import numpy as np
            
            cm = confusion_matrix(sample_data["sentiment_label"], predictions)
            
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                title="Confusion Matrix",
                labels=dict(x="Predicted", y="Actual"),
                x=["Negative", "Neutral", "Positive"],
                y=["Negative", "Neutral", "Positive"]
            )
            st.plotly_chart(fig_cm, use_container_width=True)

with tab3:
    st.header("Trading Strategy Simulation")
    
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Disclaimer:</strong> This is a hypothetical trading simulation for research purposes only. 
        Past performance does not guarantee future results. Do not use this for actual trading decisions.
    </div>
    """, unsafe_allow_html=True)
    
    # Strategy parameters
    st.subheader("Strategy Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        initial_capital = st.number_input("Initial Capital ($)", value=100000, min_value=1000)
        transaction_cost = st.slider("Transaction Cost (%)", 0.0, 1.0, 0.1) / 100
    
    with col2:
        position_size = st.slider("Position Size (%)", 10, 100, 50) / 100
        rebalance_freq = st.selectbox("Rebalance Frequency", ["Daily", "Weekly", "Monthly"])
    
    if st.button("Run Trading Simulation"):
        with st.spinner("Running trading simulation..."):
            # Simple trading strategy based on sentiment
            model = models[model_type]
            
            # Get predictions for all data
            predictions = model.predict(data["headline"].tolist())
            
            # Create trading signals
            signal_map = {"positive": 1, "negative": -1, "neutral": 0}
            signals = [signal_map[pred] for pred in predictions]
            
            # Calculate strategy returns
            data_sim = data.copy()
            data_sim["signal"] = signals
            data_sim["returns"] = data_sim.groupby("symbol")["close"].pct_change()
            data_sim["strategy_returns"] = data_sim["signal"].shift(1) * data_sim["returns"]
            
            # Remove NaN values
            data_sim = data_sim.dropna(subset=["strategy_returns"])
            
            if len(data_sim) > 0:
                # Calculate performance metrics
                total_return = (1 + data_sim["strategy_returns"]).prod() - 1
                volatility = data_sim["strategy_returns"].std() * (252 ** 0.5)
                sharpe_ratio = (total_return * 252 - 0.02) / volatility if volatility > 0 else 0
                
                # Calculate drawdown
                cumulative_returns = (1 + data_sim["strategy_returns"]).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                max_drawdown = drawdown.min()
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Return", f"{total_return:.2%}")
                
                with col2:
                    st.metric("Annualized Volatility", f"{volatility:.2%}")
                
                with col3:
                    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                
                with col4:
                    st.metric("Max Drawdown", f"{max_drawdown:.2%}")
                
                # Plot equity curve
                fig_equity = go.Figure()
                fig_equity.add_trace(go.Scatter(
                    x=data_sim["date"],
                    y=cumulative_returns,
                    mode="lines",
                    name="Strategy",
                    line=dict(color="blue")
                ))
                
                fig_equity.update_layout(
                    title="Strategy Equity Curve",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig_equity, use_container_width=True)
                
                # Plot drawdown
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=data_sim["date"],
                    y=drawdown,
                    mode="lines",
                    name="Drawdown",
                    fill="tonexty",
                    line=dict(color="red")
                ))
                
                fig_dd.update_layout(
                    title="Strategy Drawdown",
                    xaxis_title="Date",
                    yaxis_title="Drawdown",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig_dd, use_container_width=True)

with tab4:
    st.header("Model Comparison")
    
    st.subheader("Performance Metrics")
    
    # Sample data for comparison
    sample_data = data.sample(n=min(200, len(data)), random_state=42)
    test_texts = sample_data["headline"].tolist()
    test_labels = sample_data["sentiment_label"].tolist()
    
    # Compare all models
    comparison_results = []
    
    for model_name, model in models.items():
        try:
            predictions = model.predict(test_texts)
            accuracy = (pd.Series(test_labels) == pd.Series(predictions)).mean()
            
            comparison_results.append({
                "Model": model_name,
                "Accuracy": accuracy,
                "F1 Score": "N/A",  # Simplified for demo
                "Precision": "N/A",
                "Recall": "N/A"
            })
        except Exception as e:
            st.warning(f"Error evaluating {model_name}: {e}")
    
    if comparison_results:
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.sort_values("Accuracy", ascending=False)
        
        # Display comparison table
        st.dataframe(comparison_df, use_container_width=True)
        
        # Plot comparison
        fig_comparison = px.bar(
            comparison_df,
            x="Model",
            y="Accuracy",
            title="Model Accuracy Comparison",
            color="Accuracy",
            color_continuous_scale="Viridis"
        )
        fig_comparison.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Model details
    st.subheader("Model Information")
    
    model_info = {
        "FinBERT": "Pre-trained BERT model fine-tuned on financial text. Uses transformer architecture with attention mechanisms.",
        "TextBlob": "Rule-based sentiment analysis using predefined sentiment lexicons and grammatical patterns.",
        "VADER": "Valence Aware Dictionary and sEntiment Reasoner - specifically designed for social media text.",
        "Logistic Regression": "Traditional machine learning model trained on engineered text features.",
        "Naive Bayes": "Probabilistic classifier based on Bayes' theorem with strong independence assumptions."
    }
    
    selected_model_info = st.selectbox("Select model for details:", list(model_info.keys()))
    st.info(model_info[selected_model_info])

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>Market Sentiment Analysis Demo - Research and Educational Use Only</p>
    <p>⚠️ This tool is NOT providing investment advice. Results may be inaccurate.</p>
</div>
""", unsafe_allow_html=True)
