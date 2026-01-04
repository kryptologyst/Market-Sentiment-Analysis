# Market Sentiment Analysis

**Advanced Market Sentiment Analysis using NLP for Financial News and Social Media**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## ⚠️ IMPORTANT DISCLAIMER

**THIS IS A RESEARCH AND EDUCATIONAL PROJECT ONLY**

- This tool is **NOT providing investment advice**
- Results may be inaccurate and should not be used for trading decisions
- Backtests are hypothetical and do not guarantee future performance
- Always consult with qualified financial professionals before making investment decisions
- Use at your own risk - the authors are not responsible for any financial losses

## Overview

This project implements advanced sentiment analysis techniques specifically designed for financial text data. It combines state-of-the-art NLP models (FinBERT) with traditional machine learning approaches to analyze sentiment in financial news, social media, and other text sources.

### Key Features

- **Multiple Sentiment Models**: FinBERT, TextBlob, VADER, Logistic Regression, Naive Bayes
- **Financial-Specific Preprocessing**: Tailored text cleaning for financial terminology
- **Synthetic Data Generation**: Realistic financial news and market data for testing
- **Comprehensive Evaluation**: Both ML metrics and financial performance metrics
- **Interactive Demo**: Streamlit web application for real-time analysis
- **Production-Ready Structure**: Modular design with proper configuration management

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Market-Sentiment-Analysis.git
   cd Market-Sentiment-Analysis
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -e .
   ```

4. **Install development dependencies** (optional):
   ```bash
   pip install -e ".[dev]"
   ```

## Quick Start

### 1. Run the Training Pipeline

```bash
# Train all models with synthetic data
python scripts/train.py

# Run baseline comparison only
python scripts/train.py --baseline-only
```

### 2. Launch the Interactive Demo

```bash
streamlit run demo/streamlit_app.py
```

### 3. Use the API (if implemented)

```python
from src.models import FinBERTSentimentModel
from src.utils.core import load_config

# Load configuration
config = load_config("configs/config.yaml")

# Initialize model
model = FinBERTSentimentModel(config)

# Analyze sentiment
text = "Apple stock surges after strong quarterly earnings"
sentiment = model.predict(text)
print(f"Sentiment: {sentiment}")
```

## Project Structure

```
market-sentiment-analysis/
├── src/                          # Source code
│   ├── data/                     # Data handling modules
│   │   ├── synthetic_data.py     # Synthetic data generation
│   │   ├── text_preprocessing.py # Text preprocessing utilities
│   │   └── market_data.py       # Market data loading
│   ├── models/                   # Sentiment analysis models
│   │   ├── finbert_model.py     # FinBERT implementation
│   │   ├── baseline_models.py   # Baseline models
│   │   └── ensemble_model.py    # Ensemble methods
│   ├── utils/                    # Utility functions
│   │   ├── core.py              # Core utilities
│   │   └── evaluation.py        # Evaluation metrics
│   └── __init__.py
├── configs/                      # Configuration files
│   ├── config.yaml              # Main configuration
│   ├── data/                    # Data-specific configs
│   ├── model/                   # Model-specific configs
│   └── evaluation/              # Evaluation configs
├── scripts/                      # Training and evaluation scripts
│   └── train.py                 # Main training script
├── demo/                         # Demo applications
│   └── streamlit_app.py         # Streamlit demo
├── tests/                        # Unit tests
├── assets/                       # Output artifacts
│   ├── models/                  # Saved models
│   └── plots/                   # Generated plots
├── data/                         # Data storage
│   ├── raw/                     # Raw data
│   ├── processed/               # Processed data
│   └── external/                # External data sources
├── logs/                         # Log files
├── pyproject.toml               # Project configuration
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Configuration

The project uses Hydra for configuration management. Key configuration files:

- `configs/config.yaml`: Main configuration
- `configs/data/synthetic.yaml`: Synthetic data generation parameters
- `configs/model/finbert.yaml`: FinBERT model configuration
- `configs/evaluation/standard.yaml`: Evaluation metrics configuration

### Key Configuration Parameters

```yaml
# Data configuration
data:
  symbols: ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
  start_date: "2020-01-01"
  end_date: "2023-12-31"
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15

# Model configuration
model:
  name: "finbert"
  max_length: 512
  batch_size: 16
  learning_rate: 2e-5
  num_epochs: 3

# Evaluation configuration
evaluation:
  metrics:
    ml: ["accuracy", "f1_macro", "auc_roc"]
    financial: ["sharpe_ratio", "max_drawdown", "hit_rate"]
```

## Models

### FinBERT
- **Description**: Pre-trained BERT model fine-tuned on financial text
- **Architecture**: Transformer-based with attention mechanisms
- **Use Case**: High-accuracy sentiment analysis for financial news
- **Performance**: Best overall accuracy on financial text

### Baseline Models
- **TextBlob**: Rule-based sentiment analysis using predefined lexicons
- **VADER**: Valence Aware Dictionary for social media text
- **Logistic Regression**: Traditional ML with engineered features
- **Naive Bayes**: Probabilistic classifier with independence assumptions

## Data

### Synthetic Data
The project generates realistic synthetic financial data including:
- Financial news headlines with sentiment labels
- Market data (OHLCV) with technical indicators
- Temporal relationships between sentiment and price movements

### Real Data Support
- Yahoo Finance integration for real market data
- Custom data loading for proprietary datasets
- Support for multiple data formats (CSV, Parquet, JSON)

## Evaluation Metrics

### Machine Learning Metrics
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class and macro-averaged metrics
- **AUC-ROC**: Area under the ROC curve
- **AUC-PR**: Area under the Precision-Recall curve
- **Confusion Matrix**: Detailed classification breakdown

### Financial Performance Metrics
- **Total Return**: Cumulative strategy returns
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: Return-to-drawdown ratio
- **Hit Rate**: Percentage of profitable trades

## Usage Examples

### Basic Sentiment Analysis

```python
from src.models import FinBERTSentimentModel
from src.utils.core import load_config

# Load configuration
config = load_config("configs/config.yaml")

# Initialize model
model = FinBERTSentimentModel(config)

# Analyze single text
text = "Tesla stock rallies after strong Q4 earnings beat"
sentiment = model.predict(text)
print(f"Sentiment: {sentiment}")

# Analyze multiple texts
texts = [
    "Apple announces record quarterly revenue",
    "Market crashes amid economic uncertainty",
    "Fed raises interest rates by 0.25%"
]
sentiments = model.predict(texts)
print(f"Sentiments: {sentiments}")
```

### Batch Processing

```python
import pandas as pd
from src.data import SyntheticDataGenerator

# Generate synthetic data
generator = SyntheticDataGenerator(config.data)
text_data, market_data = generator.generate_combined_dataset()

# Process batch
predictions = model.predict(text_data["headline"].tolist())
text_data["predicted_sentiment"] = predictions

# Evaluate performance
from src.utils.evaluation import Evaluator
evaluator = Evaluator()
metrics = evaluator.evaluate_classification(
    text_data["sentiment_label"].tolist(),
    predictions
)
print(f"Accuracy: {metrics['accuracy']:.3f}")
```

### Custom Model Training

```python
# Prepare data
train_loader, val_loader, test_loader = model.prepare_data(
    train_df, "headline", "sentiment_label"
)

# Train model
history = model.train(train_loader, val_loader)

# Evaluate
metrics = model.evaluate(test_loader)
print(f"Test Accuracy: {metrics['accuracy']:.3f}")
```

## Demo Application

The Streamlit demo provides an interactive interface for:

1. **Data Overview**: Dataset statistics and visualizations
2. **Sentiment Analysis**: Real-time text analysis
3. **Trading Strategy**: Hypothetical trading simulation
4. **Model Comparison**: Performance comparison across models

### Launch Demo

```bash
streamlit run demo/streamlit_app.py
```

## Development

### Code Quality

The project uses several tools for code quality:

- **Black**: Code formatting
- **Ruff**: Linting and import sorting
- **MyPy**: Type checking
- **Pytest**: Unit testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_models.py
```

### Code Formatting

```bash
# Format code
black src/ scripts/ demo/

# Lint code
ruff check src/ scripts/ demo/

# Type checking
mypy src/
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write docstrings for all classes and functions
- Add unit tests for new functionality
- Update documentation for new features

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{market_sentiment_analysis,
  title={Market Sentiment Analysis: Advanced NLP for Financial Text},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Market-Sentiment-Analysis}
}
```

## Acknowledgments

- **FinBERT**: ProsusAI for the pre-trained financial BERT model
- **Transformers**: Hugging Face for the transformers library
- **Streamlit**: For the demo application framework
- **Financial Data**: Yahoo Finance for market data access

## Support

For questions, issues, or contributions:

- **Issues**: [GitHub Issues](https://github.com/example/market-sentiment-analysis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/example/market-sentiment-analysis/discussions)
- **Email**: research@example.com

## Changelog

### Version 1.0.0
- Initial release
- FinBERT sentiment analysis model
- Baseline model implementations
- Synthetic data generation
- Streamlit demo application
- Comprehensive evaluation metrics
- Production-ready project structure

---

**Remember: This is a research and educational tool only. Do not use for actual trading decisions.**
# Market-Sentiment-Analysis
