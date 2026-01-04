"""Main training script for market sentiment analysis."""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.data import SyntheticDataGenerator, TextPreprocessor, MarketDataLoader
from src.models import FinBERTSentimentModel, BaselineSentimentModel
from src.utils.core import setup_logging, set_seed, get_device
from src.utils.evaluation import Evaluator


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main training pipeline for market sentiment analysis.
    
    Args:
        config: Hydra configuration object
    """
    # Setup logging
    logger = setup_logging(config.logging.level, config.logging.log_dir)
    logger.info("Starting Market Sentiment Analysis Training Pipeline")
    
    # Set random seed for reproducibility
    set_seed(config.seed)
    
    # Create output directories
    output_dir = Path("assets")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    
    # Initialize components
    logger.info("Initializing data generators and models...")
    
    # Data generation
    data_generator = SyntheticDataGenerator(config.data)
    text_preprocessor = TextPreprocessor()
    market_loader = MarketDataLoader(config.data)
    
    # Models
    finbert_model = FinBERTSentimentModel(config)
    baseline_models = {
        "textblob": BaselineSentimentModel("textblob"),
        "vader": BaselineSentimentModel("vader"),
        "logistic": BaselineSentimentModel("logistic"),
        "naive_bayes": BaselineSentimentModel("naive_bayes")
    }
    
    # Evaluator
    evaluator = Evaluator()
    
    # Generate synthetic data
    logger.info("Generating synthetic data...")
    text_data, market_data = data_generator.generate_combined_dataset()
    
    # Preprocess text data
    logger.info("Preprocessing text data...")
    text_data = text_preprocessor.preprocess_dataframe(text_data, "headline")
    
    # Add financial entities
    logger.info("Extracting financial entities...")
    entity_features = []
    for text in text_data["headline"]:
        entities = text_preprocessor.extract_financial_entities(text)
        entity_features.append(entities)
    
    entity_df = pd.DataFrame(entity_features)
    text_data = pd.concat([text_data, entity_df], axis=1)
    
    # Merge text and market data
    logger.info("Merging text and market data...")
    merged_data = pd.merge(
        text_data, 
        market_data, 
        on=["date", "symbol"], 
        how="inner"
    )
    
    # Create labels based on future returns
    logger.info("Creating sentiment labels...")
    merged_data = market_loader.create_labels(merged_data)
    
    # Remove rows with missing labels
    merged_data = merged_data.dropna(subset=["sentiment_label"])
    
    logger.info(f"Generated dataset with {len(merged_data)} samples")
    logger.info(f"Label distribution: {merged_data['sentiment_label'].value_counts().to_dict()}")
    
    # Split data
    logger.info("Splitting data into train/validation/test sets...")
    train_df, val_df, test_df = market_loader.split_data(
        merged_data,
        config.data.train_split,
        config.data.val_split,
        config.data.test_split
    )
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Train FinBERT model
    logger.info("Training FinBERT model...")
    train_loader, val_loader, test_loader = finbert_model.prepare_data(
        train_df, "headline", "sentiment_label"
    )
    
    # Train the model
    history = finbert_model.train(train_loader, val_loader)
    
    # Evaluate FinBERT
    logger.info("Evaluating FinBERT model...")
    finbert_metrics = finbert_model.evaluate(test_loader)
    
    # Get FinBERT predictions for financial evaluation
    test_texts = test_df["headline"].tolist()
    test_labels = test_df["sentiment_label"].tolist()
    finbert_predictions = finbert_model.predict(test_texts)
    
    # Train and evaluate baseline models
    logger.info("Training and evaluating baseline models...")
    baseline_results = {}
    
    for model_name, model in baseline_models.items():
        logger.info(f"Evaluating {model_name} model...")
        
        if model_name in ["logistic", "naive_bayes"]:
            # Train ML models
            model.fit(train_df["headline"].tolist(), train_df["sentiment_label"].tolist())
        
        # Get predictions
        predictions = model.predict(test_texts)
        
        # Evaluate
        metrics = evaluator.evaluate_classification(test_labels, predictions)
        baseline_results[model_name] = metrics
    
    # Compare all models
    logger.info("Comparing all models...")
    all_models = {"finbert": finbert_model, **baseline_models}
    
    comparison_df = evaluator.evaluate_model_comparison(
        all_models, test_texts, test_labels, test_df
    )
    
    # Save results
    logger.info("Saving results...")
    
    # Save comparison results
    comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
    
    # Save FinBERT metrics
    pd.DataFrame([finbert_metrics]).to_csv(output_dir / "finbert_metrics.csv", index=False)
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    
    # Create evaluation report
    logger.info("Creating evaluation report...")
    report = evaluator.create_evaluation_report(finbert_metrics, "FinBERT")
    logger.info(report)
    
    # Save report
    with open(output_dir / "evaluation_report.txt", "w") as f:
        f.write(report)
    
    # Plot confusion matrix
    logger.info("Creating confusion matrix plot...")
    evaluator.plot_confusion_matrix(test_labels, finbert_predictions, "FinBERT")
    
    logger.info("Training pipeline completed successfully!")
    logger.info(f"Results saved to: {output_dir}")


def run_baseline_comparison(config: DictConfig) -> None:
    """Run baseline model comparison only.
    
    Args:
        config: Configuration object
    """
    logger = setup_logging(config.logging.level, config.logging.log_dir)
    logger.info("Running baseline model comparison...")
    
    # Set random seed
    set_seed(config.seed)
    
    # Generate data
    data_generator = SyntheticDataGenerator(config.data)
    text_data, market_data = data_generator.generate_combined_dataset()
    
    # Merge data
    merged_data = pd.merge(text_data, market_data, on=["date", "symbol"], how="inner")
    market_loader = MarketDataLoader(config.data)
    merged_data = market_loader.create_labels(merged_data)
    merged_data = merged_data.dropna(subset=["sentiment_label"])
    
    # Split data
    train_df, val_df, test_df = market_loader.split_data(merged_data)
    
    # Initialize models
    baseline_models = {
        "textblob": BaselineSentimentModel("textblob"),
        "vader": BaselineSentimentModel("vader"),
        "logistic": BaselineSentimentModel("logistic"),
        "naive_bayes": BaselineSentimentModel("naive_bayes")
    }
    
    # Train ML models
    for model_name, model in baseline_models.items():
        if model_name in ["logistic", "naive_bayes"]:
            model.fit(train_df["headline"].tolist(), train_df["sentiment_label"].tolist())
    
    # Evaluate
    evaluator = Evaluator()
    test_texts = test_df["headline"].tolist()
    test_labels = test_df["sentiment_label"].tolist()
    
    comparison_df = evaluator.evaluate_model_comparison(
        baseline_models, test_texts, test_labels, test_df
    )
    
    # Save results
    output_dir = Path("assets")
    output_dir.mkdir(exist_ok=True)
    comparison_df.to_csv(output_dir / "baseline_comparison.csv", index=False)
    
    logger.info("Baseline comparison completed!")
    logger.info(f"Results saved to: {output_dir / 'baseline_comparison.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market Sentiment Analysis Training")
    parser.add_argument("--baseline-only", action="store_true", 
                       help="Run baseline comparison only")
    args = parser.parse_args()
    
    if args.baseline_only:
        # Load config manually for baseline comparison
        config = OmegaConf.load("configs/config.yaml")
        run_baseline_comparison(config)
    else:
        main()
