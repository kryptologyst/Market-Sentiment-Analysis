"""Evaluation utilities for sentiment analysis models."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


class Evaluator:
    """Comprehensive evaluation class for sentiment analysis models.
    
    This class provides both ML metrics and financial performance metrics
    for evaluating sentiment analysis models.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the evaluator.
        
        Args:
            config: Configuration dictionary for evaluation parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def evaluate_classification(self, y_true: List[str], y_pred: List[str], 
                               y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Evaluate classification performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary containing classification metrics
        """
        # Convert labels to integers for some metrics
        label_map = {"negative": 0, "neutral": 1, "positive": 2}
        y_true_int = [label_map[label] for label in y_true]
        y_pred_int = [label_map[label] for label in y_pred]
        
        metrics = {}
        
        # Basic classification metrics
        metrics["accuracy"] = accuracy_score(y_true_int, y_pred_int)
        metrics["precision_macro"] = precision_score(y_true_int, y_pred_int, average="macro")
        metrics["recall_macro"] = recall_score(y_true_int, y_pred_int, average="macro")
        metrics["f1_macro"] = f1_score(y_true_int, y_pred_int, average="macro")
        metrics["f1_weighted"] = f1_score(y_true_int, y_pred_int, average="weighted")
        
        # Per-class metrics
        precision_per_class = precision_score(y_true_int, y_pred_int, average=None)
        recall_per_class = recall_score(y_true_int, y_pred_int, average=None)
        f1_per_class = f1_score(y_true_int, y_pred_int, average=None)
        
        for i, label in enumerate(["negative", "neutral", "positive"]):
            metrics[f"precision_{label}"] = precision_per_class[i]
            metrics[f"recall_{label}"] = recall_per_class[i]
            metrics[f"f1_{label}"] = f1_per_class[i]
        
        # Confusion matrix
        cm = confusion_matrix(y_true_int, y_pred_int)
        metrics["confusion_matrix"] = cm.tolist()
        
        # AUC metrics (if probabilities provided)
        if y_proba is not None:
            try:
                # Multi-class AUC
                metrics["auc_roc"] = roc_auc_score(y_true_int, y_proba, multi_class="ovr")
                metrics["auc_pr"] = average_precision_score(y_true_int, y_proba, average="macro")
            except Exception as e:
                self.logger.warning(f"Could not calculate AUC metrics: {e}")
        
        return metrics
    
    def evaluate_financial_performance(self, df: pd.DataFrame, 
                                     predictions: List[str],
                                     price_column: str = "close",
                                     date_column: str = "date",
                                     symbol_column: str = "symbol") -> Dict[str, float]:
        """Evaluate financial performance of sentiment-based trading strategy.
        
        Args:
            df: DataFrame containing market data
            predictions: Sentiment predictions
            price_column: Name of price column
            date_column: Name of date column
            symbol_column: Name of symbol column
            
        Returns:
            Dictionary containing financial metrics
        """
        df = df.copy()
        df["sentiment_pred"] = predictions
        
        # Create trading signals
        df["signal"] = df["sentiment_pred"].map({
            "positive": 1,  # Buy
            "negative": -1,  # Sell
            "neutral": 0  # Hold
        })
        
        # Calculate returns
        df["returns"] = df.groupby(symbol_column)[price_column].pct_change()
        
        # Calculate strategy returns
        df["strategy_returns"] = df["signal"].shift(1) * df["returns"]
        
        # Remove NaN values
        df = df.dropna(subset=["strategy_returns"])
        
        if len(df) == 0:
            self.logger.warning("No valid data for financial evaluation")
            return {}
        
        # Calculate financial metrics
        metrics = {}
        
        # Basic return metrics
        total_return = (1 + df["strategy_returns"]).prod() - 1
        metrics["total_return"] = total_return
        
        # Annualized metrics
        days = (df[date_column].max() - df[date_column].min()).days
        years = days / 365.25
        if years > 0:
            metrics["annualized_return"] = (1 + total_return) ** (1/years) - 1
        
        # Risk metrics
        returns_std = df["strategy_returns"].std()
        metrics["volatility"] = returns_std * np.sqrt(252)  # Annualized
        
        # Sharpe ratio
        if metrics["volatility"] > 0 and "annualized_return" in metrics:
            risk_free_rate = 0.02  # Assume 2% risk-free rate
            metrics["sharpe_ratio"] = (metrics["annualized_return"] - risk_free_rate) / metrics["volatility"]
        
        # Sortino ratio
        downside_returns = df["strategy_returns"][df["strategy_returns"] < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std() * np.sqrt(252)
            if downside_std > 0 and "annualized_return" in metrics:
                metrics["sortino_ratio"] = (metrics["annualized_return"] - risk_free_rate) / downside_std
        
        # Maximum drawdown
        cumulative_returns = (1 + df["strategy_returns"]).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        metrics["max_drawdown"] = drawdown.min()
        
        # Calmar ratio
        if metrics["max_drawdown"] != 0 and "annualized_return" in metrics:
            metrics["calmar_ratio"] = metrics["annualized_return"] / abs(metrics["max_drawdown"])
        
        # Hit rate
        positive_returns = df["strategy_returns"] > 0
        metrics["hit_rate"] = positive_returns.mean()
        
        # Average trade metrics
        trades = df[df["signal"] != 0]
        if len(trades) > 0:
            metrics["num_trades"] = len(trades)
            metrics["avg_trade_return"] = trades["strategy_returns"].mean()
            metrics["win_rate"] = (trades["strategy_returns"] > 0).mean()
        
        return metrics
    
    def evaluate_model_comparison(self, models: Dict[str, any], 
                                 test_texts: List[str], 
                                 test_labels: List[str],
                                 test_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Compare multiple models on the same test set.
        
        Args:
            models: Dictionary of model_name -> model_instance
            test_texts: Test texts
            test_labels: Test labels
            test_df: Test DataFrame for financial evaluation (optional)
            
        Returns:
            DataFrame containing comparison results
        """
        results = []
        
        for model_name, model in models.items():
            self.logger.info(f"Evaluating model: {model_name}")
            
            # Get predictions
            predictions = model.predict(test_texts)
            
            # Get probabilities if available
            try:
                probabilities = model.predict_proba(test_texts)
            except:
                probabilities = None
            
            # Calculate ML metrics
            ml_metrics = self.evaluate_classification(test_labels, predictions, probabilities)
            
            # Calculate financial metrics if market data available
            financial_metrics = {}
            if test_df is not None:
                financial_metrics = self.evaluate_financial_performance(
                    test_df, predictions
                )
            
            # Combine metrics
            combined_metrics = {**ml_metrics, **financial_metrics}
            combined_metrics["model"] = model_name
            
            results.append(combined_metrics)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results)
        
        # Sort by accuracy (or another primary metric)
        if "accuracy" in comparison_df.columns:
            comparison_df = comparison_df.sort_values("accuracy", ascending=False)
        
        return comparison_df
    
    def create_evaluation_report(self, metrics: Dict[str, float], 
                               model_name: str = "Model") -> str:
        """Create a formatted evaluation report.
        
        Args:
            metrics: Dictionary of metrics
            model_name: Name of the model
            
        Returns:
            Formatted report string
        """
        report = f"\n{'='*50}\n"
        report += f"EVALUATION REPORT: {model_name}\n"
        report += f"{'='*50}\n\n"
        
        # ML Metrics
        report += "MACHINE LEARNING METRICS:\n"
        report += "-" * 30 + "\n"
        
        ml_metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
        for metric in ml_metrics:
            if metric in metrics:
                report += f"{metric.replace('_', ' ').title()}: {metrics[metric]:.4f}\n"
        
        # Financial Metrics
        if any(key.startswith(("total_return", "sharpe", "max_drawdown", "hit_rate")) for key in metrics.keys()):
            report += "\nFINANCIAL PERFORMANCE METRICS:\n"
            report += "-" * 30 + "\n"
            
            financial_metrics = ["total_return", "annualized_return", "sharpe_ratio", 
                               "sortino_ratio", "max_drawdown", "calmar_ratio", "hit_rate"]
            for metric in financial_metrics:
                if metric in metrics:
                    report += f"{metric.replace('_', ' ').title()}: {metrics[metric]:.4f}\n"
        
        report += f"\n{'='*50}\n"
        
        return report
    
    def plot_confusion_matrix(self, y_true: List[str], y_pred: List[str], 
                             model_name: str = "Model") -> None:
        """Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Convert labels to integers
        label_map = {"negative": 0, "neutral": 1, "positive": 2}
        y_true_int = [label_map[label] for label in y_true]
        y_pred_int = [label_map[label] for label in y_pred]
        
        # Create confusion matrix
        cm = confusion_matrix(y_true_int, y_pred_int)
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                   xticklabels=["Negative", "Neutral", "Positive"],
                   yticklabels=["Negative", "Neutral", "Positive"])
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()
