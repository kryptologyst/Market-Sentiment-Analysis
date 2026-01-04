"""Core utilities for market sentiment analysis project."""

import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def setup_logging(level: str = "INFO", log_dir: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to save log files
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("market_sentiment")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / "market_sentiment.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_preference: str = "auto") -> torch.device:
    """Get the best available device for computation.
    
    Args:
        device_preference: Device preference ("auto", "cuda", "mps", "cpu")
        
    Returns:
        PyTorch device object
    """
    if device_preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_preference)


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        OmegaConf configuration object
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, save_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object to save
        save_path: Path to save configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "w") as f:
        OmegaConf.save(config, f)


def create_directories(base_path: Union[str, Path], subdirs: list[str]) -> None:
    """Create directory structure for the project.
    
    Args:
        base_path: Base directory path
        subdirs: List of subdirectories to create
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    for subdir in subdirs:
        (base_path / subdir).mkdir(parents=True, exist_ok=True)


def validate_config(config: DictConfig) -> None:
    """Validate configuration parameters.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate data splits sum to 1
    splits = [config.data.train_split, config.data.val_split, config.data.test_split]
    if not np.isclose(sum(splits), 1.0, atol=1e-6):
        raise ValueError(f"Data splits must sum to 1.0, got {sum(splits)}")
    
    # Validate date range
    if config.data.start_date >= config.data.end_date:
        raise ValueError("Start date must be before end date")
    
    # Validate model parameters
    if config.model.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    
    if config.model.learning_rate <= 0:
        raise ValueError("Learning rate must be positive")


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: Union[str, Path]):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to main configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> DictConfig:
        """Load and resolve configuration."""
        config = OmegaConf.load(self.config_path)
        return OmegaConf.to_container(config, resolve=True)
    
    def _validate_config(self) -> None:
        """Validate loaded configuration."""
        validate_config(self.config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
        """
        OmegaConf.set(self.config, updates)
        self._validate_config()
    
    def save(self, save_path: Optional[Union[str, Path]] = None) -> None:
        """Save current configuration.
        
        Args:
            save_path: Path to save configuration (defaults to original path)
        """
        save_path = save_path or self.config_path
        save_config(self.config, save_path)
