"""
Configuration management for AI Research Collection
Handles loading/saving of classification dictionaries and experiment settings.
"""

import yaml
import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR = "config"
DEFAULT_CONFIG_FILE = "ai_research_config.yaml"

class ConfigManager:
    """Manages configuration for boolean logic experiments and agent systems."""
    
    def __init__(self, config_dir: str = DEFAULT_CONFIG_DIR):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.config_file = self.config_dir / DEFAULT_CONFIG_FILE
        self.config = self._load_default_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            "boolean_logic": {
                "logical_words": [
                    "therefore", "thus", "hence", "consequently", "because", "since", "given",
                    "analysis", "data", "evidence", "proof", "logic", "reasoning", "conclusion",
                    "study", "research", "method", "approach", "theory", "hypothesis", "result"
                ],
                "emotional_words": [
                    "love", "hate", "fear", "joy", "anger", "sadness", "excitement", "passion",
                    "heart", "soul", "feeling", "emotion", "mood", "spirit", "warm", "cold"
                ],
                "logical_phrases": [
                    "in conclusion", "as a result", "it follows that", "based on evidence",
                    "the data shows", "research indicates", "studies suggest", "analysis reveals"
                ],
                "emotional_phrases": [
                    "i feel", "my heart", "deep down", "emotional connection", "gut feeling",
                    "from the heart", "touches my soul", "makes me feel"
                ]
            },
            "models": {
                "default_models": [
                    "microsoft/DialoGPT-medium",
                    "microsoft/DialoGPT-small"
                ],
                "model_cache_dir": "./model_cache"
            },
            "experiments": {
                "default_gates": ["AND", "OR", "XOR", "NOT"],
                "output_dir": "./experiment_results",
                "save_plots": True,
                "plot_format": "png"
            },
            "agents": {
                "default_faculties": ["RECALL", "CONSULT", "THINK", "TALK", "DONE"],
                "max_iterations": 10,
                "memory_limit": 1000
            },
            "cli": {
                "interactive_mode": True,
                "progress_bars": True,
                "log_level": "INFO"
            }
        }
    
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file."""
        if config_path:
            config_file = Path(config_path)
        else:
            config_file = self.config_file
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    if config_file.suffix == '.yaml' or config_file.suffix == '.yml':
                        loaded_config = yaml.safe_load(f)
                    else:
                        loaded_config = json.load(f)
                
                # Merge with defaults
                self._merge_config(self.config, loaded_config)
                logger.info(f"Loaded configuration from {config_file}")
                return self.config
            except Exception as e:
                logger.error(f"Error loading config from {config_file}: {e}")
                return self.config
        else:
            logger.info("No config file found, using defaults")
            return self.config
    
    def save_config(self, config_path: Optional[str] = None) -> bool:
        """Save current configuration to file."""
        if config_path:
            config_file = Path(config_path)
        else:
            config_file = self.config_file
        
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            logger.info(f"Saved configuration to {config_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving config to {config_file}: {e}")
            return False
    
    def _merge_config(self, base: Dict, override: Dict) -> None:
        """Recursively merge configuration dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'boolean_logic.logical_words')."""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
        logger.info(f"Set {key_path} = {value}")
    
    def add_to_list(self, key_path: str, item: Any) -> None:
        """Add item to a list configuration."""
        current_list = self.get(key_path, [])
        if isinstance(current_list, list):
            current_list.append(item)
            self.set(key_path, current_list)
        else:
            logger.error(f"Configuration key {key_path} is not a list")
    
    def extend_list(self, key_path: str, items: List[Any]) -> None:
        """Extend a list configuration with multiple items."""
        current_list = self.get(key_path, [])
        if isinstance(current_list, list):
            current_list.extend(items)
            self.set(key_path, current_list)
        else:
            logger.error(f"Configuration key {key_path} is not a list")
    
    def create_sample_config(self) -> None:
        """Create a sample configuration file for users to customize."""
        sample_config = {
            "# AI Research Collection Configuration": None,
            "# Customize these settings for your experiments": None,
            "boolean_logic": {
                "# Add your own logical/emotional words and phrases": None,
                "custom_logical_words": ["algorithm", "computation", "systematic"],
                "custom_emotional_words": ["compassionate", "empathetic", "heartfelt"],
                "# Classification sensitivity (0.0-1.0)": None,
                "classification_threshold": 0.5
            },
            "models": {
                "# Your preferred models for experiments": None,
                "preferred_models": [
                    "gpt2",
                    "microsoft/DialoGPT-medium"
                ],
                "# GPU settings": None,
                "use_gpu": True,
                "gpu_memory_fraction": 0.8
            },
            "experiments": {
                "# Default experiment settings": None,
                "num_iterations": 100,
                "random_seed": 42,
                "save_intermediate_results": True
            }
        }
        
        sample_file = self.config_dir / "sample_config.yaml"
        try:
            with open(sample_file, 'w') as f:
                yaml.dump(sample_config, f, default_flow_style=False, indent=2)
            logger.info(f"Created sample configuration at {sample_file}")
        except Exception as e:
            logger.error(f"Error creating sample config: {e}")


# Global config manager instance
config_manager = ConfigManager()

def get_config() -> ConfigManager:
    """Get the global configuration manager."""
    return config_manager

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file."""
    return config_manager.load_config(config_path)

def save_config(config_path: Optional[str] = None) -> bool:
    """Save configuration to file."""
    return config_manager.save_config(config_path)