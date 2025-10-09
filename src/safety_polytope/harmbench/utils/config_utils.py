"""
Configuration utilities for HarmBench pipeline
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load pipeline configuration from YAML file

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_model_config(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific model

    Args:
        config: Full pipeline configuration
        model_name: Name of the model

    Returns:
        Model-specific configuration
    """
    for model in config.get("models", []):
        if model["name"] == model_name:
            return model

    raise ValueError(f"Model configuration not found for: {model_name}")


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate pipeline configuration

    Args:
        config: Configuration dictionary to validate
    """
    required_sections = ["pipeline", "models", "slurm"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")

    # Validate pipeline section
    pipeline_config = config["pipeline"]
    required_pipeline_keys = ["harmbench_path", "safety_polytope_path"]
    for key in required_pipeline_keys:
        if key not in pipeline_config:
            raise ValueError(f"Missing required pipeline configuration: {key}")

    # Validate that paths exist
    harmbench_path = pipeline_config["harmbench_path"]
    safety_polytope_path = pipeline_config["safety_polytope_path"]

    if not os.path.exists(harmbench_path):
        raise ValueError(f"HarmBench path does not exist: {harmbench_path}")

    if not os.path.exists(safety_polytope_path):
        raise ValueError(f"Safety Polytope path does not exist: {safety_polytope_path}")

    # Validate models
    if not config.get("models"):
        raise ValueError("No models configured")

    for model in config["models"]:
        required_model_keys = ["name", "path", "attack_methods"]
        for key in required_model_keys:
            if key not in model:
                raise ValueError(f"Missing required model configuration: {key}")

        if not model["attack_methods"]:
            raise ValueError(f"No attack methods configured for model: {model['name']}")


def get_harmbench_model_name(model_name: str) -> str:
    """
    Transform model name to HarmBench convention.

    Args:
        model_name: Original model name

    Returns:
        HarmBench-formatted model name
    """
    name_lower = model_name.lower()

    if "llama" in name_lower:
        return "safe_llama2_7b"
    elif "mistral" in name_lower:
        return "safe_mistral_8b"
    elif "qwen" in name_lower:
        return "safe_qwen_1.5b"
    else:
        return f"safe_{model_name}"


def get_results_directory(
    harmbench_path: Path,
    method: str,
    model_name: str,
    transform_model_name: bool = True,
) -> Path:
    """
    Get results directory path for HarmBench method and model.

    Args:
        harmbench_path: Base HarmBench path
        method: Attack method name
        model_name: Model name
        transform_model_name: Whether to transform model name to HarmBench convention

    Returns:
        Results directory path
    """
    # Methods that use /default/ instead of /{model_name}/
    default_methods = ["DirectRequest", "HumanJailbreaks"]

    if method in default_methods:
        return harmbench_path / "results" / method / "default"
    else:
        if transform_model_name:
            model_name = get_harmbench_model_name(model_name)
        return harmbench_path / "results" / method / model_name


def get_method_path(
    harmbench_path: Path,
    method: str,
    model_name: str,
    subdirectory: str,
    transform_model_name: bool = True,
) -> Path:
    """
    Get the complete path for a given attack method and subdirectory.

    Args:
        harmbench_path: Base HarmBench path
        method: Attack method name
        model_name: Model name
        subdirectory: Subdirectory (e.g., 'test_cases', 'hidden_states', 'completions')
        transform_model_name: Whether to transform model name to HarmBench convention

    Returns:
        Complete path
    """
    results_dir = get_results_directory(
        harmbench_path, method, model_name, transform_model_name
    )
    return results_dir / subdirectory


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Setup logging configuration

    Args:
        config: Configuration dictionary
    """
    from loguru import logger

    log_config = config.get("logging", {})
    log_level = log_config.get("level", "INFO")
    log_dir = log_config.get("log_dir", "./logs")

    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # Configure loguru
    logger.remove()
    logger.add(
        os.path.join(log_dir, "pipeline.log"),
        level=log_level.upper(),
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
    logger.add(sys.stdout, level=log_level.upper())
