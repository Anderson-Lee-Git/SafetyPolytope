"""
Stage 5: Steering Evaluation
Evaluates polytope effectiveness through steering experiments.
"""

import logging
import os
import subprocess
from typing import Any, Dict


class SteeringEvaluationStage:
    """Stage 5: Evaluate polytope effectiveness with steering"""

    def __init__(self, config: Dict[str, Any], harmbench_path: str):
        self.config = config
        self.harmbench_path = harmbench_path
        self.logger = logging.getLogger(__name__)

    def run(self, model_name: str, polytope_model_path: str) -> str:
        """
        Evaluate polytope effectiveness through steering

        Args:
            model_name: Name of the model
            polytope_model_path: Path to trained polytope model

        Returns:
            Path to evaluation results
        """
        self.logger.info(f"Starting steering evaluation for {model_name}")

        # Change to HarmBench directory
        original_dir = os.getcwd()
        os.chdir(self.harmbench_path)

        try:
            # Run steering evaluation
            results_path = self._run_steering_evaluation(
                model_name, polytope_model_path
            )

            return results_path

        finally:
            os.chdir(original_dir)

    def _run_steering_evaluation(
        self, model_name: str, polytope_model_path: str
    ) -> str:
        """
        Run steering evaluation using run_steering.sh

        Args:
            model_name: Name of the model
            polytope_model_path: Path to polytope model

        Returns:
            Path to evaluation results
        """
        self.logger.info(f"Running steering evaluation for {model_name}")

        # Get model-specific steering configuration
        model_config = None
        for model in self.config.get("models", []):
            if model["name"] == model_name:
                model_config = model
                break

        if not model_config:
            raise RuntimeError(
                f"Model configuration not found for {model_name}"
            )

        steering_config = model_config.get("steering", {})

        # Prepare environment variables
        env = os.environ.copy()
        env["MODEL_NAME"] = model_name
        env["MODEL_PATH"] = model_config["path"]
        env["POLYTOPE_MODEL_PATH"] = polytope_model_path
        env["LAMBDA_WEIGHT"] = str(steering_config.get("lambda_weight", 1.0))
        env["SAFE_VIOLATION_WEIGHT"] = str(
            steering_config.get("safe_violation_weight", 0.0)
        )
        env["STEER_LAYER"] = str(steering_config.get("steer_layer", 20))

        # Choose appropriate steering script based on model
        if "mistral" in model_name.lower():
            script_name = "run_steering_mistral.sh"
        else:
            script_name = "run_steering.sh"

        # Run steering evaluation
        cmd = ["bash", f"scripts/{script_name}"]
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)

        if result.returncode != 0:
            self.logger.error(f"Steering evaluation failed: {result.stderr}")
            raise RuntimeError(f"Steering evaluation failed: {result.stderr}")

        self.logger.info(
            f"Successfully completed steering evaluation for {model_name}"
        )

        # Return path to evaluation results
        return f"./results/steering_evaluation_{model_name}"
