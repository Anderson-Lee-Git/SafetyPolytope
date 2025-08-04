"""
Stage 5: Steering Evaluation
Evaluates polytope effectiveness through steering experiments using HarmBench's step_2_and_3.sh script.
"""

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

from ..utils.config_utils import get_harmbench_model_name


class SteeringEvaluationStage:
    """Stage 5: Evaluate polytope effectiveness with steering"""

    def __init__(self, config: Dict[str, Any], harmbench_path: str):
        self.config = config
        self.harmbench_path = Path(harmbench_path)
        self.logger = logging.getLogger(__name__)

        # Get execution mode and configuration
        steering_config = config.get("steering_evaluation", {})
        self.part_size = steering_config.get("part_size", 500)
        self.execution_mode = steering_config.get("execution_mode", "local")

    def run(
        self,
        model_name: str,
        polytope_model_path: str,
        model_config: Dict[str, Any],
    ) -> None:
        """
        Evaluate polytope effectiveness through steering

        Args:
            model_name: Name of the model
            polytope_model_path: Path to trained polytope model
            model_config: Model-specific configuration
        """
        self.logger.info(f"Starting steering evaluation for {model_name}")

        try:
            # Validate inputs
            self._validate_inputs(
                model_name, polytope_model_path, model_config
            )

            # Get HarmBench model name
            harmbench_model_name = get_harmbench_model_name(model_name)

            # Get evaluation methods from steering config
            steering_config = model_config.get("steering", {})
            evaluation_methods = steering_config.get("evaluation_methods", [])

            if not evaluation_methods:
                raise ValueError(
                    f"No evaluation methods configured for {model_name}"
                )

            # Create timestamp for unique identification
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            exp_ident = f"steering_{timestamp}"

            # Execute the step_2_and_3.sh command
            self._run_harmbench_script(
                harmbench_model_name,
                polytope_model_path,
                evaluation_methods,
                exp_ident,
            )

            self.logger.info("Steering evaluation completed")

        except Exception as e:
            self.logger.error(f"Steering evaluation failed: {e}")
            raise

    def _validate_inputs(
        self,
        model_name: str,
        polytope_model_path: str,
        model_config: Dict[str, Any],
    ) -> None:
        """Validate inputs before running evaluation."""
        if not Path(polytope_model_path).exists():
            raise FileNotFoundError(
                f"Polytope model not found: {polytope_model_path}"
            )

        if not self.harmbench_path.exists():
            raise FileNotFoundError(
                f"HarmBench path not found: {self.harmbench_path}"
            )

        if "steering" not in model_config:
            raise ValueError(
                f"Steering configuration missing for model: {model_name}"
            )

        if "attack_methods" not in model_config:
            raise ValueError(
                f"Attack methods not configured for model: {model_name}"
            )

    def _run_harmbench_script(
        self,
        harmbench_model_name: str,
        polytope_model_path: str,
        evaluation_methods: list,
        exp_ident: str,
    ) -> None:
        """
        Run HarmBench's step_2_and_3.sh script to perform generation and evaluation.
        The script handles both local and slurm execution modes internally.

        Args:
            harmbench_model_name: HarmBench-compatible model name
            polytope_model_path: Path to polytope weights
            evaluation_methods: List of evaluation methods
            exp_ident: Experiment identifier for output files
        """
        # Convert evaluation methods list to comma-separated string
        algorithms = ",".join(evaluation_methods)

        # Build command for step_2_and_3.sh
        script_path = self.harmbench_path / "scripts" / "step_2_and_3.sh"

        # Parameters for the script:
        # $1: MODEL
        # $2: MODE (local or sbatch)
        # $3: ALGORITHMS (comma-separated)
        # $4: EXP_IDENT (optional, for filename)
        # $5: PART_SIZE (default 500)
        # $6: POLYTOPE_PATH (path to polytope weights)
        # $7: USE_DEFENSE (False for polytope steering)
        # $8: DEFENSE_METHOD (None for polytope steering)

        cmd = [
            "bash",
            str(script_path),
            harmbench_model_name,
            self.execution_mode,  # Pass execution mode to the script
            algorithms,
            exp_ident,
            str(self.part_size),
            polytope_model_path,
            "False",  # USE_DEFENSE
            "None",  # DEFENSE_METHOD
        ]

        # Change to HarmBench directory for execution
        original_dir = os.getcwd()
        os.chdir(self.harmbench_path)

        try:
            # Run the script - it handles local vs slurm internally
            self.logger.info(f"Running HarmBench script: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                self.logger.error(f"Script execution failed: {result.stderr}")
                raise RuntimeError(f"HarmBench script failed: {result.stderr}")

            self.logger.info("HarmBench script completed successfully")

            # Log the stdout for debugging
            if result.stdout:
                self.logger.debug(f"Script output:\n{result.stdout}")

        finally:
            os.chdir(original_dir)
