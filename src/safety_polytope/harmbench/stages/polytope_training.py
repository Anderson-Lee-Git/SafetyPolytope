"""
Stage 4: Polytope Training
Trains safety polytope constraints using processed hidden states.
"""

import logging
import os
import subprocess
from typing import Any, Dict, Optional


class PolytopeTrainingStage:
    """Stage 4: Train safety polytope constraints"""

    def __init__(self, config: Dict[str, Any], safety_polytope_path: str):
        self.config = config
        self.safety_polytope_path = safety_polytope_path
        self.logger = logging.getLogger(__name__)

    def run(
        self,
        model_name: str,
        processed_data_path: str,
        model_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Train polytope constraints

        Args:
            model_name: Name of the model
            processed_data_path: Path to processed hidden state data
            model_config: Model-specific configuration containing polytope training hyperparameters

        """
        self.logger.info(f"Starting polytope training for {model_name}")

        # Change to safety_polytope directory
        original_dir = os.getcwd()
        os.chdir(self.safety_polytope_path)

        try:
            # Train polytope
            self._train_polytope(model_name, processed_data_path, model_config)

        finally:
            os.chdir(original_dir)

    def _train_polytope(
        self,
        model_name: str,
        processed_data_path: str,
        model_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Train polytope using learn_polytope.py

        Args:
            model_name: Name of the model
            processed_data_path: Path to processed data
            model_config: Model-specific configuration containing polytope training hyperparameters

        """
        self.logger.info(f"Training polytope for {model_name}")

        # Get training configuration - use model-specific config if available, otherwise fall back to global config
        if model_config and "polytope_training" in model_config:
            training_config = model_config["polytope_training"]
            self.logger.info(
                f"Using model-specific polytope training config for {model_name}"
            )
        else:
            training_config = self.config.get("polytope_training", {})
            self.logger.info(
                f"Using global polytope training config for {model_name}"
            )

        execution_mode = self.config.get("polytope_training", {}).get(
            "execution_mode", "slurm"
        )

        # Prepare command arguments with model-specific hyperparameters using Hydra override syntax
        cmd = [
            "python",
            "src/safety_polytope/polytope/learn_polytope.py",
            "dataset=harmbench",
            f"dataset.hidden_states_path={processed_data_path}",
            f'seed={self.config.get("polytope_training", {}).get("seed", 5)}',
            f'dataset.num_epochs={self.config.get("polytope_training", {}).get("num_epochs", 1)}',
            f'dataset.num_phi={training_config.get("num_phi", 30)}',
            f'learning_rate={training_config.get("learning_rate", 0.01)}',
            f'batch_size={training_config.get("batch_size", 128)}',
            f'feature_dim={training_config.get("feature_dim", 16384)}',
            f'entropy_weight={training_config.get("entropy_weight", 1.0)}',
            f'unsafe_weight={training_config.get("unsafe_weight", 3.0)}',
            f'f_l1_weight={training_config.get("lambda_constraint", 1.0)}',
            f'margin={training_config.get("margin", 1.0)}',
            f"exp_ident=harmbench_polytope_{model_name}",
        ]

        if execution_mode == "slurm":
            cmd.append("--multirun")

        self.logger.info(
            f"Training with hyperparameters: learning_rate={training_config.get('learning_rate', 0.01)}, "
            f"batch_size={training_config.get('batch_size', 128)}, "
            f"feature_dim={training_config.get('feature_dim', 16384)}, "
            f"entropy_weight={training_config.get('entropy_weight', 1.0)}, "
            f"lambda_constraint={training_config.get('lambda_constraint', 1.0)}, "
            f"margin={training_config.get('margin', 1.0)}"
        )

        self.logger.info(f"Running command: {' '.join(cmd)}")

        # Run training
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            self.logger.error(f"Polytope training failed: {result.stderr}")
            raise RuntimeError(f"Polytope training failed: {result.stderr}")

        self.logger.info(f"Successfully trained polytope for {model_name}")
