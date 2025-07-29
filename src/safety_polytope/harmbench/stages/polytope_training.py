"""
Stage 4: Polytope Training
Trains safety polytope constraints using processed hidden states.
"""

import logging
import os
import subprocess
from typing import Any, Dict


class PolytopeTrainingStage:
    """Stage 4: Train safety polytope constraints"""

    def __init__(self, config: Dict[str, Any], safety_polytope_path: str):
        self.config = config
        self.safety_polytope_path = safety_polytope_path
        self.logger = logging.getLogger(__name__)

    def run(self, model_name: str, processed_data_path: str) -> str:
        """
        Train polytope constraints

        Args:
            model_name: Name of the model
            processed_data_path: Path to processed hidden state data

        Returns:
            Path to trained polytope model
        """
        self.logger.info(f"Starting polytope training for {model_name}")

        # Change to safety_polytope directory
        original_dir = os.getcwd()
        os.chdir(self.safety_polytope_path)

        try:
            # Train polytope
            model_path = self._train_polytope(model_name, processed_data_path)

            return model_path

        finally:
            os.chdir(original_dir)

    def _train_polytope(
        self, model_name: str, processed_data_path: str
    ) -> str:
        """
        Train polytope using learn_polytope.py

        Args:
            model_name: Name of the model
            processed_data_path: Path to processed data

        Returns:
            Path to trained model
        """
        self.logger.info(f"Training polytope for {model_name}")

        # Get training configuration
        training_config = self.config.get("polytope_training", {})

        # Prepare command arguments
        cmd = [
            "python",
            "src/safety_polytope/polytope/learn_polytope.py",
            f"model_name={model_name}",
            f"data_path={processed_data_path}",
            f'seed={training_config.get("seed", 42)}',
            f'num_epochs={training_config.get("num_epochs", 100)}',
            f'learning_rate={training_config.get("learning_rate", 0.01)}',
            f'entropy_weight={training_config.get("entropy_weight", 1.0)}',
            f'unsafe_weight={training_config.get("unsafe_weight", 2.0)}',
            f'margin={training_config.get("margin", 1.0)}',
        ]

        # Run training
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            self.logger.error(f"Polytope training failed: {result.stderr}")
            raise RuntimeError(f"Polytope training failed: {result.stderr}")

        self.logger.info(f"Successfully trained polytope for {model_name}")
