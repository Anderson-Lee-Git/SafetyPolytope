"""
Stage 3: Data Processing
Processes and consolidates hidden states for polytope training.
"""

import logging
import os
import subprocess
from typing import Any, Dict, List


class DataProcessingStage:
    """Stage 3: Process and consolidate hidden states"""

    def __init__(self, config: Dict[str, Any], safety_polytope_path: str):
        self.config = config
        self.safety_polytope_path = safety_polytope_path
        self.harmbench_path = config["pipeline"]["harmbench_path"]
        self.logger = logging.getLogger(__name__)

    def run(self, model_name: str, attack_methods: List[str]) -> str:
        """
        Process and consolidate hidden states

        Args:
            model_name: Name of the model
            attack_methods: List of attack methods

        Returns:
            Path to processed data
        """
        self.logger.info(f"Starting data processing for {model_name}")

        # Change to safety_polytope directory
        original_dir = os.getcwd()
        os.chdir(self.safety_polytope_path)

        try:
            # Process hidden states
            processed_data_path = self._process_hidden_states(
                model_name, attack_methods
            )

            # Validate processed data
            self._validate_processed_data(processed_data_path)

            return processed_data_path

        finally:
            os.chdir(original_dir)

    def _process_hidden_states(
        self, model_name: str, attack_methods: List[str]
    ) -> str:
        """
        Process hidden states using process_hb_states.py

        Args:
            model_name: Name of the model
            attack_methods: List of attack methods

        Returns:
            Path to processed data
        """
        self.logger.info(f"Processing hidden states for {model_name}")

        # Prepare environment variables for processing
        env = os.environ.copy()
        env["HARMBENCH_PATH"] = self.harmbench_path
        env["MODEL_NAME"] = model_name
        env["ATTACK_METHODS"] = ",".join(attack_methods)

        # Run the processing script
        cmd = [
            "python",
            "src/safety_polytope/data/process_hb_states.py",
            "--root",
            self.harmbench_path + "/results",
            "--model",
            model_name,
            "--methods",
            ",".join(attack_methods),
            "--output",
            f"./hs_data/{model_name}/harmbench_processed.pt",
        ]

        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=600
        )  # 10 minute timeout

        if result.returncode != 0:
            self.logger.error(
                f"Hidden state processing failed: {result.stderr}"
            )
            raise RuntimeError(
                f"Hidden state processing failed: {result.stderr}"
            )

        self.logger.info(
            f"Successfully processed hidden states for {model_name}"
        )

        # Expected output path (this may need adjustment based on actual script behavior)
        processed_data_path = f"./hs_data/{model_name}/harmbench_processed.pt"
        return processed_data_path

    def _validate_processed_data(self, processed_data_path: str) -> None:
        """
        Validate that processed data exists and is valid

        Args:
            processed_data_path: Path to processed data file
        """
        if not os.path.exists(processed_data_path):
            raise RuntimeError(
                f"Processed data not found at: {processed_data_path}"
            )

        # Check file size
        file_size = os.path.getsize(processed_data_path)
        if file_size == 0:
            raise RuntimeError(
                f"Processed data file is empty: {processed_data_path}"
            )

        self.logger.info(
            f"Validated processed data: {processed_data_path} ({file_size} bytes)"
        )
