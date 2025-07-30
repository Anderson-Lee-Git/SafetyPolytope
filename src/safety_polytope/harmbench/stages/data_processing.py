"""
Stage 3: Data Processing
Processes and consolidates hidden states for polytope training.
"""

import glob
import logging
import os
import re
import subprocess
from typing import Any, Dict, List, Optional

import torch


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

        # Merge the hidden states parts
        # self._merge_hidden_states_parts(model_name, attack_methods)

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

    def _get_hidden_states_directories(
        self, model_name: str, attack_methods: List[str]
    ) -> List[str]:
        """
        Get all possible hidden states directories for the given model and methods.
        Based on logic from src/safety_polytope/data/process_hb_states.py

        Args:
            model_name: Name of the model (e.g., 'qwen_1.5b')

        Returns:
            List of hidden states directory paths to check
        """
        root = os.path.join(self.harmbench_path, "results")
        directories = []

        # Methods that use "default" subdirectory
        methods_with_default = ["DirectRequest", "HumanJailbreaks"]

        for method in attack_methods:
            if method in methods_with_default:
                hidden_states_dir = os.path.join(
                    root, method, "default", "hidden_states"
                )
            else:
                hidden_states_dir = os.path.join(
                    root, method, model_name, "hidden_states"
                )

            if os.path.exists(hidden_states_dir):
                directories.append(hidden_states_dir)

        return directories

    def _merge_hidden_states_parts(
        self, model_name: str, attack_methods: Optional[List[str]] = None
    ) -> None:
        """
        Merge hidden states part files into single files.

        Args:
            model_name: Name of the model (e.g., 'qwen_1.5b')
            attack_methods: List of attack methods to process
        """
        # Get all hidden states directories for the given model and methods
        if attack_methods is None:
            # Default to common attack methods if none provided
            attack_methods = [
                "DirectRequest",
                "HumanJailbreaks",
                "AutoPrompt",
                "GCG",
                "PAIR",
                "TAP",
                "MultiStepJailbreak",
            ]

        hidden_states_dirs = self._get_hidden_states_directories(
            model_name, attack_methods
        )

        if not hidden_states_dirs:
            self.logger.info(
                f"No hidden states directories found for model {model_name}"
            )
            return

        # Process each directory
        for hidden_states_dir in hidden_states_dirs:
            self.logger.info(f"Processing directory: {hidden_states_dir}")

            # Find all part files in this directory
            part_pattern = os.path.join(hidden_states_dir, "*_part*.pth")
            part_files = glob.glob(part_pattern)

            if not part_files:
                self.logger.info(f"No part files found in {hidden_states_dir}")
                continue

            # Group part files by base pattern (model_layer, ignoring timestamp and part)
            groups: Dict[str, Dict[int, str]] = {}
            # Pattern: hidden_states_MODEL_layerN_TIMESTAMP_partN.pth
            part_regex = re.compile(
                r"(hidden_states_.+?_layer\d+)_\d+_\d+_part(\d+)\.pth$"
            )

            for part_file in part_files:
                basename = os.path.basename(part_file)
                match = part_regex.match(basename)
                if match:
                    base_pattern = match.group(
                        1
                    )  # e.g., "hidden_states_qwen_1_5b_layer20"
                    part_num = int(match.group(2))

                    if base_pattern not in groups:
                        groups[base_pattern] = {}
                    groups[base_pattern][part_num] = part_file

            # Process each group in this directory
            for base_pattern, parts_dict in groups.items():
                self._merge_parts_group(
                    hidden_states_dir, base_pattern, parts_dict
                )

    def _merge_parts_group(
        self,
        hidden_states_dir: str,
        base_pattern: str,
        parts_dict: Dict[int, str],
    ) -> None:
        """
        Merge a group of part files with the same base pattern.

        Args:
            hidden_states_dir: Directory containing the hidden states files
            base_pattern: Base pattern of the files (e.g., "hidden_states_qwen_1_5b_layer20")
            parts_dict: Dictionary mapping part number to file path
        """
        # Sort parts by number
        sorted_parts = sorted(parts_dict.items())

        if len(sorted_parts) <= 1:
            self.logger.info(
                f"Only {len(sorted_parts)} part found for {base_pattern}, no merge needed"
            )
            return

        self.logger.info(
            f"Merging {len(sorted_parts)} parts for {base_pattern}"
        )

        # Extract timestamp from part1 file
        part1_file = parts_dict.get(1)
        if not part1_file:
            self.logger.error(
                f"Part 1 not found for {base_pattern}, cannot determine timestamp"
            )
            return

        # Extract timestamp from part1 filename
        part1_basename = os.path.basename(part1_file)
        timestamp_match = re.search(
            r"_(\d{8}_\d{6})_part1\.pth$", part1_basename
        )
        if not timestamp_match:
            self.logger.error(
                f"Could not extract timestamp from part1: {part1_basename}"
            )
            return
        timestamp = timestamp_match.group(1)

        # Load and concatenate tensors
        tensors = []
        for part_num, part_file in sorted_parts:
            try:
                self.logger.info(
                    f"Loading part {part_num}: {os.path.basename(part_file)}"
                )
                tensor = torch.load(part_file, map_location="cpu")
                tensors.append(tensor)
            except Exception as e:
                self.logger.error(f"Failed to load part {part_num}: {e}")
                return

        if not tensors:
            self.logger.warning(f"No tensors loaded for {base_pattern}")
            return

        # Concatenate along the first dimension (batch dimension)
        try:
            merged_tensor = torch.cat(tensors, dim=0)
            self.logger.info(f"Merged tensor shape: {merged_tensor.shape}")
        except Exception as e:
            self.logger.error(
                f"Failed to concatenate tensors for {base_pattern}: {e}"
            )
            return

        # Create output filename using base pattern and part1's timestamp
        output_filename = f"{base_pattern}_{timestamp}.pth"
        output_file = os.path.join(hidden_states_dir, output_filename)

        # Save merged tensor
        try:
            torch.save(merged_tensor, output_file)
            self.logger.info(f"Saved merged file: {output_filename}")

            # Optionally remove original part files to save space
            # Uncomment the following lines if you want to clean up part files
            # for _, part_file in sorted_parts:
            #     os.remove(part_file)
            #     self.logger.info(f"Removed part file: {os.path.basename(part_file)}")

        except Exception as e:
            self.logger.error(f"Failed to save merged file {output_file}: {e}")

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

        self.logger.info(f"Running command: {' '.join(cmd)}")
        self.logger.info(f"Working directory: {os.getcwd()}")
        self.logger.info(f"HARMBENCH_PATH: {self.harmbench_path}")

        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=600
        )  # 10 minute timeout

        if result.returncode != 0:
            error_msg = f"Hidden state processing failed with return code {result.returncode}"
            if result.stdout:
                error_msg += f"\nSTDOUT:\n{result.stdout}"
            if result.stderr:
                error_msg += f"\nSTDERR:\n{result.stderr}"
            else:
                error_msg += "\nNo stderr output captured"

            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

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
