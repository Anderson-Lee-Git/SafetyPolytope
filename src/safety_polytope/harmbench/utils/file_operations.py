"""
File operations utilities for HarmBench pipeline.

This module provides utilities for file merging, test case counting,
and other file-related operations used in the steering evaluation stage.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

from .job_monitoring import SlurmJobMonitor


class FileOperationsManager:
    """Utility class for file operations in steering evaluation"""

    def __init__(self, config: dict, harmbench_path: Path):
        """
        Initialize file operations manager

        Args:
            config: Pipeline configuration
            harmbench_path: Path to HarmBench directory
        """
        self.config = config
        self.harmbench_path = harmbench_path
        self.logger = logging.getLogger(__name__)

        # Initialize job monitor for Slurm operations
        self.job_monitor = SlurmJobMonitor(config)

    def get_test_case_count(self, test_cases_path: Path) -> int:
        """Get total number of test cases from a JSON file."""
        with open(test_cases_path, "r") as f:
            data = json.load(f)
        return sum(len(cases) for cases in data.values())

    def get_test_case_model_name(self, model_name: str) -> str:
        """Get test case model name by removing safe_ and defense_ prefixes."""
        test_case_model = model_name
        if test_case_model.startswith("safe_"):
            test_case_model = test_case_model[5:]  # Remove "safe_" prefix
        if test_case_model.startswith("defense_"):
            test_case_model = test_case_model[8:]  # Remove "defense_" prefix
        return test_case_model

    def get_test_cases_path(self, method: str, test_case_model: str) -> Path:
        """Get test cases path for an attack method."""
        # Methods that use /default/ instead of /{model_name}/
        default_methods = ["DirectRequest", "HumanJailbreaks"]

        if method in default_methods:
            return (
                self.harmbench_path
                / "results"
                / method
                / "default"
                / "test_cases"
                / "test_cases.json"
            )
        else:
            return (
                self.harmbench_path
                / "results"
                / method
                / test_case_model
                / "test_cases"
                / "test_cases.json"
            )

    def merge_completion_parts(
        self, part_files: List[Path], output_file: Path
    ) -> None:
        """Merge completion parts into a single file."""
        input_pattern = str(part_files[0]).replace(
            "_part1.json", "_part*.json"
        )

        cmd = [
            "python",
            "merge_completion_parts.py",
            "--input_pattern",
            input_pattern,
            "--output_file",
            str(output_file),
        ]

        self.logger.info(f"Merging completion parts: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            self.logger.error(f"Part merging failed: {result.stderr}")
            raise RuntimeError(f"Part merging failed: {result.stderr}")

        self.logger.info(f"Successfully merged parts into: {output_file}")

    def merge_completions_locally(
        self,
        completion_results: Dict[str, Tuple[Path, List[Path]]],
        generation_job_ids: Dict[str, List[str]],
    ) -> Dict[str, Path]:
        """
        DEPRECATED: This method is kept for compatibility but job monitoring
        should be done at the workflow level before calling this method.

        Wait for generation jobs to complete, then merge completion parts locally.

        Args:
            completion_results: Dict mapping method to (merged_file_path, part_files)
            generation_job_ids: Dict mapping method to list of job IDs

        Returns:
            Dict mapping method to merged completion file path
        """
        merged_files = {}

        for method in completion_results.keys():
            merged_file_path, part_files = completion_results[method]
            method_job_ids = generation_job_ids.get(method, [])

            self.logger.info(
                f"Processing method {method} with {len(method_job_ids)} generation jobs"
            )

            # Wait for all generation jobs for this method to complete
            if method_job_ids:
                self.logger.info(
                    f"Waiting for generation jobs to complete for {method}"
                )
                if not self.job_monitor.wait_for_jobs(
                    method_job_ids, f"{method} generation jobs"
                ):
                    raise RuntimeError(
                        f"Generation jobs failed for method: {method}"
                    )

            # Perform local merging
            self.logger.info(f"Merging completion parts for {method}")
            self.merge_completion_parts(part_files, merged_file_path)

            merged_files[method] = merged_file_path
            self.logger.info(
                f"Successfully merged completions for {method}: {merged_file_path}"
            )

        return merged_files
