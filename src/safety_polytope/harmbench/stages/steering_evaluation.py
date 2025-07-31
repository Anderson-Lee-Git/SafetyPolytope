"""
Stage 5: Steering Evaluation
Evaluates polytope effectiveness through steering experiments.
Direct Python implementation replacing shell script hierarchy.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from ..utils.command_builders import CommandBuilder
from ..utils.config_utils import (
    get_harmbench_model_name,
    get_results_directory,
)
from ..utils.file_operations import FileOperationsManager
from ..utils.job_monitoring import SlurmJobMonitor


class SteeringEvaluationStage:
    """Stage 5: Evaluate polytope effectiveness with steering"""

    def __init__(self, config: Dict[str, Any], harmbench_path: str):
        self.config = config
        self.harmbench_path = Path(harmbench_path)
        self.logger = logging.getLogger(__name__)

        # Get execution mode
        steering_config = config.get("steering_evaluation", {})
        self.part_size = steering_config.get("part_size", 500)
        self.execution_mode = steering_config.get("execution_mode", "local")

        # Initialize utility modules
        self.file_ops = FileOperationsManager(config, self.harmbench_path)
        self.command_builder = CommandBuilder(config, self.harmbench_path)
        self.job_monitor = SlurmJobMonitor(config)

    def run(
        self,
        model_name: str,
        polytope_model_path: str,
        model_config: Dict[str, Any],
    ) -> str:
        """
        Evaluate polytope effectiveness through steering

        Args:
            model_name: Name of the model
            polytope_model_path: Path to trained polytope model
            model_config: Model-specific configuration

        Returns:
            Path to evaluation results
        """
        self.logger.info(f"Starting steering evaluation for {model_name}")

        try:
            # Validate inputs
            self._validate_inputs(
                model_name, polytope_model_path, model_config
            )

            # Run direct Python evaluation
            results_path = self._run_direct_evaluation(
                model_name, polytope_model_path, model_config
            )

            self.logger.info(f"Steering evaluation completed: {results_path}")
            return results_path

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

    def _run_direct_evaluation(
        self,
        model_name: str,
        polytope_model_path: str,
        model_config: Dict[str, Any],
    ) -> str:
        """
        Run steering evaluation using direct Python implementation.
        Replaces the complex shell script hierarchy.
        """
        self.logger.info(
            f"Running direct Python steering evaluation for {model_name}"
        )

        # Create results directory with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_dir = (
            self.harmbench_path
            / "results"
            / f"steering_{model_name}_{timestamp}"
        )
        results_dir.mkdir(parents=True, exist_ok=True)

        # Change to HarmBench directory
        original_dir = os.getcwd()
        os.chdir(self.harmbench_path)

        try:
            # Update HarmBench model configuration
            self._update_harmbench_model_config(
                model_name, model_config, polytope_model_path
            )

            # Get evaluation methods from steering config
            steering_config = model_config.get("steering", {})
            evaluation_methods = steering_config.get("evaluation_methods", [])

            # Phase 1: Generate completions for all evaluation methods
            completion_results, generation_job_ids = (
                self._generate_completions(
                    model_name,
                    model_config,
                    evaluation_methods,
                    results_dir,
                    timestamp,
                )
            )

            # Phase 2: Wait for ALL generation jobs and merge locally (for Slurm mode)
            if self.execution_mode == "slurm":
                # Collect all generation job IDs across all methods
                all_generation_job_ids = []
                for method_job_ids in generation_job_ids.values():
                    all_generation_job_ids.extend(method_job_ids)

                # Wait for ALL generation jobs to complete before any merging
                if all_generation_job_ids:
                    self.logger.info(
                        f"Waiting for ALL {len(all_generation_job_ids)} generation jobs to complete"
                    )
                    if not self.job_monitor.wait_for_jobs(
                        all_generation_job_ids, "generation jobs"
                    ):
                        raise RuntimeError("Some generation jobs failed")
                    self.logger.info(
                        "All generation jobs completed successfully"
                    )

                # Now perform local merging for all methods
                merged_completion_files = {}
                for method in completion_results.keys():
                    merged_file_path, part_files = completion_results[method]
                    self.logger.info(f"Merging completion parts for {method}")
                    self.file_ops.merge_completion_parts(
                        part_files, merged_file_path
                    )
                    merged_completion_files[method] = merged_file_path
                    self.logger.info(
                        f"Successfully merged completions for {method}: {merged_file_path}"
                    )
            else:
                # For local mode, extract merged file paths from completion_results
                merged_completion_files = {
                    method: path_tuple[0]
                    for method, path_tuple in completion_results.items()
                }

            # Phase 3: Evaluate completions
            evaluation_results, evaluation_job_ids = (
                self._evaluate_completions(
                    merged_completion_files, evaluation_methods, timestamp
                )
            )

            # Phase 4: Wait for evaluation jobs to complete (for Slurm mode)
            if self.execution_mode == "slurm" and evaluation_job_ids:
                self.logger.info("Waiting for evaluation jobs to complete")
                if not self.job_monitor.wait_for_jobs(
                    evaluation_job_ids, "evaluation jobs"
                ):
                    raise RuntimeError("Some evaluation jobs failed")
                self.logger.info("All evaluation jobs completed successfully")

            self.logger.info("Completed steering evaluation")

            self.logger.info(
                f"All evaluations completed. Results: {results_dir}"
            )
            return str(results_dir)

        finally:
            os.chdir(original_dir)

    def _update_harmbench_model_config(
        self,
        model_name: str,
        model_config: Dict[str, Any],
        polytope_model_path: str,
    ) -> None:
        """Update HarmBench model configuration with polytope weights."""
        config_path = (
            self.harmbench_path / "configs" / "model_configs" / "models.yaml"
        )

        # Read current config
        with open(config_path, "r") as f:
            harmbench_configs = yaml.safe_load(f)

        # Determine HarmBench model key
        harmbench_model_name = get_harmbench_model_name(model_name)

        # Update polytope weights path using the correct key structure
        if harmbench_model_name in harmbench_configs:
            if "model" not in harmbench_configs[harmbench_model_name]:
                harmbench_configs[harmbench_model_name]["model"] = {}
            harmbench_configs[harmbench_model_name]["model"][
                "polytope_weight_path"
            ] = polytope_model_path
        else:
            # Create new model configuration
            harmbench_configs[harmbench_model_name] = {
                "model": {
                    "model_name": model_config["path"],
                    "polytope_weight_path": polytope_model_path,
                }
            }

        # Write updated config
        with open(config_path, "w") as f:
            yaml.safe_dump(harmbench_configs, f, default_flow_style=False)

        self.logger.info(
            f"Updated HarmBench config for {harmbench_model_name}: {polytope_model_path}"
        )

    def _generate_completions(
        self,
        model_name: str,
        model_config: Dict[str, Any],
        evaluation_methods: List[str],
        results_base_dir: Path,
        timestamp: str,
    ) -> Tuple[Dict[str, Tuple[Path, List[Path]]], Dict[str, List[str]]]:
        """Generate model completions for steering evaluation.

        Returns:
            Tuple of (completion_file_paths, generation_job_ids)"""
        self.logger.info("Generating completions for steering evaluation")

        # Get HarmBench model name
        harmbench_model_name = get_harmbench_model_name(model_name)
        completion_results = {}
        generation_job_ids = {}

        # Process each evaluation method
        for method in evaluation_methods:
            self.logger.info(f"Processing evaluation method: {method}")

            # Get test cases path and determine model directory structure
            test_case_model = self.file_ops.get_test_case_model_name(
                model_name
            )
            test_cases_path = self.file_ops.get_test_cases_path(
                method, test_case_model
            )

            if not test_cases_path.exists():
                self.logger.warning(
                    f"Test cases not found for method {method}: {test_cases_path}"
                )
                continue

            # Set up output directories based on shell script logic
            results_dir = get_results_directory(
                self.harmbench_path, method, model_name
            )
            if method in ["DirectRequest", "HumanJailbreaks"]:
                output_dir = (
                    self.harmbench_path
                    / "slurm_logs"
                    / "generate_completions"
                    / method
                    / "default"
                )
            else:
                output_dir = (
                    self.harmbench_path
                    / "slurm_logs"
                    / "generate_completions"
                    / method
                    / harmbench_model_name
                )

            # Create directories
            output_dir.mkdir(parents=True, exist_ok=True)
            (results_dir / "completions").mkdir(parents=True, exist_ok=True)

            # Count total test cases to determine number of parts
            total_cases = self.file_ops.get_test_case_count(test_cases_path)
            num_parts = (total_cases + self.part_size - 1) // self.part_size

            self.logger.info(
                f"Total test cases: {total_cases}, Number of parts: {num_parts}"
            )

            # Generate completions for each part
            part_files = []
            method_job_ids = []
            for part in range(1, num_parts + 1):
                part_output_file = (
                    results_dir
                    / "completions"
                    / f"{harmbench_model_name}_{timestamp}_part{part}.json"
                )

                # Build generation command
                cmd = self.command_builder.build_generation_command(
                    harmbench_model_name,
                    test_cases_path,
                    part_output_file,
                    part,
                    method,
                    self.part_size,
                )

                # Execute generation
                if self.execution_mode == "local":
                    self.command_builder.run_local_generation(
                        cmd, method, part
                    )
                else:
                    job_id = self.command_builder.run_slurm_generation(
                        cmd, method, part, output_dir, timestamp
                    )
                    method_job_ids.append(job_id)

                part_files.append(part_output_file)

            # Store paths and job IDs for later processing
            completion_results[method] = (
                results_dir
                / "completions"
                / f"{harmbench_model_name}_{timestamp}_merged.json",
                part_files,
            )
            generation_job_ids[method] = method_job_ids

            if self.execution_mode == "local":
                # For local execution, merge immediately
                merged_file = completion_results[method][0]
                self.file_ops.merge_completion_parts(part_files, merged_file)
                self.logger.info(
                    f"Successfully generated completions for {method}: {merged_file}"
                )
            else:
                self.logger.info(
                    f"Submitted {len(method_job_ids)} generation jobs for {method}"
                )

        return completion_results, generation_job_ids

    def _evaluate_completions(
        self,
        merged_completion_files: Dict[str, Path],
        evaluation_methods: List[str],
        timestamp: str,
    ) -> Tuple[Dict[str, Path], List[str]]:
        """Evaluate generated completions.

        Returns:
            Tuple of (evaluation_results, evaluation_job_ids)"""
        self.logger.info(
            f"Evaluating completions for {len(merged_completion_files)} methods"
        )

        evaluation_results = {}
        evaluation_job_ids = []

        # Process each evaluation method
        for method in evaluation_methods:
            if method not in merged_completion_files:
                self.logger.warning(
                    f"No completion results found for method: {method}"
                )
                continue

            completion_file = merged_completion_files[method]
            if not completion_file.exists():
                self.logger.warning(
                    f"Completion file not found: {completion_file}"
                )
                continue

            self.logger.info(f"Evaluating completions for method: {method}")

            # Set up results directory based on shell script logic
            model_name_from_path = completion_file.parent.parent.name.split(
                "_"
            )[
                0
            ]  # Extract model from path
            results_dir = get_results_directory(
                self.harmbench_path, method, model_name_from_path
            )

            # Create results directory
            (results_dir / "results").mkdir(parents=True, exist_ok=True)

            # Build evaluation output file
            eval_output_file = (
                results_dir
                / "results"
                / f"{completion_file.stem.replace('_merged', '')}_{timestamp}.json"
            )

            # Build evaluation command using the shell script
            cmd = self.command_builder.build_evaluation_command(
                completion_file, eval_output_file
            )

            # Execute evaluation
            if self.execution_mode == "local":
                self.command_builder.run_local_evaluation(cmd, method)
            else:
                job_id = self.command_builder.run_slurm_evaluation(
                    cmd, method, timestamp
                )
                evaluation_job_ids.append(job_id)

            evaluation_results[method] = eval_output_file

            if self.execution_mode == "local":
                self.logger.info(
                    f"Successfully evaluated completions for {method}: {eval_output_file}"
                )
            else:
                self.logger.info(f"Submitted evaluation job for {method}")

        return evaluation_results, evaluation_job_ids
