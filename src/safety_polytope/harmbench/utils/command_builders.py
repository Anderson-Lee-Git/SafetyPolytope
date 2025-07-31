"""
Command builders for HarmBench pipeline shell scripts.

This module provides utilities for building shell commands for generation
and evaluation tasks, including both local and Slurm execution modes.
"""

import logging
import subprocess
from pathlib import Path
from typing import List


class CommandBuilder:
    """Utility class for building and executing shell commands"""

    def __init__(self, config: dict, harmbench_path: Path):
        """
        Initialize command builder

        Args:
            config: Pipeline configuration
            harmbench_path: Path to HarmBench directory
        """
        self.config = config
        self.harmbench_path = harmbench_path
        self.logger = logging.getLogger(__name__)

        # Get configuration parameters
        steering_config = config.get("steering_evaluation", {})
        self.max_new_tokens = steering_config.get("max_new_tokens", 64)
        self.incremental_update = steering_config.get(
            "incremental_update", False
        )

        eval_config = config.get("evaluation", {})
        self.cls_path = eval_config.get(
            "cls_path", "cais/HarmBench-Llama-2-13b-cls"
        )
        self.behavior_dataset = eval_config.get(
            "behavior_dataset",
            "./data/behavior_datasets/harmbench_behaviors_text_all.csv",
        )

        # Slurm config
        slurm_config = config.get("slurm", {})
        self.slurm_partition = slurm_config.get("partition", "gpu")
        self.slurm_time_limit = slurm_config.get("time_limit", "24:00:00")

    def build_generation_command(
        self,
        harmbench_model_name: str,
        test_cases_path: Path,
        output_file: Path,
        part_number: int,
        method: str,
        part_size: int,
    ) -> List[str]:
        """Build command for generating completions."""
        cmd = [
            "scripts/generate_completions.sh",
            harmbench_model_name,
            self.behavior_dataset,
            str(test_cases_path),
            str(output_file),
            str(self.max_new_tokens),
            str(
                self.incremental_update
            ).capitalize(),  # Convert to "True"/"False"
        ]

        # Default to not saving hidden states
        cmd.append("None")

        # Add part information
        cmd.extend([str(part_size), str(part_number)])

        # Add defense configuration - disable since polytope_weights is not a valid defense method
        cmd.extend(["False", "None"])  # use_defense and defense_method

        return cmd

    def build_evaluation_command(
        self, completion_file: Path, output_file: Path
    ) -> List[str]:
        """Build command for evaluating completions."""
        cmd = [
            "scripts/evaluate_completions.sh",
            self.cls_path,
            self.behavior_dataset,
            str(completion_file),
            str(output_file),
        ]
        return cmd

    def run_local_generation(
        self, cmd: List[str], method: str, part: int
    ) -> None:
        """Run generation locally using bash."""
        self.logger.info(
            f"Running generation for {method} part {part}: {' '.join(cmd)}"
        )
        result = subprocess.run(["bash"] + cmd, capture_output=True, text=True)

        if result.returncode != 0:
            self.logger.error(
                f"Generation failed for {method} part {part}: {result.stderr}"
            )
            raise RuntimeError(
                f"Generation failed for {method} part {part}: {result.stderr}"
            )

        self.logger.info(
            f"Successfully generated completions for {method} part {part}"
        )

    def run_slurm_generation(
        self,
        cmd: List[str],
        method: str,
        part: int,
        output_dir: Path,
        timestamp: str,
    ) -> str:
        """Run generation using slurm sbatch.

        Returns:
            Job ID of submitted job"""
        job_name = f"generate_{method}_{part}"
        log_file = output_dir / f"generation_{timestamp}_part{part}.log"

        sbatch_cmd = [
            "sbatch",
            f"--partition={self.slurm_partition}",
            f"--job-name={job_name}",
            "--nodes=1",
            "--gpus-per-node=1",
            f"--time={self.slurm_time_limit}",
            f"--output={log_file}",
            "--mem-per-cpu=128G",
            "--gres=gpumem:32g",
            "--parsable",
        ] + cmd

        self.logger.info(
            f"Submitting slurm job for {method} part {part}: {' '.join(sbatch_cmd)}"
        )
        result = subprocess.run(sbatch_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            self.logger.error(
                f"Slurm submission failed for {method} part {part}: {result.stderr}"
            )
            raise RuntimeError(
                f"Slurm submission failed for {method} part {part}: {result.stderr}"
            )

        job_id = result.stdout.strip()
        self.logger.info(
            f"Submitted slurm job {job_id} for {method} part {part}"
        )

        return job_id

    def run_local_evaluation(self, cmd: List[str], method: str) -> None:
        """Run evaluation locally using bash."""
        self.logger.info(f"Running evaluation for {method}: {' '.join(cmd)}")
        result = subprocess.run(["bash"] + cmd, capture_output=True, text=True)

        if result.returncode != 0:
            self.logger.error(
                f"Evaluation failed for {method}: {result.stderr}"
            )
            raise RuntimeError(
                f"Evaluation failed for {method}: {result.stderr}"
            )

        self.logger.info(f"Successfully evaluated completions for {method}")

    def run_slurm_evaluation(
        self, cmd: List[str], method: str, timestamp: str
    ) -> str:
        """Run evaluation using slurm sbatch.

        Returns:
            Job ID of submitted job"""
        job_name = f"evaluate_{method}"
        output_dir = (
            self.harmbench_path
            / "slurm_logs"
            / "evaluate_completions"
            / method
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / f"evaluation_{timestamp}.log"

        sbatch_cmd = [
            "sbatch",
            f"--partition={self.slurm_partition}",
            f"--job-name={job_name}",
            "--nodes=1",
            "--gpus-per-node=1",
            "--time=4:00:00",  # Evaluation typically takes less time
            f"--output={log_file}",
            "--mem-per-cpu=128G",
            "--gres=gpumem:40g",
            "--parsable",
        ] + cmd

        self.logger.info(
            f"Submitting evaluation job for {method}: {' '.join(sbatch_cmd)}"
        )
        result = subprocess.run(sbatch_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            self.logger.error(
                f"Evaluation submission failed for {method}: {result.stderr}"
            )
            raise RuntimeError(
                f"Evaluation submission failed for {method}: {result.stderr}"
            )

        job_id = result.stdout.strip()
        self.logger.info(f"Submitted evaluation job {job_id} for {method}")

        return job_id
