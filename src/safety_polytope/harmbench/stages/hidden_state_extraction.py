"""
Stage 2: Hidden State Extraction
Python implementation of save_reps.sh core functionality.
Extracts hidden states from model responses to adversarial test cases.
"""

import json
import logging
import os
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, List


class HiddenStateExtractionStage:
    """Stage 2: Extract hidden states from model responses using Python implementation"""

    def __init__(self, config: Dict[str, Any], harmbench_path: str):
        self.config = config
        self.harmbench_path = harmbench_path
        self.logger = logging.getLogger(__name__)

    def run(
        self, model_name: str, attack_methods: List[str], layer_num: int = 20
    ) -> None:
        """
        Extract hidden states from model responses using Python implementation of save_reps.sh

        Args:
            model_name: Name of the model
            attack_methods: List of attack methods
            layer_num: Layer number to extract hidden states from
        """
        self.logger.info(
            f"Starting hidden state extraction for {model_name} at layer {layer_num}"
        )

        # Change to HarmBench directory
        original_dir = os.getcwd()
        os.chdir(self.harmbench_path)

        try:
            for algorithm in attack_methods:
                self._extract_hidden_states_for_algorithm(
                    model_name, algorithm, layer_num
                )

        finally:
            os.chdir(original_dir)

    def _extract_hidden_states_for_algorithm(
        self, model_name: str, algorithm: str, layer_num: int
    ) -> None:
        """Extract hidden states for a specific algorithm using Python implementation"""

        # Determine paths based on algorithm type (replicating save_reps.sh logic)
        if algorithm in ["DirectRequest", "HumanJailbreaks"]:
            test_case_dir = f"./results/{algorithm}/default"
            results_dir = f"./results/{algorithm}/default"
        else:
            test_case_dir = f"./results/{algorithm}/{model_name}"
            results_dir = f"./results/{algorithm}/{model_name}"

        # Get test case count and calculate parts (replicating save_reps.sh logic)
        test_cases_file = f"{test_case_dir}/test_cases/test_cases.json"

        if not os.path.exists(test_cases_file):
            raise RuntimeError(f"Test cases file not found: {test_cases_file}")

        total_cases = self._get_test_case_count(test_cases_file)
        part_size = self.config.get("hidden_state_extraction", {}).get(
            "part_size", 100
        )
        num_parts = (total_cases + part_size - 1) // part_size

        self.logger.info(
            f"Processing {total_cases} test cases in {num_parts} parts for {algorithm}"
        )

        # Create hidden states directory
        hidden_states_dir = f"{results_dir}/hidden_states"
        os.makedirs(hidden_states_dir, exist_ok=True)

        # Process each part (like save_reps.sh loop)
        execution_mode = self.config.get("hidden_state_extraction", {}).get(
            "execution_mode", "slurm"
        )

        if execution_mode == "slurm":
            job_names = []
            for part in range(1, num_parts + 1):
                job_name = self._extract_hidden_states_part_slurm(
                    model_name,
                    algorithm,
                    layer_num,
                    part,
                    part_size,
                    test_cases_file,
                    results_dir,
                )
                job_names.append(job_name)

            # Wait for all SLURM jobs to complete
            self._wait_for_slurm_jobs(job_names)
        else:
            for part in range(1, num_parts + 1):
                self._extract_hidden_states_part_local(
                    model_name,
                    algorithm,
                    layer_num,
                    part,
                    part_size,
                    test_cases_file,
                    results_dir,
                )

        # Validate hidden state files
        self._validate_hidden_states(
            algorithm, model_name, results_dir, layer_num
        )

    def _get_test_case_count(self, test_cases_file: str) -> int:
        """Count total test cases in JSON file (replicating save_reps.sh logic)"""
        with open(test_cases_file, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            return len(data)
        elif isinstance(data, dict):
            return len(data.keys())
        else:
            raise ValueError(
                f"Unexpected test cases format in {test_cases_file}"
            )

    def _extract_hidden_states_part_local(
        self,
        model_name: str,
        algorithm: str,
        layer_num: int,
        part: int,
        part_size: int,
        test_cases_file: str,
        results_dir: str,
    ) -> None:
        """Extract hidden states for a single part - direct Python call to generate_completions_save.py"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        script_to_use = f"{self.harmbench_path}/generate_completions_save.py"

        cmd = [
            "python",
            "-u",
            script_to_use,
            "--model_name",
            model_name,
            "--behaviors_path",
            "./data/behavior_datasets/harmbench_behaviors_text_all.csv",
            "--test_cases_path",
            test_cases_file,
            "--save_path",
            f"{results_dir}/completions/{model_name}_{timestamp}_part{part}.json",
            "--max_new_tokens",
            str(
                self.config.get("hidden_state_extraction", {}).get(
                    "max_new_tokens", 512
                )
            ),
            "--layer_num",
            str(layer_num),
            "--save_hidden_states_path",
            f"{results_dir}/hidden_states/hidden_states_{model_name}_layer{layer_num}_{timestamp}_part{part}.pth",
            "--part_size",
            str(part_size),
            "--part_number",
            str(part),
        ]

        self.logger.info(
            f"Running local hidden state extraction for {algorithm} part {part}"
        )

        self.logger.info(f"Running command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd, cwd=self.harmbench_path, capture_output=True, text=True
        )
        if result.returncode != 0:
            self.logger.error(
                f"Hidden state extraction failed for {algorithm} part {part}: {result.stderr}"
            )
            raise RuntimeError(
                f"Hidden state extraction failed for {algorithm} part {part}: {result.stderr}"
            )

        self.logger.info(
            f"Successfully extracted hidden states for {algorithm} part {part}"
        )

    def _extract_hidden_states_part_slurm(
        self,
        model_name: str,
        algorithm: str,
        layer_num: int,
        part: int,
        part_size: int,
        test_cases_file: str,
        results_dir: str,
    ) -> str:
        """Submit hidden state extraction as SLURM job"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create job name
        job_name = f"extract_hs_{algorithm}_{model_name}_part_{part}"

        # Create log directory
        log_dir = (
            f"./slurm_logs/hidden_state_extraction/{algorithm}/{model_name}"
        )
        os.makedirs(log_dir, exist_ok=True)
        log_path = f"{log_dir}/part_{part}.log"

        # Get SLURM configuration
        slurm_config = self.config.get("slurm", {})
        partition = slurm_config.get("partition", "gpu")
        time_limit = slurm_config.get("time_limit", "24:00:00")

        # Prepare Python command as string for SLURM
        python_cmd = (
            f"python -u {self.harmbench_path}/generate_completions_save.py "
            + f"--model_name {model_name} "
            + "--behaviors_path ./data/behavior_datasets/harmbench_behaviors_text_all.csv "
            + f"--test_cases_path {test_cases_file} "
            + f"--save_path {results_dir}/completions/{model_name}_{timestamp}_part{part}.json "
            + f"--max_new_tokens {self.config.get('hidden_state_extraction', {}).get('max_new_tokens', 512)} "
            + f"--layer_num {layer_num} "
            + f"--save_hidden_states_path {results_dir}/hidden_states/hidden_states_{model_name}_layer{layer_num}_{timestamp}_part{part}.pth "
            + f"--part_size {part_size} "
            + f"--part_number {part}"
        )

        # Submit job using sbatch with direct command
        cmd = [
            "sbatch",
            f"--partition={partition}",
            f"--job-name={job_name}",
            "--nodes=1",
            "--gpus-per-node=1",
            f"--time={time_limit}",
            f"--output={log_path}",
            "--wrap",
            python_cmd,
        ]

        result = subprocess.run(
            cmd, cwd=self.harmbench_path, capture_output=True, text=True
        )

        if result.returncode != 0:
            self.logger.error(
                f"SLURM job submission failed for {algorithm} part {part}: {result.stderr}"
            )
            raise RuntimeError(
                f"SLURM job submission failed for {algorithm} part {part}"
            )

        self.logger.info(f"Submitted SLURM job: {job_name}")
        return job_name

    def _wait_for_slurm_jobs(self, job_names: List[str]) -> None:
        """Wait for SLURM jobs to complete"""
        self.logger.info(f"Waiting for SLURM jobs to complete: {job_names}")

        timeout_minutes = self.config.get("slurm", {}).get(
            "job_timeout_minutes", 30
        )
        check_interval = self.config.get("slurm", {}).get(
            "check_interval_seconds", 60
        )

        start_time = time.time()
        timeout_seconds = timeout_minutes * 60

        while time.time() - start_time < timeout_seconds:
            # Check SLURM queue
            result = subprocess.run(
                ["squeue", "-u", os.getenv("USER")],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                self.logger.warning(
                    f"Failed to check SLURM queue: {result.stderr}"
                )
                time.sleep(check_interval)
                continue

            # Check if any of our expected jobs are still running
            running_jobs = result.stdout.lower()
            jobs_still_running = []

            for job_name in job_names:
                if job_name.lower() in running_jobs:
                    jobs_still_running.append(job_name)

            if not jobs_still_running:
                self.logger.info("All hidden state extraction jobs completed")
                return

            self.logger.info(
                f"Still waiting for extraction jobs: {jobs_still_running}"
            )
            time.sleep(check_interval)

        # Timeout reached
        raise TimeoutError(
            f"Extraction jobs did not complete within {timeout_minutes} minutes: {job_names}"
        )

    def _validate_hidden_states(
        self, algorithm: str, model_name: str, results_dir: str, layer_num: int
    ) -> None:
        """Validate that hidden states were extracted successfully"""
        hidden_state_path = f"{results_dir}/hidden_states"

        if not os.path.exists(hidden_state_path):
            raise RuntimeError(
                f"Hidden states directory not found: {hidden_state_path}"
            )

        # Check for .pth files (new format)
        import glob

        hidden_files = glob.glob(
            f"{hidden_state_path}/hidden_states_{model_name}_layer{layer_num}_*.pth"
        )

        if not hidden_files:
            raise RuntimeError(
                f"No hidden state files found for {algorithm}/{model_name} in {hidden_state_path}"
            )

        # Validate file sizes (should be non-empty)
        valid_files = 0
        for hidden_file in hidden_files:
            file_size = os.path.getsize(hidden_file)
            if file_size == 0:
                self.logger.warning(f"Empty hidden state file: {hidden_file}")
            else:
                valid_files += 1

        if valid_files == 0:
            raise RuntimeError(
                f"All hidden state files are empty for {algorithm}/{model_name}"
            )

        self.logger.info(
            f"Validated hidden states for {algorithm}/{model_name}: {valid_files} valid files out of {len(hidden_files)} total"
        )
