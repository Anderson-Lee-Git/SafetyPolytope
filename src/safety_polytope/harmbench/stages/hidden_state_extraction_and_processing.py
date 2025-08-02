"""
Merged Stage 2&3: Hidden State Extraction and Processing
Directly calls HarmBench's save_reps.sh and process_hb_states.py for exact reproducibility.
"""

import logging
import os
import subprocess
import time
from typing import Any, Dict, List


class HiddenStateExtractionAndProcessingStage:
    """Merged stage that calls save_reps.sh and process_hb_states.py directly"""

    def __init__(
        self,
        config: Dict[str, Any],
        harmbench_path: str,
        safety_polytope_path: str,
    ):
        self.config = config
        self.harmbench_path = harmbench_path
        self.safety_polytope_path = safety_polytope_path
        self.logger = logging.getLogger(__name__)

    def run(
        self, model_name: str, attack_methods: List[str], layer_num: int = 20
    ) -> str:
        """
        Run hidden state extraction and processing using exact previous commands

        Args:
            model_name: Name of the model (e.g., 'qwen_1.5b')
            attack_methods: List of attack methods from config
            layer_num: Layer number to extract hidden states from

        Returns:
            Path to processed data file
        """
        self.logger.info(
            f"Starting hidden state extraction and processing for {model_name}"
        )
        self.logger.info(f"Attack methods: {attack_methods}")
        self.logger.info(f"Layer number: {layer_num}")

        # Step 1: Run save_reps.sh
        self._run_save_reps(model_name, attack_methods, layer_num)

        # Step 2: Run process_hb_states.py
        output_path = self._run_process_hb_states(model_name, attack_methods)

        return output_path

    def _run_save_reps(
        self, model_name: str, attack_methods: List[str], layer_num: int
    ) -> None:
        """
        Run save_reps.sh script from HarmBench

        Command: scripts/save_reps.sh qwen_1.5b sbatch AutoPrompt,DirectRequest,GBDA,GCG 20
        """
        self.logger.info("Running save_reps.sh...")

        # Change to HarmBench directory
        original_dir = os.getcwd()
        os.chdir(self.harmbench_path)

        try:
            # Prepare command
            methods_str = ",".join(attack_methods)
            execution_mode = self.config.get(
                "hidden_state_extraction", {}
            ).get("execution_mode", "sbatch")

            # Map 'slurm' to 'sbatch' for compatibility with save_reps.sh
            if execution_mode == "slurm":
                execution_mode = "sbatch"

            cmd = [
                "scripts/save_reps.sh",
                model_name,
                execution_mode,
                methods_str,
                str(layer_num),
            ]

            self.logger.info(f"Running command: {' '.join(cmd)}")

            # Run the script
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                error_msg = (
                    f"save_reps.sh failed with return code {result.returncode}"
                )
                if result.stdout:
                    error_msg += f"\nSTDOUT:\n{result.stdout}"
                if result.stderr:
                    error_msg += f"\nSTDERR:\n{result.stderr}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)

            self.logger.info("save_reps.sh completed successfully")

            # If using sbatch mode, wait for jobs to complete
            if execution_mode == "sbatch":
                self._wait_for_all_jobs_completion()

        finally:
            os.chdir(original_dir)

    def _wait_for_all_jobs_completion(self) -> None:
        """
        Wait for all SLURM jobs with name starting with "generate_save_" to complete by monitoring the queue
        """
        self.logger.info(
            "Waiting for SLURM jobs with name 'generate_save_' to complete..."
        )

        # First check if squeue is available
        squeue_check = subprocess.run(
            ["which", "squeue"], capture_output=True, text=True
        )

        if squeue_check.returncode != 0:
            self.logger.warning(
                "squeue command not found. This might be because:"
            )
            self.logger.warning("1. You're not on a SLURM cluster node")
            self.logger.warning("2. SLURM is not installed or not in PATH")
            self.logger.warning(
                "Skipping job monitoring. Please check job status manually."
            )
            # Give some time for jobs to be submitted
            time.sleep(30)
            return

        timeout_minutes = self.config.get("slurm", {}).get(
            "job_timeout_minutes", 240
        )
        check_interval = self.config.get("slurm", {}).get(
            "check_interval_seconds", 60
        )

        start_time = time.time()
        timeout_seconds = timeout_minutes * 60

        user = os.getenv("USER")
        if not user:
            raise RuntimeError("USER environment variable not set")

        while time.time() - start_time < timeout_seconds:
            # Check SLURM queue for jobs with name starting with "generate_save_"
            result = subprocess.run(
                [
                    "squeue",
                    "-u",
                    user,
                    "-h",
                    "--name",
                    "generate_save_*",
                ],  # -h for no header, filter by job name
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                self.logger.warning(
                    f"Failed to check SLURM queue: {result.stderr}"
                )
                time.sleep(check_interval)
                continue

            # Check if there are any jobs still running
            output = result.stdout.strip()
            if not output:
                self.logger.info("All 'generate_save_' SLURM jobs completed")
                # Give a bit more time for file system sync
                time.sleep(10)
                return

            # Count jobs
            job_lines = output.split("\n")
            job_count = len([line for line in job_lines if line.strip()])

            self.logger.info(
                f"Still waiting for {job_count} 'generate_save_' SLURM jobs to complete..."
            )
            time.sleep(check_interval)

        # Timeout reached
        raise TimeoutError(
            f"'generate_save_' SLURM jobs did not complete within {timeout_minutes} minutes"
        )

    def _run_process_hb_states(
        self, model_name: str, attack_methods: List[str]
    ) -> str:
        """
        Run process_hb_states.py script

        Command: python src/safety_polytope/data/process_hb_states.py
                 --methods AutoPrompt,DirectRequest,GBDA,GCG
                 --model qwen_1.5b
                 --output ./hs_data/qwen_1.5b/harmbench_processed.pt
        """
        self.logger.info("Running process_hb_states.py...")

        # Change to safety_polytope directory
        original_dir = os.getcwd()
        os.chdir(self.safety_polytope_path)

        try:
            # Prepare output path
            output_path = f"./hs_data/{model_name}/harmbench_processed.pt"

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Prepare command
            methods_str = ",".join(attack_methods)

            cmd = [
                "python",
                "src/safety_polytope/data/process_hb_states.py",
                "--root",
                os.path.join(self.harmbench_path, "results"),
                "--methods",
                methods_str,
                "--model",
                model_name,
                "--output",
                output_path,
            ]

            self.logger.info(f"Running command: {' '.join(cmd)}")

            # Run the script
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            if result.returncode != 0:
                error_msg = f"process_hb_states.py failed with return code {result.returncode}"
                if result.stdout:
                    error_msg += f"\nSTDOUT:\n{result.stdout}"
                if result.stderr:
                    error_msg += f"\nSTDERR:\n{result.stderr}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)

            self.logger.info("process_hb_states.py completed successfully")
            self.logger.info(f"Output saved to: {output_path}")

            # Validate output file
            full_output_path = os.path.join(
                self.safety_polytope_path, output_path
            )
            if not os.path.exists(full_output_path):
                raise RuntimeError(
                    f"Expected output file not found: {full_output_path}"
                )

            file_size = os.path.getsize(full_output_path)
            if file_size == 0:
                raise RuntimeError(f"Output file is empty: {full_output_path}")

            self.logger.info(
                f"Validated output file: {full_output_path} ({file_size} bytes)"
            )

            return output_path

        finally:
            os.chdir(original_dir)
