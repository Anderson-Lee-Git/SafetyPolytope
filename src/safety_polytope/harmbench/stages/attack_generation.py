"""
Stage 1: Complete HarmBench Pipeline
Runs the complete HarmBench evaluation pipeline using run_pipeline.py.
"""

from loguru import logger
import os
import subprocess
from typing import Any, Dict, List


class AttackGenerationStage:
    """Stage 1: Run complete HarmBench evaluation pipeline"""

    def __init__(self, config: Dict[str, Any], harmbench_path: str):
        self.config = config
        self.harmbench_path = harmbench_path
        self.logger = logger

    def run(self, model_name: str, attack_methods: List[str]) -> None:
        """
        Run complete HarmBench evaluation pipeline using run_pipeline.py

        Args:
            model_name: Name of the model (e.g., 'qwen_1.5b')
            attack_methods: List of attack methods to use
        """
        self.logger.info(
            f"Starting complete HarmBench pipeline for {model_name} with methods {attack_methods}"
        )

        # Change to HarmBench directory
        original_dir = os.getcwd()
        os.chdir(self.harmbench_path)

        try:
            self._run_harmbench_pipeline(model_name, attack_methods)
            self._validate_pipeline_results(model_name, attack_methods)

        finally:
            os.chdir(original_dir)

    def _run_harmbench_pipeline(
        self, model_name: str, attack_methods: List[str]
    ) -> None:
        """
        Run complete HarmBench pipeline using run_pipeline.py

        Args:
            model_name: Name of the model
            attack_methods: List of attack methods
        """
        self.logger.info(
            f"Running complete HarmBench pipeline for {model_name} with methods {attack_methods}"
        )

        # Prepare arguments for run_pipeline.py
        methods_str = ",".join(attack_methods)
        attack_config = self.config.get("attack_generation", {})
        pipeline_config = self.config.get("pipeline", {})
        slurm_config = self.config.get("slurm", {})

        cmd = [
            "python",
            "./scripts/run_pipeline.py",
            "--methods",
            methods_str,
            "--models",
            model_name,
            "--step",
            "all",
            "--mode",
            attack_config.get("execution_mode", "slurm"),
            "--base_save_dir",
            pipeline_config.get("base_save_dir", "./results"),
            "--base_log_dir",
            pipeline_config.get("base_log_dir", "./slurm_logs"),
            "--behaviors_path",
            attack_config.get(
                "behavior_dataset",
                "./data/behavior_datasets/harmbench_behaviors_text_val.csv",
            ),
            "--partition",
            slurm_config.get("partition", "gpu"),
            "--cls_path",
            pipeline_config.get("cls_path", "cais/HarmBench-Llama-2-13b-cls"),
        ]

        self.logger.info(f"Executing HarmBench pipeline: {' '.join(cmd)}")

        # Run the complete pipeline
        # Stream subprocess output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            for line in process.stdout:
                self.logger.info(line.rstrip())
        finally:
            if process.stdout:
                process.stdout.close()
        returncode = process.wait()

        if returncode != 0:
            raise RuntimeError("HarmBench pipeline failed (see logs above)")

        self.logger.info(f"Successfully completed HarmBench pipeline for {model_name}")

    def _validate_pipeline_results(
        self, model_name: str, attack_methods: List[str]
    ) -> None:
        """
        Validate that HarmBench pipeline completed successfully

        Args:
            model_name: Name of the model
            attack_methods: List of attack methods
        """
        pipeline_config = self.config.get("pipeline", {})
        base_save_dir = pipeline_config.get("base_save_dir", "./results")

        for method in attack_methods:
            # Check for test cases
            test_case_path = f"{base_save_dir}/{method}/{model_name}/test_cases"
            if not os.path.exists(test_case_path):
                raise RuntimeError(
                    f"HarmBench pipeline results not found for {method}/{model_name} at {test_case_path}"
                )

            # Check for completions
            completions_path = f"{base_save_dir}/{method}/{model_name}/completions"
            if not os.path.exists(completions_path):
                raise RuntimeError(
                    f"HarmBench completions not found for {method}/{model_name} at {completions_path}"
                )

            # Check for evaluation results
            results_path = f"{base_save_dir}/{method}/{model_name}/results"
            if not os.path.exists(results_path):
                raise RuntimeError(
                    f"HarmBench evaluation results not found for {method}/{model_name} at {results_path}"
                )

            self.logger.info(
                f"Validated HarmBench pipeline results for {method}/{model_name}"
            )
