#!/usr/bin/env python3
"""
HarmBench Pipeline Orchestrator

This script orchestrates the complete HarmBench experimental pipeline:
1. Attack Generation: Generate adversarial test cases
2. Hidden State Extraction and Processing: Extract and process hidden states
3. Polytope Training: Train safety polytope constraints
4. Steering Evaluation: Evaluate polytope effectiveness

Usage:
python src/safety_polytope/harmbench/run_harmbench_pipeline.py --config src/safety_polytope/harmbench/config/pipeline_config.yaml --model qwen_1.5b --stages 2,3
"""

import argparse
import logging
import os
import sys
from typing import List, Optional

# Add the safety_polytope package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from safety_polytope.harmbench.stages.attack_generation import (  # noqa: E402
    AttackGenerationStage,
)
from safety_polytope.harmbench.stages.hidden_state_extraction_and_processing import (  # noqa: E402
    HiddenStateExtractionAndProcessingStage,
)
from safety_polytope.harmbench.stages.polytope_training import (  # noqa: E402
    PolytopeTrainingStage,
)
from safety_polytope.harmbench.stages.steering_evaluation import (  # noqa: E402
    SteeringEvaluationStage,
)
from safety_polytope.harmbench.utils.config_utils import (  # noqa: E402
    get_model_config,
    load_config,
    setup_logging,
    validate_config,
)


class HarmBenchPipeline:
    """Main pipeline orchestrator for HarmBench experiments"""

    def __init__(self, config_path: str):
        """
        Initialize pipeline

        Args:
            config_path: Path to pipeline configuration file
        """
        self.config = load_config(config_path)
        validate_config(self.config)
        setup_logging(self.config)

        self.logger = logging.getLogger(__name__)
        self.harmbench_path = self.config["pipeline"]["harmbench_path"]
        self.safety_polytope_path = self.config["pipeline"][
            "safety_polytope_path"
        ]

        # Initialize stages
        self.attack_generation = AttackGenerationStage(
            self.config, self.harmbench_path
        )
        self.hidden_state_extraction_and_processing = (
            HiddenStateExtractionAndProcessingStage(
                self.config, self.harmbench_path, self.safety_polytope_path
            )
        )
        self.polytope_training = PolytopeTrainingStage(
            self.config, self.safety_polytope_path
        )
        self.steering_evaluation = SteeringEvaluationStage(
            self.config, self.harmbench_path
        )

    def run(self, model_name: str, stages: Optional[List[int]] = None) -> None:
        """
        Run the complete pipeline for a specific model

        Args:
            model_name: Name of the model to process
            stages: List of stage numbers to run (1-4). If None, runs all stages.
        """
        if stages is None:
            stages = [1, 2, 3, 4]

        self.logger.info(
            f"Starting HarmBench pipeline for {model_name} with stages {stages}"
        )

        # Get model configuration
        model_config = get_model_config(self.config, model_name)
        attack_methods = model_config["attack_methods"]
        hidden_state_layer = model_config.get("hidden_state_layer", 20)

        self.logger.info(f"Model: {model_name}")
        self.logger.info(f"Attack methods: {attack_methods}")
        self.logger.info(f"Hidden state layer: {hidden_state_layer}")

        # Stage 1: Attack Generation
        if 1 in stages:
            self.logger.info("=== Stage 1: Attack Generation ===")
            try:
                self.attack_generation.run(model_name, attack_methods)
                self.logger.info("Stage 1 completed successfully")
            except Exception as e:
                self.logger.error(f"Stage 1 failed: {e}")
                raise

        # Stage 2: Hidden State Extraction and Processing
        processed_data_path = None
        if 2 in stages:
            self.logger.info(
                "=== Stage 2: Hidden State Extraction and Processing ==="
            )
            try:
                processed_data_path = (
                    self.hidden_state_extraction_and_processing.run(
                        model_name, attack_methods, hidden_state_layer
                    )
                )
                self.logger.info(
                    f"Stage 2 completed successfully: {processed_data_path}"
                )
            except Exception as e:
                self.logger.error(f"Stage 2 failed: {e}")
                raise

        # Stage 3: Polytope Training
        if 3 in stages:
            self.logger.info("=== Stage 3: Polytope Training ===")
            try:
                if processed_data_path is None:
                    # Assume default path if not from previous stage
                    processed_data_path = model_config["polytope_training"][
                        "hidden_states_path"
                    ]

                self.polytope_training.run(
                    model_name, processed_data_path, model_config
                )
                self.logger.info("Stage 3 completed successfully.")
            except Exception as e:
                self.logger.error(f"Stage 3 failed: {e}")
                raise

        # Stage 4: Steering Evaluation
        if 4 in stages:
            self.logger.info("=== Stage 4: Steering Evaluation ===")
            try:
                # Get polytope model path from config
                polytope_model_path = model_config["steering"][
                    "polytope_model_path"
                ]
                if polytope_model_path is None or not os.path.exists(
                    polytope_model_path
                ):
                    raise ValueError("Polytope model path not found in config")

                results_path = self.steering_evaluation.run(
                    model_name, polytope_model_path, model_config
                )
                self.logger.info(
                    f"Stage 4 completed successfully: {results_path}"
                )
            except Exception as e:
                self.logger.error(f"Stage 4 failed: {e}")
                raise

        self.logger.info(f"Pipeline completed successfully for {model_name}")

    def run_all_models(self, stages: Optional[List[int]] = None) -> None:
        """
        Run pipeline for all configured models

        Args:
            stages: List of stage numbers to run (1-4). If None, runs all stages.
        """
        models = [model["name"] for model in self.config["models"]]
        self.logger.info(f"Running pipeline for all models: {models}")

        for model_name in models:
            try:
                self.run(model_name, stages)
            except Exception as e:
                self.logger.error(f"Pipeline failed for {model_name}: {e}")
                # Continue with next model
                continue


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="HarmBench Pipeline Orchestrator"
    )
    parser.add_argument(
        "--config", required=True, help="Path to pipeline configuration file"
    )
    parser.add_argument(
        "--model",
        required=True,
        help='Model name to process, or "all" for all models',
    )
    parser.add_argument(
        "--stages",
        help="Comma-separated list of stages to run (1-4). Default: all stages",
    )

    args = parser.parse_args()

    # Parse stages
    stages = None
    if args.stages:
        try:
            stages = [int(s.strip()) for s in args.stages.split(",")]
            for stage in stages:
                if stage not in [1, 2, 3, 4]:
                    raise ValueError(f"Invalid stage number: {stage}")
        except ValueError as e:
            print(f"Error parsing stages: {e}")
            sys.exit(1)

    # Initialize pipeline
    try:
        pipeline = HarmBenchPipeline(args.config)
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        sys.exit(1)

    # Run pipeline
    try:
        if args.model.lower() == "all":
            pipeline.run_all_models(stages)
        else:
            pipeline.run(args.model, stages)
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        sys.exit(1)

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
