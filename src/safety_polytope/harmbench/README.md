# HarmBench Pipeline

Minimal instructions for running the Safety Polytope HarmBench experiments pipeline.

## Prerequisites

1. **Download and Install HarmBench**
   ```bash
   git clone https://github.com/RPC2/HarmBench.git
   cd HarmBench
   pip install -e .
   ```
   This repository contains the code and scripts used for generating the paper's experiments.

2. **Configure HarmBench Path**
   Update `harmbench_path` in `config/pipeline_config.yaml` to point to your HarmBench installation.

## Pipeline Stages

The pipeline consists of 4 sequential stages:

### Stage 1: Attack Generation
Generates adversarial test cases using HarmBench's attack methods.
```bash
python src/safety_polytope/harmbench/run_harmbench_pipeline.py \
    --config src/safety_polytope/harmbench/config/pipeline_config.yaml \
    --model qwen_1.5b \
    --stages 1
```

### Stage 2: Hidden State Extraction & Processing
Extracts and processes hidden states from the model on generated attacks.
```bash
python src/safety_polytope/harmbench/run_harmbench_pipeline.py \
    --config src/safety_polytope/harmbench/config/pipeline_config.yaml \
    --model qwen_1.5b \
    --stages 2
```

### Stage 3: Polytope Training
Trains safety polytope constraints from extracted hidden states.
```bash
python src/safety_polytope/harmbench/run_harmbench_pipeline.py \
    --config src/safety_polytope/harmbench/config/pipeline_config.yaml \
    --model qwen_1.5b \
    --stages 3
```

### Stage 4: Steering Evaluation
Evaluates polytope effectiveness for safety steering.
```bash
python src/safety_polytope/harmbench/run_harmbench_pipeline.py \
    --config src/safety_polytope/harmbench/config/pipeline_config.yaml \
    --model qwen_1.5b \
    --stages 4
```

### Running Multiple Stages
Run stages sequentially:
```bash
python src/safety_polytope/harmbench/run_harmbench_pipeline.py \
    --config src/safety_polytope/harmbench/config/pipeline_config.yaml \
    --model qwen_1.5b \
    --stages 1 2 3 4
```

## Available Models
- `qwen_1.5b` - Qwen2-1.5B-Instruct
- `llama2_7b` - Llama-2-7b
- `mistral_8b` - Ministral-8B-Instruct-2410

## Important Caveats

**Current Experimental Setting:** The hyperparameters used in these experiments are optimized for **inhibiting harmful outputs**, not for generating cohesive answers. This may result in:
- Some steering results outputting nonsensical sentences
- Over-conservative safety responses

**Improving Output Quality:** Standard approaches to improve sentence quality include:
1. Training the polytope with both adversarial datasets AND standard language generation datasets
2. Using the polytope as a rejection classifier rather than for direct steering
3. Adjusting the `margin` and `unsafe_weight` parameters in the config for your specific use case
4. Reducing the `lambda_weight` and `safe_violation_weight` in Harmbench's `models.yaml` configuration

## Configuration
Edit `config/pipeline_config.yaml` to:
- Adjust model-specific hyperparameters
- Change Slurm settings for cluster execution
- Modify attack methods per model
- Configure output directories
