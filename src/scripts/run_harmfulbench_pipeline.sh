#!/bin/bash
#SBATCH --job-name=harmfulbench_pipeline
#SBATCH --output=/scratch/gpfs/KOROLOVA/cl6486/SafetyPolytope/src/logs/harmfulbench_pipeline_%j.out
#SBATCH --error=/scratch/gpfs/KOROLOVA/cl6486/SafetyPolytope/src/logs/harmfulbench_pipeline_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=cl6486@princeton.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --chdir=/scratch/gpfs/KOROLOVA/cl6486/SafetyPolytope


export XDG_CACHE_HOME="/scratch/gpfs/KOROLOVA/cl6486/.cache"
export HF_TOKEN=""

uv run python src/safety_polytope/harmbench/run_harmbench_pipeline.py \
    --config src/safety_polytope/harmbench/config/pipeline_config.yaml \
    --model qwen_3_4b \
    --stages 1

echo "HarmfulBench pipeline completed"