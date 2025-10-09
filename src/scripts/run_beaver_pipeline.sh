#!/bin/bash
#SBATCH --job-name=beaver_pipeline
#SBATCH --output=/scratch/gpfs/KOROLOVA/cl6486/SafetyPolytope/src/logs/beaver_pipeline.out
#SBATCH --error=/scratch/gpfs/KOROLOVA/cl6486/SafetyPolytope/src/logs/beaver_pipeline.err
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=cl6486@princeton.edu
#SBATCH --chdir=/scratch/gpfs/KOROLOVA/cl6486/SafetyPolytope


export XDG_CACHE_HOME="/scratch/gpfs/KOROLOVA/cl6486/.cache"
export HF_TOKEN=""

nvidia-smi

accelerate launch --num_processes 1 src/safety_polytope/polytope/run_beaver_pipeline.py \
    --model_path=/scratch/gpfs/KOROLOVA/cl6486/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee \
    --mode=local \
    --base_path=/scratch/gpfs/KOROLOVA/cl6486/data
