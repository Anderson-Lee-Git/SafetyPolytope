#!/bin/bash
#SBATCH --job-name=save_hidden_states
#SBATCH --output=/scratch/gpfs/KOROLOVA/cl6486/SafetyPolytope/src/logs/save_hidden_states_%j.out
#SBATCH --error=/scratch/gpfs/KOROLOVA/cl6486/SafetyPolytope/src/logs/save_hidden_states_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=cl6486@princeton.edu
#SBATCH --chdir=/scratch/gpfs/KOROLOVA/cl6486/SafetyPolytope


export XDG_CACHE_HOME="/scratch/gpfs/KOROLOVA/cl6486/.cache"
export HF_TOKEN=""

uv run python src/safety_polytope/data/save_hs.py \
  dataset=beaver_tails \
  model_path=/scratch/gpfs/KOROLOVA/cl6486/.cache/huggingface/hub/models--Qwen--Qwen2-1.5B-Instruct/snapshots/ba1cf1846d7df0a0591d6c00649f57e798519da8 \
  dataset.dataset_type=prompt_response \
  iterate_categories=true \
  merge_after_generation=true \
  use_cache=true

echo "Hidden state generation completed"

