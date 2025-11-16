#!/bin/bash
#SBATCH --job-name=increasing_subset_size_learn_polytope
#SBATCH --output=/scratch/gpfs/KOROLOVA/cl6486/SafetyPolytope/src/logs/increasing_subset_size_learn_polytope_%j.out
#SBATCH --error=/scratch/gpfs/KOROLOVA/cl6486/SafetyPolytope/src/logs/increasing_subset_size_learn_polytope_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=cl6486@princeton.edu
#SBATCH --chdir=/scratch/gpfs/KOROLOVA/cl6486/SafetyPolytope


export XDG_CACHE_HOME="/scratch/gpfs/KOROLOVA/cl6486/.cache"
export HF_TOKEN=""

uv run accelerate launch --num_processes 1 src/safety_polytope/polytope/learn_polytope.py \
    --multirun \
    dataset=beaver_tails \
    dataset.dataset_type=prompt_response \
    learning_rate=0.01 \
    phi_l1_weight=0.01 \
    margin=10.0 \
    num_feature_extractor_layers=1 \
    dataset.hidden_states_path=/scratch/gpfs/KOROLOVA/cl6486/data/data/beaver_tails/qwen2-1.5b-instruct/all_hidden_states_with_safe.pth \
    # subset_size=0.001,0.003162,0.01,0.03162,0.1 \
    seed=4821,736,5914,2087,9643 \