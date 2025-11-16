#!/bin/bash
#SBATCH --job-name=vary_label_distribution_learn_polytope
#SBATCH --output=/scratch/gpfs/KOROLOVA/cl6486/SafetyPolytope/src/logs/vary_label_distribution_learn_polytope_%j.out
#SBATCH --error=/scratch/gpfs/KOROLOVA/cl6486/SafetyPolytope/src/logs/vary_label_distribution_learn_polytope_%j.err
#SBATCH --time=6:00:00
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
    curate_label_distribution=true \
    curation.num_samples=10000 \
    curation.pos_sample_ratio=0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95 \
    dataset.hidden_states_path=/scratch/gpfs/KOROLOVA/cl6486/data/data/beaver_tails/qwen2-1.5b-instruct/all_hidden_states_with_safe.pth \
    seed=4821,736,5914,2087,9643 \