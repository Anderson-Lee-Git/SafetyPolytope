#!/bin/bash
#SBATCH --job-name=steered_generation
#SBATCH --output=/scratch/gpfs/KOROLOVA/cl6486/SafetyPolytope/src/logs/steered_generation.out
#SBATCH --error=/scratch/gpfs/KOROLOVA/cl6486/SafetyPolytope/src/logs/steered_generation.err
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

uv run accelerate launch --num_processes 1 src/safety_polytope/evaluation/generation.py \
    model_path=/scratch/gpfs/KOROLOVA/cl6486/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554 \
    guard_model_path=/scratch/gpfs/KOROLOVA/cl6486/.cache/huggingface/hub/models--Qwen--Qwen3Guard-Gen-4B/snapshots/ec9748ae4e3acd5abc74f62d93ffbd458f1ca38e \
    polytope_weight_path=/scratch/gpfs/KOROLOVA/cl6486/SafetyPolytope/outputs/beaver_tails/2025-09-25/polytope_training_all-21-06-09/weights.pth \
    dataset=beaver_tails \
    dataset.dataset_type=prompt \