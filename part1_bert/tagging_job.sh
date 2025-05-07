#!/bin/bash

#SBATCH --job-name=tagging
#SBATCH --output=tagging.%j.out
#SBATCH --error=tagging.%j.err
#SBATCH --partition=scavenge
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mem=32G

hostname

module load GCCcore/12.3.0

echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate matus_env

python Tagging_GPU.py