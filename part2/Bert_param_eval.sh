#!/bin/bash

#SBATCH --job-name=gridsearch_bert
#SBATCH --output=gridsearch_bert.%j.out
#SBATCH --error=gridsearch_bert.%j.err
#SBATCH --partition=scavenge
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --mail-type=END  

hostname
module load GCCcore/12.3.0

echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate matus_env

python Bert_param_eval.py