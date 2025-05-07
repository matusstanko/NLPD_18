#!/bin/bash

#SBATCH --job-name=bert_training
#SBATCH --output=bert_training.%j.out
#SBATCH --error=bert_training.%j.err
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=END  

echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate matus_env

python train_fixed-Copy1.py
