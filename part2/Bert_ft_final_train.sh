#!/bin/bash

#SBATCH --job-name=train_part2_bert
#SBATCH --output=train_part2_bert.%j.out
#SBATCH --error=train_part2_bert.%j.err
#SBATCH --partition=scavenge
#SBATCH --gres=gpu:1
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

python Bert_ft_final_train.py