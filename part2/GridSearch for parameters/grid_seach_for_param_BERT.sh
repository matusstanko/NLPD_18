#!/bin/bash

#SBATCH --job-name=bert_grid
#SBATCH --output=train_%A_%a.out
#SBATCH --error=train_%A_%a.err
#SBATCH --partition=scavenge
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --array=0-59
#SBATCH --mail-type=END  

hostname
module load GCCcore/12.3.0

echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate matus_env

python Grid_search_for_param_BERT.py $SLURM_ARRAY_TASK_ID