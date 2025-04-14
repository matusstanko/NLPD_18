#!/bin/bash

#SBATCH --job-name=BilSTM_GloVe_base_train
#SBATCH --output=BilSTM_GloVe_base_train.%j.out
#SBATCH --error=BilSTM_GloVe_base_train.%j.err

#SBATCH --partition=scavenge
##SBATCH --gres=gpu:a100_40gb:1 
##SBATCH --gres=gpu:v100:1      
##SBATCH --gres=gpu:rtx6000:1   

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
echo "Loaded..."
python BilSTM_GloVe_base_train.py
