#!/bin/bash

#SBATCH -N 1

#SBATCH -p lrz-v100x2

#SBATCH --gres=gpu:1

#SBATCH --ntasks=1

#SBATCH -e logs/scd-%j.err

#SBATCH -o logs/scd-%j.out

#SBATCH --time=08:00:00

WANDB_API_KEY=

source .venv/bin/activate
srun python main.py