#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 6:00:00
#SBATCH --mem=10000
#SBATCH --gres=gpu:v100:1
#SBATCH -J dl2
#SBATCH -o ./out/dl2.out.%j.txt
#SBATCH -e ./out/dl2.err.%j.txt
#SBATCH --account=project_2002605
#SBATCH

module purge
module load pytorch/1.13

python chris_hw3.py "$@"
